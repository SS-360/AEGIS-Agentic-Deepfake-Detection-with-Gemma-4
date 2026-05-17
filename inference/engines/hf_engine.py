"""
HuggingFace Transformers engine for Gemma 4 models.
Handles multimodal input (image + text) for both e2b and e4b.
"""

import gc
import os
from typing import Optional

import torch
from PIL import Image

os.environ["TORCH_FLEX_ATTENTION"] = "0"


class HuggingFaceEngine:
    """Unified engine for Gemma 4 models via HuggingFace."""

    def __init__(
        self,
        model_name: str,
        adapter_name: Optional[str] = None,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.is_vision_model = "E4B" in model_name or "E2B" in model_name
        self._processor = None

        self._clear_cuda()
        print(f"[HFEngine] Loading {model_name}...")

        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        auto_model_cls = (
            AutoModelForImageTextToText if self.is_vision_model else AutoModelForCausalLM
        )

        self.model = auto_model_cls.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float32,
            device_map={"": device},
            attn_implementation="eager",
        )

        self._raw_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._processor = AutoProcessor.from_pretrained(model_name)

        if adapter_name:
            print(f"[HFEngine] Loading adapter: {adapter_name}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_name,
                is_trainable=False,
            )
            print("[HFEngine] Adapter loaded")

        self.model.eval()
        self._clear_cuda()
        print(f"[HFEngine] Loaded {model_name}")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a chat template."""
        has_images = any(
            isinstance(content, dict) and content.get("type") == "image"
            for message in messages
            for content in message.get("content", [])
            if isinstance(content, dict)
        )

        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)

        if has_images:
            import base64
            import io

            def pil_to_data_url(img: Image.Image) -> str:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"

            rebuilt_messages = []
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", [])
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if isinstance(item, str):
                            new_content.append({"type": "text", "text": item})
                        elif isinstance(item, dict) and item.get("type") == "image":
                            image = item.get("image")
                            if isinstance(image, Image.Image):
                                new_content.append(
                                    {"type": "image", "url": pil_to_data_url(image)}
                                )
                            else:
                                new_content.append(item)
                        else:
                            new_content.append(item)
                    rebuilt_messages.append({"role": role, "content": new_content})
                else:
                    rebuilt_messages.append(
                        {
                            "role": role,
                            "content": [{"type": "text", "text": str(content)}],
                        }
                    )

            inputs = self._processor.apply_chat_template(
                rebuilt_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        else:
            text = self._raw_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if not isinstance(text, str) or not text.strip():
                text = self._build_text_from_messages(messages)
            inputs = self._raw_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        inputs = {
            key: (
                value.to(self.device).to(torch.float32)
                if key == "pixel_values"
                else value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in inputs.items()
        }

        self._clear_cuda()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        input_ids = inputs.get("input_ids")
        result = self._raw_tokenizer.decode(
            outputs[0][input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        del outputs, inputs
        self._clear_cuda()
        return result

    def _build_text_from_messages(self, messages: list[dict]) -> str:
        text_parts: list[str] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if isinstance(content, str):
                text_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
        return " ".join(text_parts) if text_parts else "Describe this image."

    def _clear_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def unload(self):
        del self.model
        if self._processor:
            del self._processor
        if hasattr(self, "_raw_tokenizer"):
            del self._raw_tokenizer
        self._processor = None
        self._clear_cuda()
        print(f"[HFEngine] Unloaded {self.model_name}")

    def memory_stats(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(
                f"[HFEngine] GPU memory: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved"
            )
