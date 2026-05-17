"""
Engine factory — selects engine based on configuration.
Swap engines via engine_type parameter without code changes.
"""

from typing import Literal, Optional


def create_engine(
    engine_type: Literal["hf", "llamacpp"],
    model_name: str,
    adapter_name: Optional[str] = None,
    **kwargs,
):
    if engine_type == "hf":
        from .hf_engine import HuggingFaceEngine

        return HuggingFaceEngine(
            model_name=model_name,
            adapter_name=adapter_name,
            **kwargs,
        )

    if engine_type == "llamacpp":
        from .llamacpp_engine import LlamaCPPEngine

        return LlamaCPPEngine(model_path=kwargs["model_path"], **kwargs)

    raise ValueError(f"Unknown engine type: {engine_type}")
