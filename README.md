# AEGIS

**Agentic deepfake detection with Gemma 4**

AEGIS is a **deepfake-analysis prototype** built for the **Gemma 4 Good Hackathon** and focused on the **Safety & Trust** problem space. Instead of treating image authenticity as a single black-box classification problem, AEGIS breaks the task into a **reasoning pipeline**:

- a **Visionary** agent observes the image and produces a structured scene manifest
- an **Inquisitor** agent probes that manifest for physical inconsistencies
- a **Self-Critique** step challenges weak arguments
- an **NLI Judge** checks whether contradiction claims are actually supported by the observed evidence

The result is a more **interpretable** and **conservative** authenticity-analysis workflow that prioritizes grounded reasoning over unsupported certainty.

## Why AEGIS

Deepfake detection is a trust-and-safety problem where a raw score is often not enough. If a system claims an image is manipulated, users need to know:

- what the model observed
- what it tested
- whether the contradiction claim is actually grounded

AEGIS is designed around that idea. Its goal is not only to say **fake** or **real**, but to expose a reasoning trail that can be inspected and challenged.

## How It Works

AEGIS uses **Gemma 4 E2B** as a **single-model multi-agent engine** through prompt switching.

### Pipeline

1. **Image Input**
   The pipeline starts from a single RGB image.

2. **Visionary**
   Gemma 4 extracts a structured scene manifest covering objects, lighting, shadows, depth, perspective, and related visual cues.

3. **Inquisitor**
   The same base model, under a different prompt, tests the manifest for forensic inconsistencies such as lighting mismatch, reflection errors, anatomical issues, and scene incoherence.

4. **Self-Critique**
   A review pass pressure-tests the argument and pushes the system away from overclaiming on ambiguous evidence.

5. **NLI Judge**
   A DeBERTa cross-encoder evaluates whether contradiction claims are actually supported by the manifest.

6. **Verdict Aggregation**
   Debate signals, contradiction support, and groundedness penalties are combined into a final conservative manipulation-risk verdict.

### Simplified Flow

```text
Image
  -> Visionary manifest
  -> Inquisitor contradiction probe
  -> Self-Critique review
  -> NLI grounding
  -> Final verdict
```

## Training and Inference

AEGIS combines **Unsloth-based fine-tuning** with a modular **Hugging Face inference pipeline**.

### Fine-tuning

- **SFT adapter**
  Trained with **Unsloth** to teach the model a structured forensic reasoning format.

- **GRPO adapter**
  Trained with **Unsloth** to optimize contradiction-focused reasoning behavior.

### Inference

- **Main demo pipeline**
  Runs with **Hugging Face Transformers**, **PEFT adapters**, and a **DeBERTa NLI judge**.

- **Deployment variant**
  A separate **4-bit GGUF** model is available for lightweight local inference with `llama.cpp`-compatible runtimes.

## Current Status

AEGIS is a **working prototype**, not a benchmark-validated final detector.

What is already working:

- end-to-end Kaggle inference
- multi-stage reasoning flow
- adapter-based Gemma 4 inference
- NLI-grounded contradiction filtering
- conservative verdict calibration for benign images

What is still limited:

- the contradiction agent is still noisy
- the NLI stage currently carries much of the reliability burden
- no held-out benchmark evaluation is included in this repo

The strongest way to understand AEGIS is as an **interpretable trust-and-safety prototype**, not as a production-ready detector.

## Repository Contents

This workspace currently contains the inference and submission assets used for the hackathon demo:

```text
.
├── inference/
│   ├── agentic_pipeline.py
│   ├── debater.py
│   ├── inquisitor.py
│   ├── nli_judge.py
│   ├── pipeline.py
│   ├── verdict_aggregator.py
│   ├── visionary.py
│   └── engines/
│       ├── engine_factory.py
│       └── hf_engine.py
├── notebooks/
│   └── aegis-kaggle-runner.ipynb
└── README.md
```

## Quick Start

### Kaggle

Use the notebook in:

- `notebooks/aegis-kaggle-runner.ipynb`

The notebook:

- installs the required inference dependencies
- loads `HF_TOKEN` from Kaggle Secrets
- finds the local project root automatically
- initializes the AEGIS pipeline
- runs inference on a local image path or URL
- prints the verdict, reasoning trace, and per-claim NLI scores

### Python Example

```python
import os
import sys
from PIL import Image

os.environ["TORCH_FLEX_ATTENTION"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, "/path/to/project")

from inference.agentic_pipeline import create_agentic_pipeline

pipeline = create_agentic_pipeline(
    model_name="unsloth/gemma-4-E2B-it",
    model_adapter="SamSankar/aegis-gemma4-e2b-grpo-v2",
    max_debate_rounds=3,
    engine_type="hf",
)

img = Image.open("image.jpg").convert("RGB")
result = pipeline.analyze(img)

print(result["verdict"])
print(result["display_label"])
print(result["manipulation_risk"])

pipeline.unload()
```

## Installation

For the current inference pipeline, use a **Gemma 4 compatible Transformers version**.

```bash
pip install \
  "transformers>=5.5.3" \
  "sentence-transformers>=2.6.0" \
  "bitsandbytes>=0.43.0" \
  "peft>=0.10.0" \
  "accelerate>=0.27.0" \
  "pillow" \
  "requests"
```

Recommended environment variables:

```bash
export HF_TOKEN="your_huggingface_token"
export TORCH_FLEX_ATTENTION="0"
export TOKENIZERS_PARALLELISM="false"
```

## Project Assets

### Kaggle

- **Demo notebook**  
  <https://www.kaggle.com/code/samsankarp/aegis-agentic-deepfake-detection-with-gemma-4>

- **Inference dataset**  
  <https://www.kaggle.com/datasets/samsankarp/aegis-inference>

### Hugging Face

- **SFT adapter**  
  <https://huggingface.co/SamSankar/aegis-gemma4-e2b-sft-v2>

- **GRPO adapter**  
  <https://huggingface.co/SamSankar/aegis-gemma4-e2b-grpo-v2>

- **Merged model**  
  <https://huggingface.co/SamSankar/aegis-gemma4-e2b-merged>

- **4-bit GGUF model**  
  <https://huggingface.co/SamSankar/aegis-gemma4-e2b-merged-Q4_K_M-GGUF>

- **SFT dataset**  
  <https://huggingface.co/datasets/SamSankar/aegis-synthetic-sft-v2>

## Submission Materials

The repo also includes:

- `submission/aegis-video-explainer.html`
  A polished explainer page for a quick screen-recorded demo video.

- `submission/writeup-assets.md`
  A helper file with upload guidance, media gallery ideas, and writeup asset planning.

## Limitations

AEGIS should be presented carefully and honestly.

- It is a **prototype**
- It is **not benchmark-validated**
- It is **not production-ready**
- Its biggest value today is **interpretable multimodal reasoning** for authenticity analysis

## Why This Fits Gemma 4 Good

- It addresses **Safety & Trust** directly
- It uses **Gemma 4 for multimodal reasoning**, not simple chat
- It demonstrates **agentic orchestration** in a meaningful real-world problem
- It uses **Unsloth** for efficient SFT and GRPO fine-tuning
- It prioritizes **interpretable outputs and safer calibration** over unsupported certainty

## Acknowledgment

Built by **Sam Sankar P** for the **Gemma 4 Good Hackathon**.
