"""
AGENTIC PIPELINE — Single-model multi-agent deepfake detection.

The same underlying model is reused for multiple roles by switching prompts:
Visionary observes, Inquisitor probes, and Self-Critique pressure-tests claims.
"""

import json
import re
from typing import Optional

from PIL import Image

from .engines.engine_factory import create_engine
from .nli_judge import NLIJudge
from .verdict_aggregator import VerdictAggregator


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between XML tags."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def contains_non_english(text: str, threshold: float = 0.1) -> bool:
    """Check whether the model drifted into non-ASCII-heavy output."""
    non_ascii = len([char for char in text if ord(char) > 127])
    total_alpha = len([char for char in text if char.isalpha()])
    if total_alpha == 0:
        return False
    return (non_ascii / total_alpha) > threshold


VISIONARY_SYSTEM_PROMPT = """You are Visionary, a forensic visual analyst.

Rules:
- Describe ONLY directly observable evidence
- Mention uncertainty explicitly
- Avoid assumptions
- Never directly claim fake or manipulated
- Natural photography may contain reflections, bounce lighting, blur, occlusion,
  compression artifacts, and lens distortion

Analyze:
1. Lighting direction
2. Shadows
3. Reflections
4. Perspective
5. Facial consistency
6. Hands/fingers
7. Texture continuity
8. Background coherence
9. Edge transitions
10. Object relationships

Requirements:
- Minimum 8 observations
- Use cautious language
- Mention ambiguity when present
"""


VISIONARY_JSON_PROMPT = """Analyze this image carefully. Output ONLY a valid JSON object with exactly these fields:

{
  "objects": [
    {"name": "string", "position": "left|right|center|top|bottom|foreground|background", "occludes": []}
  ],
  "lighting": {
    "primary_source": "top-left|top-right|top-center|overhead|left|right|bottom|unknown",
    "shadow_directions": [],
    "intensity": "harsh|soft|diffuse|overcast|unknown"
  },
  "depth": {
    "foreground": [],
    "midground": [],
    "background": []
  },
  "perspective": {
    "vanishing_point": "left|center|right|none|unknown",
    "camera_height": "low|eye-level|high|unknown"
  },
  "temporal_cues": [],
  "ocr_text": []
}

Output ONLY the JSON object. No markdown and no explanation.
"""


INQUISITOR_ROUND_FOCUS = [
    "lighting, shadows, and light direction",
    "reflections, material response, and surface coherence",
    "anatomy, geometry, edges, and background consistency",
]


class AgenticPipeline:
    """
    Single-model agentic deepfake detection pipeline.

    Each agent gets a fresh conversation with a tightly scoped system prompt to
    avoid role leakage across phases.
    """

    def __init__(
        self,
        engine_type: str = "hf",
        model_name: str = "unsloth/gemma-4-E2B-it",
        model_adapter: Optional[str] = None,
        max_debate_rounds: int = 3,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
    ):
        print("=" * 60)
        print("AEGIS AGENTIC PIPELINE - Initializing")
        print("=" * 60)

        self.engine_type = engine_type
        self.model_name = model_name
        self.model_adapter = model_adapter
        self.max_debate_rounds = max_debate_rounds

        print(f"\nLoading engine ({model_name})...")
        self.engine = create_engine(
            engine_type=engine_type,
            model_name=model_name,
            adapter_name=model_adapter,
        )

        hf_token = __import__("os").environ.get("HF_TOKEN")
        self.nli_judge = NLIJudge(model_name=nli_model_name, token=hf_token)
        self.verdict_aggregator = VerdictAggregator()

        print("\nAgentic pipeline initialized")
        print(f"  Model: {model_name}")
        print(f"  Adapter: {model_adapter or 'None'}")
        print(f"  Max debate rounds: {max_debate_rounds}")

    def _generate_response(
        self,
        system_prompt: str,
        user_message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response for a specific agent role."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ]
        return self.engine.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def analyze(self, image: Image.Image) -> dict:
        """Run the full pipeline on an image."""
        print("\n" + "=" * 60)
        print("AEGIS AGENTIC ANALYSIS")
        print("=" * 60)

        print("\n[Phase 1] Visionary analyzing image...")
        manifest = self._run_visionary(image)
        manifest_text = self._format_manifest(manifest)
        print(f"  Manifest generated: {len(manifest_text)} chars")

        print(f"\n[Phase 2] Running {self.max_debate_rounds}-round agentic debate...")
        debate_rounds = self._run_debate_rounds(manifest_text)

        print("\n[Phase 3] Running NLI scoring...")
        nli_result = self.nli_judge.score_debate(
            manifest_text=manifest_text,
            debate_rounds=debate_rounds,
        )

        print(
            f"\n[NLI Debug] Per-claim scores: "
            f"{len(nli_result.get('per_claim', []))} claims scored"
        )
        for claim in nli_result.get("per_claim", []):
            print(
                f"  Round {claim['round']}: "
                f"contradiction={claim['scores']['contradiction']:.3f}, "
                f"entailment={claim['scores']['entailment']:.3f}, "
                f"neutral={claim['scores']['neutral']:.3f}"
            )

        print("\n[Phase 4] Aggregating verdict...")
        final_result = self.verdict_aggregator.aggregate(
            debate_result={"debate_rounds": debate_rounds},
            nli_result=nli_result,
        )

        final_result["manifest"] = manifest
        final_result["manifest_text"] = manifest_text
        final_result["debate_rounds"] = debate_rounds
        final_result["cot_trace"] = self._build_cot_trace(debate_rounds)
        final_result["per_claim_scores"] = nli_result.get("per_claim", [])

        print("\n" + "=" * 60)
        print("FINAL VERDICT")
        print("=" * 60)
        print(f"  Label:      {final_result['verdict']}")
        print(f"  Display:    {final_result['display_label']}")
        print(f"  Confidence: {final_result['confidence']}")
        print(f"  Manipulation Risk: {final_result['manipulation_risk']}")
        print(f"  AEGIS Score: {final_result['aegis_score']}")
        print(f"  NLI Score:  {final_result['nli_score']}")
        print(
            "  NLI Aggregate: "
            f"max_contradiction={final_result.get('nli_max_contradiction', 'N/A')}, "
            f"mean_entailment={final_result.get('nli_mean_entailment', 'N/A')}, "
            f"mean_neutral={final_result.get('nli_mean_neutral', 'N/A')}"
        )

        return final_result

    def _run_visionary(self, image: Image.Image) -> dict:
        """Run the Visionary phase and parse the JSON manifest."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VISIONARY_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VISIONARY_JSON_PROMPT},
                ],
            },
        ]

        response = self.engine.generate(messages, max_new_tokens=500, temperature=0.2)

        if contains_non_english(response):
            print("  Non-English detected in Visionary, retrying...")
            retry_messages = list(messages)
            retry_messages[1] = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            VISIONARY_JSON_PROMPT
                            + "\n\nOUTPUT MUST BE IN ENGLISH AND VALID JSON ONLY."
                        ),
                    },
                ],
            }
            response = self.engine.generate(
                retry_messages,
                max_new_tokens=500,
                temperature=0.1,
            )

        cleaned = response.strip()
        for fence in ["```json", "```JSON", "```"]:
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
                break
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            return {"raw_description": response, "parse_error": True}

    def _build_inquisitor_system_prompt(self, round_num: int) -> str:
        """Build a safer round-specific probing prompt."""
        round_focus = INQUISITOR_ROUND_FOCUS[(round_num - 1) % len(INQUISITOR_ROUND_FOCUS)]
        return f"""You are Inquisitor, a forensic verification agent.

Focus this round on: {round_focus}

Rules:
1. Respond in ENGLISH ONLY.
2. Use ONLY information present in Visionary's Manifest.
3. Ambiguity is NOT contradiction.
4. Unknown lighting is NOT contradiction.
5. Consider natural explanations before alleging manipulation.
6. If evidence is weak or inconclusive, return CONSISTENT with lower confidence.
7. Use confidence above 0.8 ONLY for directly grounded, clearly stated
   impossibilities in the manifest.
8. Account for object position before judging shadow direction.
9. Do not use absolute phrases like "100%", "physically impossible",
   "certainly fake", or "definitely manipulated".
10. Probe exactly one grounded forensic concern this round.

OUTPUT FORMAT:
<observe>[specific observation from manifest]</observe>
<hypothesize>[forensic concern being tested]</hypothesize>
<test>[concrete test applied to manifest values]</test>
<verdict>[CONSISTENT or CONTRADICTION] — confidence [0.0-1.0]
Region: [x_min, y_min, x_max, y_max]</verdict>

STRICTLY follow the XML format. No extra text outside tags.
"""

    def _run_debate_rounds(self, manifest_text: str) -> list[dict]:
        """Run independent inquisitor rounds with critique passes."""
        debate_rounds: list[dict] = []

        for round_num in range(1, self.max_debate_rounds + 1):
            print(f"\n[Debate Round {round_num}]")
            inquisitor_context = f"Visionary's Manifest:\n{manifest_text}"

            print("  [Inquisitor probing...]")
            inquisitor_system = self._build_inquisitor_system_prompt(round_num)
            inquisitor_response = self._generate_response(
                system_prompt=inquisitor_system,
                user_message=(
                    f"{inquisitor_context}\n\n"
                    "Return the strongest grounded conclusion in the required XML format."
                ),
                max_new_tokens=700,
                temperature=0.35,
            )

            if contains_non_english(inquisitor_response):
                print("  Non-English detected, retrying with stronger English prompt...")
                inquisitor_response = self._generate_response(
                    system_prompt=inquisitor_system,
                    user_message=(
                        f"{inquisitor_context}\n\n"
                        "Output ONLY the XML format in English. Use CONSISTENT if the"
                        " evidence does not support a contradiction."
                    ),
                    max_new_tokens=700,
                    temperature=0.3,
                )

            round_obj = {
                "round_num": round_num,
                "agent": "Inquisitor",
                "content": inquisitor_response,
                "tags": self._extract_tags(inquisitor_response),
            }
            debate_rounds.append(round_obj)

            verdict = self._extract_verdict(inquisitor_response)
            if (
                verdict["verdict"] == "CONTRADICTION"
                and verdict["confidence"] > 0.92
                and round_num > 1
            ):
                print(
                    f"  Strong contradiction detected "
                    f"(conf={verdict['confidence']}), ending early"
                )
                break

            print("  [Self-Critique reviewing...]")
            critique_system = """You are Self-Critique, a forensic meta-reviewer.

Your task:
1. Check whether the Inquisitor overclaimed beyond the manifest
2. Identify missing uncertainty or natural explanations
3. Suggest one sharper grounded test for the next round

OUTPUT FORMAT:
<review>[assessment of the argument]</review>
<question>[probing follow-up question]</question>
<improvement>[how to make the claim more grounded]</improvement>

STRICTLY follow the XML format. No extra text outside tags.
"""

            critique_response = self._generate_response(
                system_prompt=critique_system,
                user_message=(
                    "Output ONLY the XML format.\n\n"
                    f"Manifest:\n{manifest_text}\n\n"
                    f"Inquisitor's Argument:\n{inquisitor_response}\n\n"
                    "Provide a grounded critique:"
                ),
                max_new_tokens=256,
                temperature=0.35,
            )

            if contains_non_english(critique_response):
                critique_response = self._generate_response(
                    system_prompt=critique_system,
                    user_message=(
                        "Output ONLY the XML format in English.\n\n"
                        f"Manifest:\n{manifest_text}\n\n"
                        f"Inquisitor's Argument:\n{inquisitor_response}\n\n"
                        "Provide critique:"
                    ),
                    max_new_tokens=256,
                    temperature=0.3,
                )

            debate_rounds.append(
                {
                    "round_num": round_num,
                    "agent": "Self-Critique",
                    "content": critique_response,
                    "tags": self._extract_critique_tags(critique_response),
                }
            )

        return debate_rounds

    def _format_manifest(self, manifest: dict) -> str:
        """Convert the Visionary JSON into readable debate context."""
        if manifest.get("parse_error"):
            return f"Raw description: {manifest.get('raw_description', '')}"

        lighting = manifest.get("lighting", {})
        depth = manifest.get("depth", {})
        perspective = manifest.get("perspective", {})

        lines = [
            "VISIONARY'S SCENE MANIFEST:",
            f"- Primary light source: {lighting.get('primary_source', 'unknown')}",
            f"- Shadow directions: {', '.join(lighting.get('shadow_directions', []))}",
            f"- Lighting intensity: {lighting.get('intensity', 'unknown')}",
            f"- Foreground objects: {', '.join(depth.get('foreground', []))}",
            f"- Midground objects: {', '.join(depth.get('midground', []))}",
            f"- Background objects: {', '.join(depth.get('background', []))}",
            f"- Camera height: {perspective.get('camera_height', 'unknown')}",
            f"- Vanishing point: {perspective.get('vanishing_point', 'unknown')}",
            f"- Temporal cues: {', '.join(manifest.get('temporal_cues', []))}",
            f"- OCR text: {', '.join(manifest.get('ocr_text', []))}",
        ]

        for obj in manifest.get("objects", []):
            name = obj.get("name", "unknown object")
            position = obj.get("position", "unknown")
            occludes = ", ".join(obj.get("occludes", []))
            lines.append(f"- Object: {name} at {position}")
            if occludes:
                lines.append(f"  occludes: {occludes}")

        return "\n".join(lines)

    def _extract_tags(self, text: str) -> dict:
        """Extract XML tags from an Inquisitor response."""
        return {
            "observe": extract_tag(text, "observe"),
            "hypothesize": extract_tag(text, "hypothesize"),
            "test": extract_tag(text, "test"),
            "verdict": extract_tag(text, "verdict"),
        }

    def _extract_critique_tags(self, text: str) -> dict:
        """Extract XML tags from a Self-Critique response."""
        return {
            "review": extract_tag(text, "review"),
            "question": extract_tag(text, "question"),
            "improvement": extract_tag(text, "improvement"),
        }

    def _extract_verdict(self, text: str) -> dict:
        """Extract verdict data from an Inquisitor response."""
        verdict_match = re.search(
            r"<verdict>\s*(CONSISTENT|CONTRADICTION)[^<]*confidence\s*([\d.]+)",
            text,
            re.IGNORECASE,
        )
        region_match = re.search(
            r"Region:\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]",
            text,
        )

        verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
        confidence = float(verdict_match.group(2)) if verdict_match else 0.0
        region = None
        if region_match:
            region = [float(region_match.group(index)) for index in range(1, 5)]

        return {"verdict": verdict, "confidence": confidence, "region": region}

    def _build_cot_trace(self, debate_rounds: list[dict]) -> str:
        """Build a readable trace from extracted round tags."""
        trace: list[str] = []
        for round_obj in debate_rounds:
            tags = round_obj.get("tags", {})
            trace.append(f"[Round {round_obj['round_num']}] {round_obj['agent']}")
            if tags.get("observe"):
                trace.append(f"  Observe: {tags['observe'][:140]}...")
            if tags.get("hypothesize"):
                trace.append(f"  Hypothesize: {tags['hypothesize'][:140]}...")
            if tags.get("test"):
                trace.append(f"  Test: {tags['test'][:140]}...")
            if tags.get("verdict"):
                trace.append(f"  Verdict: {tags['verdict']}")
        return "\n".join(trace)

    def unload(self):
        """Free GPU memory."""
        self.engine.unload()


def create_agentic_pipeline(
    model_name: str = "unsloth/gemma-4-E2B-it",
    model_adapter: Optional[str] = None,
    max_debate_rounds: int = 3,
    engine_type: str = "hf",
) -> AgenticPipeline:
    """Create an agentic pipeline with sensible defaults."""
    return AgenticPipeline(
        engine_type=engine_type,
        model_name=model_name,
        model_adapter=model_adapter,
        max_debate_rounds=max_debate_rounds,
    )
