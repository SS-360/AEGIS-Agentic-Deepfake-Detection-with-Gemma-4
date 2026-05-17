"""
THE VISIONARY — Agent A
Produces a semantic manifest from an image using Gemma 4.
"""

import json

from PIL import Image


VISIONARY_SYSTEM = """You are Visionary, a forensic visual analyst.

Generate a detailed semantic manifest of the image.

Rules:
- Describe only directly observable evidence
- Avoid assumptions
- Mention uncertainty explicitly
- Natural photography may contain reflections, blur, compression artifacts,
  bounce lighting, and lens distortion

You MUST analyze:
1. Lighting direction
2. Shadow consistency
3. Facial structure
4. Hands/fingers
5. Reflections
6. Perspective
7. Background coherence
8. Texture continuity
9. Edge transitions
10. Anatomical realism
11. Object relationships
12. Image artifacts

Requirements:
- Minimum 8 observations
- Use cautious language
- Never directly claim manipulation
"""

VISIONARY_PROMPT = """Analyze this image carefully. Output ONLY a valid JSON object with exactly these fields, no markdown, no explanation:

{
  "objects": [
    {"name": "string describing the object", "position": "one of: left, right, center, top, bottom, foreground, background", "occludes": ["list of occluded objects"]}
  ],
  "lighting": {
    "primary_source": "one of: top-left, top-right, top-center, overhead, left, right, bottom, unknown",
    "shadow_directions": ["list of shadow directions like: bottom-left, bottom-right, bottom-center, left, right, none"],
    "intensity": "one of: harsh, soft, diffuse, overcast, unknown"
  },
  "depth": {
    "foreground": ["list of objects in foreground"],
    "midground": ["list of objects in midground, if any"],
    "background": ["list of objects in background"]
  },
  "perspective": {
    "vanishing_point": "one of: left, center, right, none, unknown",
    "camera_height": "one of: low, eye-level, high, unknown"
  },
  "temporal_cues": ["list of time/weather indicators like: daytime, nighttime, indoor, outdoor, summer, winter"],
  "ocr_text": ["any visible text in the image, empty list if none"]
}

Output ONLY the JSON object."""


def clean_json_response(raw: str) -> str:
    """Strip markdown fences and whitespace from JSON response."""
    raw = raw.strip()
    for fence in ["```json", "```JSON", "```"]:
        if raw.startswith(fence):
            raw = raw[len(fence) :]
            break
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


class Visionary:
    """Agent A — generates semantic manifests from images."""

    def __init__(self, engine):
        self.engine = engine

    def analyze(self, image: Image.Image) -> dict:
        """Generate a semantic manifest for an image."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VISIONARY_SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VISIONARY_PROMPT},
                ],
            },
        ]

        response = self.engine.generate(
            messages,
            max_new_tokens=500,
            temperature=0.2,
            do_sample=True,
        )

        cleaned = clean_json_response(response)

        try:
            manifest = json.loads(cleaned)
            manifest_text = json.dumps(manifest)
            if manifest_text.count(":") < 8:
                manifest["additional_observation_required"] = True
            return manifest
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            return {"raw_description": response, "parse_error": True}

    def format_manifest_as_text(self, manifest: dict) -> str:
        """Convert a manifest into readable text for debate."""
        if manifest.get("parse_error"):
            return f"Raw description: {manifest.get('raw_description', '')}"

        lines = [
            "VISIONARY'S SCENE MANIFEST:",
            f"- Primary light source: {manifest['lighting']['primary_source']}",
            f"- Shadow directions: {', '.join(manifest['lighting']['shadow_directions'])}",
            f"- Lighting intensity: {manifest['lighting']['intensity']}",
            f"- Foreground objects: {', '.join(manifest['depth']['foreground'])}",
            f"- Midground objects: {', '.join(manifest['depth']['midground'])}",
            f"- Background objects: {', '.join(manifest['depth']['background'])}",
            f"- Camera height: {manifest['perspective']['camera_height']}",
            f"- Temporal cues: {', '.join(manifest['temporal_cues'])}",
            "Objects:",
        ]
        for obj in manifest.get("objects", []):
            lines.append(f"  - {obj['name']} at {obj['position']}")
            if obj.get("occludes"):
                lines.append(f"    occludes: {', '.join(obj['occludes'])}")

        return "\n".join(lines)

    def respond_text(
        self,
        manifest_text: str,
        inquisitor_challenge: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Respond to an Inquisitor challenge textually."""
        rebuttal_prompt = f"""You are the Visionary. The Inquisitor has challenged your manifest.

Your original manifest:
{manifest_text}

The Inquisitor's challenge:
{inquisitor_challenge}

DEBATE RULES:
1. Respond specifically to each challenge raised
2. If the Inquisitor is wrong, explain why with physical reasoning
3. If the Inquisitor found a real issue, admit it honestly
4. Maintain consistency with your original analysis or revise if needed

Respond using this format:
<observe>
[Your observation of the challenge]
</observe>
<hypothesize>
[Your hypothesis about whether the challenge is valid]
</hypothesize>
<test>
[Your test — verify your claim against the physical evidence]
</test>
<verdict>
[CONSISTENT or CONTRADICTION] — confidence [0.0-1.0]
</verdict>"""

        messages = [{"role": "user", "content": [{"type": "text", "text": rebuttal_prompt}]}]
        return self.engine.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
        )
