"""
THE INQUISITOR — Agent B
Argues against the Visionary's manifest using deepfake pattern knowledge.
"""


INQUISITOR_SYSTEM = """You are Inquisitor, a forensic verification agent.

Your job:
Search for POSSIBLE inconsistencies in the manifest without overclaiming.

CRITICAL RULES:
- Only use directly observable evidence from the manifest
- Ambiguity is NOT contradiction
- Unknown lighting is NOT contradiction
- Natural images may contain complex lighting, reflections, blur, compression,
  occlusion, and imperfect perspective cues
- Consider natural explanations before alleging manipulation
- If evidence is weak, return CONSISTENT with a lower confidence
- Never use absolute phrases:
  "100%"
  "physically impossible"
  "certainly fake"
  "definitely manipulated"
- Account for object position before judging shadow direction

You must:
1. Cite specific evidence
2. Explain uncertainty
3. Focus on one grounded forensic concern per round
4. Lower confidence when evidence is weak
5. Use confidence above 0.8 only for clearly grounded contradictions

OUTPUT FORMAT:
<observe>[specific observation]</observe>
<hypothesize>[forensic concern being tested]</hypothesize>
<test>[concrete test using manifest values]</test>
<verdict>[CONSISTENT or CONTRADICTION] — confidence [0.0-1.0]
Region: [x_min, y_min, x_max, y_max]</verdict>
"""

INQUISITOR_DEBATE_PROMPT = """You are in a DEBATE with the Visionary about this image.

The Visionary claims:
{manifest_text}

DEBATE RULES:
1. You are pressure-testing the manifest, not forcing a contradiction
2. Use deepfake pattern knowledge only when it is grounded by the manifest
3. Be specific — reference exact fields in the manifest
4. If you cannot support a contradiction, say CONSISTENT
5. If you find a contradiction, provide a bounding box region

Respond in the format:
<observe>
[Your observation of the Visionary's claim]
</observe>
<hypothesize>
[Your hypothesis about what might be inconsistent]
</hypothesize>
<test>
[Your grounded test using physics or pattern knowledge]
</test>
<verdict>
[CONSISTENT or CONTRADICTION] — confidence [0.0-1.0]
Region: [x_min, y_min, x_max, y_max] (if contradiction)
</verdict>"""


class Inquisitor:
    """Agent B — probes the Visionary's manifest for contradictions."""

    def __init__(self, engine):
        self.engine = engine

    def generate_argument(
        self,
        manifest_text: str,
        debate_history: list[str] | None = None,
        round_num: int = 1,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate an argument against the Visionary's manifest."""
        if debate_history is None:
            debate_history = []

        messages = [
            {"role": "system", "content": [{"type": "text", "text": INQUISITOR_SYSTEM}]},
        ]

        context = f"DEBATE ROUND {round_num}\n\n"
        if debate_history:
            context += "Previous debate rounds:\n"
            for index, past in enumerate(debate_history):
                context += f"\n[Round {index // 2 + 1}]\n{past}\n"

        context += f"\n{INQUISITOR_DEBATE_PROMPT.format(manifest_text=manifest_text)}"
        messages.append({"role": "user", "content": [{"type": "text", "text": context}]})

        return self.engine.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            do_sample=True,
        )

    def extract_verdict(self, response: str) -> dict:
        """Parse verdict data from the Inquisitor response."""
        import re

        verdict_match = re.search(
            r"<verdict>\s*(CONSISTENT|CONTRADICTION)[^<]*confidence\s*([\d.]+)",
            response,
            re.IGNORECASE,
        )
        region_match = re.search(
            r"Region:\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]",
            response,
        )

        verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
        confidence = float(verdict_match.group(2)) if verdict_match else 0.0

        hallucination_phrases = [
            "100%",
            "physically impossible",
            "impossible lighting",
            "certainly fake",
            "definitely manipulated",
            "physical law violation",
        ]

        lower_response = response.lower()
        penalty = 0.0
        for phrase in hallucination_phrases:
            if phrase.lower() in lower_response:
                penalty += 0.15
        if "unknown" in lower_response:
            penalty += 0.10

        confidence = max(0.0, confidence - penalty)
        region = None
        if region_match:
            region = [float(region_match.group(index)) for index in range(1, 5)]

        return {
            "verdict": verdict,
            "confidence": confidence,
            "region": region,
            "raw_response": response,
        }
