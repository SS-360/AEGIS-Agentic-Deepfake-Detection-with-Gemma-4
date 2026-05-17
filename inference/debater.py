"""
Multi-round debate orchestrator.
Manages the back-and-forth between Visionary and Inquisitor.
"""

from PIL import Image


class DebateRound:
    def __init__(
        self,
        round_num: int,
        agent: str,
        content: str,
        tags: dict | None = None,
    ):
        self.round_num = round_num
        self.agent = agent
        self.content = content
        self.tags = tags or {}


class DebateOrchestrator:
    """Manages multi-round adversarial debate between Visionary and Inquisitor."""

    def __init__(self, visionary, inquisitor, max_rounds: int = 3):
        self.visionary = visionary
        self.inquisitor = inquisitor
        self.max_rounds = max_rounds

    def run_debate(self, image: Image.Image) -> dict:
        """Run a multi-round debate on the given image."""
        manifest = self.visionary.analyze(image)
        manifest_text = self.visionary.format_manifest_as_text(manifest)

        debate_rounds: list[DebateRound] = []
        debate_history: list[str] = []

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n[Debate] Round {round_num}/{self.max_rounds}")

            if round_num % 2 == 1:
                inquisitor_response = self.inquisitor.generate_argument(
                    manifest_text=manifest_text,
                    debate_history=debate_history,
                    round_num=round_num,
                    max_new_tokens=512,
                )

                round_obj = DebateRound(
                    round_num=round_num,
                    agent="inquisitor",
                    content=inquisitor_response,
                )
                debate_rounds.append(round_obj)
                debate_history.append(inquisitor_response)

                print(f"  [Inquisitor] {inquisitor_response[:200]}...")

                verdict = self.inquisitor.extract_verdict(inquisitor_response)
                if (
                    verdict["verdict"] == "CONTRADICTION"
                    and verdict["confidence"] > 0.92
                    and round_num > 1
                ):
                    print(
                        f"  [Debate] Strong contradiction detected "
                        f"(conf={verdict['confidence']}), ending early"
                    )
                    break

            else:
                last_inquisitor = debate_history[-1]
                visionary_response = self.visionary.respond_text(
                    manifest_text=manifest_text,
                    inquisitor_challenge=last_inquisitor,
                    max_new_tokens=512,
                )

                round_obj = DebateRound(
                    round_num=round_num,
                    agent="visionary",
                    content=visionary_response,
                )
                debate_rounds.append(round_obj)
                debate_history.append(visionary_response)

                print(f"  [Visionary] {visionary_response[:200]}...")

        final_verdict = None
        for round_obj in reversed(debate_rounds):
            if round_obj.agent == "inquisitor":
                final_verdict = self.inquisitor.extract_verdict(round_obj.content)
                break

        return {
            "manifest": manifest,
            "manifest_text": manifest_text,
            "debate_rounds": [
                {
                    "round_num": round_item.round_num,
                    "agent": round_item.agent,
                    "content": round_item.content,
                }
                for round_item in debate_rounds
            ],
            "final_verdict": final_verdict,
        }

    def run_debate_text_only(self, manifest_text: str) -> dict:
        """Run debate when the manifest has already been generated."""
        debate_rounds: list[DebateRound] = []
        debate_history: list[str] = []

        visionary_rebuttal = """You are the Visionary. The Inquisitor has challenged your manifest.

Your original manifest:
{manifest_text}

The Inquisitor's challenge:
{inquisitor_challenge}

DEBATE RULES:
1. Respond specifically to each challenge raised
2. If the Inquisitor is wrong, explain why with physical reasoning
3. If the Inquisitor found a real issue, admit it honestly
4. Maintain consistency with your original analysis or revise if needed

Respond to the Inquisitor's challenge using this format:
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

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n[Debate] Round {round_num}/{self.max_rounds}")

            if round_num % 2 == 1:
                inquisitor_response = self.inquisitor.generate_argument(
                    manifest_text=manifest_text,
                    debate_history=debate_history,
                    round_num=round_num,
                    max_new_tokens=512,
                )

                round_obj = DebateRound(
                    round_num=round_num,
                    agent="inquisitor",
                    content=inquisitor_response,
                )
                debate_rounds.append(round_obj)
                debate_history.append(inquisitor_response)

                print(f"  [Inquisitor] {inquisitor_response[:200]}...")

                verdict = self.inquisitor.extract_verdict(inquisitor_response)
                if (
                    verdict["verdict"] == "CONTRADICTION"
                    and verdict["confidence"] > 0.92
                    and round_num > 1
                ):
                    print(
                        f"  [Debate] Strong contradiction detected "
                        f"(conf={verdict['confidence']}), ending early"
                    )
                    break

            else:
                last_inquisitor = debate_history[-1]
                rebuttal_prompt = visionary_rebuttal.format(
                    manifest_text=manifest_text,
                    inquisitor_challenge=last_inquisitor,
                )

                rebuttal_messages = [
                    {"role": "user", "content": [{"type": "text", "text": rebuttal_prompt}]}
                ]
                visionary_response = self.inquisitor.engine.generate(
                    rebuttal_messages,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                )

                round_obj = DebateRound(
                    round_num=round_num,
                    agent="visionary",
                    content=visionary_response,
                )
                debate_rounds.append(round_obj)
                debate_history.append(visionary_response)

                print(f"  [Visionary] {visionary_response[:200]}...")

        final_verdict = None
        for round_obj in reversed(debate_rounds):
            if round_obj.agent == "inquisitor":
                final_verdict = self.inquisitor.extract_verdict(round_obj.content)
                break

        return {
            "manifest_text": manifest_text,
            "debate_rounds": [
                {
                    "round_num": round_item.round_num,
                    "agent": round_item.agent,
                    "content": round_item.content,
                }
                for round_item in debate_rounds
            ],
            "final_verdict": final_verdict,
        }
