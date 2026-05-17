"""
VERDICT AGGREGATOR
Combines debate verdicts + NLI scores into a calibrated final decision.
"""

from typing import Optional


class VerdictAggregator:
    """Combine debate and NLI signals into a conservative final verdict."""

    def __init__(
        self,
        debate_weight: float = 0.25,
        nli_weight: float = 0.60,
        support_weight: float = 0.15,
        contradiction_threshold: float = 0.60,
    ):
        self.debate_weight = debate_weight
        self.nli_weight = nli_weight
        self.support_weight = support_weight
        self.contradiction_threshold = contradiction_threshold

    def aggregate(self, debate_result: dict, nli_result: dict) -> dict:
        """Combine debate and NLI results into a final verdict."""
        parsed_verdicts = self._collect_inquisitor_verdicts(debate_result)
        final_verdict = parsed_verdicts[-1] if parsed_verdicts else {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "region": None,
            "round_num": None,
        }

        contradiction_conf = max(
            [item["confidence"] for item in parsed_verdicts if item["verdict"] == "CONTRADICTION"],
            default=0.0,
        )
        consistent_conf = max(
            [item["confidence"] for item in parsed_verdicts if item["verdict"] == "CONSISTENT"],
            default=0.0,
        )

        debate_score = self._compute_debate_score(
            contradiction_conf=contradiction_conf,
            consistent_conf=consistent_conf,
            final_verdict=final_verdict["verdict"],
            final_confidence=final_verdict["confidence"],
        )

        nli_aggregate = nli_result.get("aggregate", {})
        max_contradiction = nli_aggregate.get("max_contradiction", 0.0)
        mean_entailment = nli_aggregate.get("mean_entailment", 0.0)
        mean_neutral = nli_aggregate.get("mean_neutral", 0.0)
        max_consistency_support = nli_aggregate.get("max_consistency_support", 0.0)
        mean_consistency_support = nli_aggregate.get("mean_consistency_support", 0.0)
        max_consistency_challenge = nli_aggregate.get("max_consistency_challenge", 0.0)
        max_contradiction_challenge = nli_aggregate.get("max_contradiction_challenge", 0.0)
        support_score = max(0.0, min(1.0, 1.0 - mean_neutral))

        base_score = (
            self.debate_weight * debate_score
            + self.nli_weight * max_contradiction
            + self.support_weight * support_score
        )

        groundedness_penalty = self._groundedness_penalty(
            debate_rounds=debate_result.get("debate_rounds", []),
            contradiction_conf=contradiction_conf,
            consistent_conf=consistent_conf,
            max_contradiction=max_contradiction,
            mean_entailment=mean_entailment,
            mean_neutral=mean_neutral,
            max_consistency_support=max_consistency_support,
            mean_consistency_support=mean_consistency_support,
            max_consistency_challenge=max_consistency_challenge,
            max_contradiction_challenge=max_contradiction_challenge,
        )

        final_confidence = max(0.0, min(1.0, base_score - groundedness_penalty))
        final_label = (
            "CONTRADICTION"
            if final_confidence >= self.contradiction_threshold
            else "CONSISTENT"
        )
        display_label = self._display_label(final_confidence)
        label_confidence = self._label_confidence(final_confidence, final_label)

        evidence_region = final_verdict.get("region")
        if not evidence_region:
            per_claim = nli_result.get("per_claim", [])
            if per_claim:
                best_claim = max(per_claim, key=lambda claim: claim["scores"]["contradiction"])
                evidence_region = self._extract_region_from_round(
                    debate_result.get("debate_rounds", []),
                    best_claim["round"],
                )

        cot_trace = self._build_cot_trace(debate_result.get("debate_rounds", []))

        return {
            "aegis_score": round(final_confidence, 3),
            "manipulation_risk": round(final_confidence, 3),
            "verdict": final_label,
            "display_label": display_label,
            "confidence": round(label_confidence, 3),
            "label_confidence": round(label_confidence, 3),
            "evidence_region": evidence_region,
            "debate_score": round(debate_score, 3),
            "nli_score": round(max_contradiction, 3),
            "nli_max_contradiction": round(max_contradiction, 3),
            "nli_mean_entailment": round(mean_entailment, 3),
            "nli_mean_neutral": round(mean_neutral, 3),
            "nli_max_consistency_support": round(max_consistency_support, 3),
            "nli_mean_consistency_support": round(mean_consistency_support, 3),
            "nli_max_consistency_challenge": round(max_consistency_challenge, 3),
            "nli_max_contradiction_challenge": round(max_contradiction_challenge, 3),
            "groundedness_penalty": round(groundedness_penalty, 3),
            "per_claim_scores": nli_result.get("per_claim", []),
            "cot_trace": cot_trace,
            "num_debate_rounds": len(debate_result.get("debate_rounds", [])),
        }

    def _collect_inquisitor_verdicts(self, debate_result: dict) -> list[dict]:
        import re

        verdicts = []
        for round_obj in debate_result.get("debate_rounds", []):
            if round_obj.get("agent", "").lower() != "inquisitor":
                continue

            content = round_obj.get("content", "")
            match = re.search(
                r"<verdict>\s*(CONSISTENT|CONTRADICTION)[^<]*confidence\s*([\d.]+)",
                content,
                re.IGNORECASE,
            )
            if not match:
                continue

            region_match = re.search(
                r"Region:\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]",
                content,
            )
            region = None
            if region_match:
                region = [float(region_match.group(index)) for index in range(1, 5)]

            verdicts.append(
                {
                    "round_num": round_obj.get("round_num"),
                    "verdict": match.group(1).upper(),
                    "confidence": float(match.group(2)),
                    "region": region,
                }
            )

        return verdicts

    def _compute_debate_score(
        self,
        contradiction_conf: float,
        consistent_conf: float,
        final_verdict: str,
        final_confidence: float,
    ) -> float:
        if contradiction_conf and consistent_conf:
            return max(0.0, min(1.0, 0.5 + (contradiction_conf - consistent_conf) / 2.0))
        if contradiction_conf:
            return contradiction_conf
        if consistent_conf:
            return 1.0 - consistent_conf
        if final_verdict == "CONTRADICTION":
            return final_confidence
        if final_verdict == "CONSISTENT":
            return 1.0 - final_confidence
        return 0.5

    def _groundedness_penalty(
        self,
        debate_rounds: list[dict],
        contradiction_conf: float,
        consistent_conf: float,
        max_contradiction: float,
        mean_entailment: float,
        mean_neutral: float,
        max_consistency_support: float,
        mean_consistency_support: float,
        max_consistency_challenge: float,
        max_contradiction_challenge: float,
    ) -> float:
        debate_text = " ".join(
            round_obj.get("content", "") for round_obj in debate_rounds
        ).lower()

        penalty = 0.0
        if "unknown" in debate_text:
            penalty += 0.10

        for phrase in [
            "100%",
            "physically impossible",
            "certainly fake",
            "definitely manipulated",
            "physical law violation",
        ]:
            if phrase in debate_text:
                penalty += 0.15

        if contradiction_conf >= 0.75 and max_contradiction < 0.15:
            penalty += 0.25
        elif contradiction_conf >= 0.60 and max_contradiction < 0.25:
            penalty += 0.15

        if consistent_conf >= 0.70 and max_consistency_support < 0.10:
            penalty += 0.12

        if max_consistency_challenge > 0.85 and max_contradiction < 0.20:
            penalty += 0.08

        if max_contradiction_challenge > 0.50 and max_contradiction > 0.50:
            penalty += 0.08

        if mean_neutral > 0.80:
            penalty += 0.15
        elif mean_neutral > 0.65:
            penalty += 0.08

        if contradiction_conf >= 0.60 and mean_entailment > 0.30:
            penalty += 0.08

        if consistent_conf >= 0.70 and mean_consistency_support > 0.50:
            penalty = max(0.0, penalty - 0.05)

        return min(0.85, penalty)

    def _display_label(self, confidence: float) -> str:
        if confidence >= 0.85:
            return "STRONG CONTRADICTION"
        if confidence >= self.contradiction_threshold:
            return "POSSIBLE INCONSISTENCY"
        return "NO STRONG EVIDENCE OF MANIPULATION"

    def _label_confidence(self, score: float, label: str) -> float:
        """Convert contradiction risk into confidence for the chosen label."""
        if label == "CONTRADICTION":
            if score <= self.contradiction_threshold:
                return 0.0
            return min(
                1.0,
                (score - self.contradiction_threshold)
                / (1.0 - self.contradiction_threshold),
            )

        if self.contradiction_threshold <= 0:
            return 0.0
        return min(1.0, max(0.0, (self.contradiction_threshold - score) / self.contradiction_threshold))

    def _extract_region_from_round(
        self,
        debate_rounds: list[dict],
        round_num: int,
    ) -> Optional[list[float]]:
        import re

        for round_obj in debate_rounds:
            if round_obj["round_num"] != round_num:
                continue
            match = re.search(
                r"Region:\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]",
                round_obj["content"],
            )
            if match:
                return [float(match.group(index)) for index in range(1, 5)]
        return None

    def _build_cot_trace(self, debate_rounds: list[dict]) -> str:
        trace_parts = []
        for round_obj in debate_rounds:
            agent = round_obj["agent"].upper()
            trace_parts.append(f"\n--- ROUND {round_obj['round_num']}: {agent} ---\n")
            trace_parts.append(round_obj["content"])
        return "".join(trace_parts)
