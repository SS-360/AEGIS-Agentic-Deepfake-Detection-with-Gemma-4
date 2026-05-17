"""
THE NLI JUDGE — DeBERTa-v3-base cross-encoder.
Scores each debate claim for contradiction, entailment, or neutral support.
"""

import os
import re
from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder


class NLIJudge:
    """DeBERTa-v3-based NLI scoring for debate claims."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        token: Optional[str] = None,
    ):
        print(f"[NLIJudge] Loading {model_name}...")
        if token is None:
            token = os.environ.get("HF_TOKEN")
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            trust_remote_code=True,
            token=token,
        )
        print("[NLIJudge] Loaded on CPU")

    def score_claim(self, premise: str, hypothesis: str) -> dict:
        """
        Score the relationship between premise and hypothesis.

        Label order for cross-encoder/nli-deberta-v3-base:
        0=contradiction, 1=entailment, 2=neutral
        """
        raw = self.model.predict([[premise, hypothesis]], show_progress_bar=False)

        logits = np.array(raw[0], dtype=np.float64)
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        return {
            "contradiction": float(probs[0]),
            "entailment": float(probs[1]),
            "neutral": float(probs[2]),
        }

    def score_debate(self, manifest_text: str, debate_rounds: list) -> dict:
        """Score all Inquisitor claims in the debate."""
        claim_scores = []

        for round_obj in debate_rounds:
            if round_obj["agent"].lower() != "inquisitor":
                continue

            content = round_obj["content"]
            observe = self._extract_tag(content, "observe") or ""
            hypothesize = self._extract_tag(content, "hypothesize") or ""
            test = self._extract_tag(content, "test") or ""
            verdict_text = self._extract_tag(content, "verdict") or ""

            is_contradiction = "CONTRADICTION" in verdict_text.upper()

            if is_contradiction:
                parts = [part for part in [observe, hypothesize] if len(part.strip()) > 10]
            else:
                parts = [part for part in [hypothesize, test] if len(part.strip()) > 10]

            hypothesis = " ".join(parts).strip()

            if len(hypothesis) < 20:
                hypothesis = self._extract_between_tags(content, "observe", None) or ""

            if len(hypothesis.strip()) < 20:
                continue

            score = self.score_claim(manifest_text, hypothesis)
            support_score = (
                score["contradiction"] if is_contradiction else score["entailment"]
            )
            challenge_score = (
                score["entailment"] if is_contradiction else score["contradiction"]
            )
            claim_scores.append(
                {
                    "round": round_obj["round_num"],
                    "hypothesis": hypothesis,
                    "scores": score,
                    "verdict_label": (
                        "CONTRADICTION" if is_contradiction else "CONSISTENT"
                    ),
                    "support_score": support_score,
                    "challenge_score": challenge_score,
                }
            )

        contradiction_claims = [
            item for item in claim_scores if item["verdict_label"] == "CONTRADICTION"
        ]
        consistent_claims = [
            item for item in claim_scores if item["verdict_label"] == "CONSISTENT"
        ]

        if claim_scores:
            mean_entailment = (
                sum(item["scores"]["entailment"] for item in claim_scores)
                / len(claim_scores)
            )
            mean_neutral = (
                sum(item["scores"]["neutral"] for item in claim_scores)
                / len(claim_scores)
            )
        else:
            mean_entailment = mean_neutral = 0.0

        max_contradiction = max(
            (item["support_score"] for item in contradiction_claims),
            default=0.0,
        )
        mean_contradiction_support = (
            sum(item["support_score"] for item in contradiction_claims)
            / len(contradiction_claims)
            if contradiction_claims
            else 0.0
        )
        max_consistency_support = max(
            (item["support_score"] for item in consistent_claims),
            default=0.0,
        )
        mean_consistency_support = (
            sum(item["support_score"] for item in consistent_claims)
            / len(consistent_claims)
            if consistent_claims
            else 0.0
        )
        max_consistency_challenge = max(
            (item["challenge_score"] for item in consistent_claims),
            default=0.0,
        )
        max_contradiction_challenge = max(
            (item["challenge_score"] for item in contradiction_claims),
            default=0.0,
        )

        return {
            "per_claim": claim_scores,
            "aggregate": {
                "max_contradiction": max_contradiction,
                "mean_entailment": mean_entailment,
                "mean_neutral": mean_neutral,
                "mean_contradiction_support": mean_contradiction_support,
                "max_consistency_support": max_consistency_support,
                "mean_consistency_support": mean_consistency_support,
                "max_consistency_challenge": max_consistency_challenge,
                "max_contradiction_challenge": max_contradiction_challenge,
            },
        }

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_between_tags(
        self,
        text: str,
        start_tag: str,
        end_tag: Optional[str],
    ) -> Optional[str]:
        if end_tag:
            match = re.search(rf"<{start_tag}>(.*?)</{end_tag}>", text, re.DOTALL)
        else:
            match = re.search(rf"<{start_tag}>(.*?)$", text, re.DOTALL)
        return match.group(1).strip() if match else None
