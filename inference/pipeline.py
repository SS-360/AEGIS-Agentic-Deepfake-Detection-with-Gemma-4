"""
END-TO-END PIPELINE
Complete Aegis deepfake detection pipeline.
"""

from typing import Optional

from PIL import Image

from .debater import DebateOrchestrator
from .engines.engine_factory import create_engine
from .inquisitor import Inquisitor
from .nli_judge import NLIJudge
from .verdict_aggregator import VerdictAggregator
from .visionary import Visionary


class AegisPipeline:
    """
    Complete end-to-end pipeline for adversarial deepfake detection.

    Design: all engines are swappable via config.
    """

    def __init__(
        self,
        visionary_engine_type: str = "hf",
        visionary_model: str = "unsloth/gemma-4-E4B-it",
        inquisitor_engine_type: str = "hf",
        inquisitor_model: str = "unsloth/gemma-4-E2B-it",
        inquisitor_adapter: Optional[str] = None,
        max_debate_rounds: int = 3,
        sequential_loading: bool = True,
    ):
        print("=" * 60)
        print("AEGIS PIPELINE — Initializing")
        print("=" * 60)

        self.visionary_model = visionary_model
        self.inquisitor_model = inquisitor_model
        self.inquisitor_adapter = inquisitor_adapter
        self.visionary_engine_type = visionary_engine_type
        self.inquisitor_engine_type = inquisitor_engine_type
        self.max_debate_rounds = max_debate_rounds
        self.sequential_loading = sequential_loading

        self.visionary = None
        self.inquisitor = None
        self.nli_judge = None
        self.debate_orchestrator = None
        self.verdict_aggregator = VerdictAggregator()

        print("\nLoading NLI Judge (cross-encoder/nli-deberta-v3-base)...")
        import os

        hf_token = os.environ.get("HF_TOKEN")
        self.nli_judge = NLIJudge(token=hf_token)

        print("\nPipeline initialized (models will be loaded on demand)")
        print(f"  Visionary: {visionary_model} (loaded on first analyze)")
        print(f"  Inquisitor: {inquisitor_model} (loaded when needed)")
        print(f"  Debate rounds: {max_debate_rounds}")
        print(f"  Sequential loading: {sequential_loading}")

    def _load_visionary(self):
        if self.visionary is None:
            print(f"\nLoading Visionary ({self.visionary_model})...")
            visionary_engine = create_engine(
                engine_type=self.visionary_engine_type,
                model_name=self.visionary_model,
            )
            self.visionary = Visionary(engine=visionary_engine)

    def _load_inquisitor(self):
        if self.inquisitor is None:
            print(f"\nLoading Inquisitor ({self.inquisitor_model})...")
            inquisitor_engine = create_engine(
                engine_type=self.inquisitor_engine_type,
                model_name=self.inquisitor_model,
                adapter_name=self.inquisitor_adapter,
            )
            self.inquisitor = Inquisitor(engine=inquisitor_engine)

    def _init_orchestrator(self):
        if self.debate_orchestrator is None and self.inquisitor is not None:
            self.debate_orchestrator = DebateOrchestrator(
                visionary=self.visionary,
                inquisitor=self.inquisitor,
                max_rounds=self.max_debate_rounds,
            )

    def _unload_visionary(self):
        if self.visionary is not None:
            print("\nUnloading Visionary...")
            self.visionary.engine.unload()
            self.visionary = None

    def _unload_inquisitor(self):
        if self.inquisitor is not None:
            print("\nUnloading Inquisitor...")
            self.inquisitor.engine.unload()
            self.inquisitor = None
            self.debate_orchestrator = None

    def analyze(self, image: Image.Image) -> dict:
        """
        Run the complete pipeline on an image.
        Returns a structured verdict with trace and evidence region.
        """
        print("\n" + "=" * 60)
        print("AEGIS ANALYZE")
        print("=" * 60)

        manifest = None
        manifest_text = None

        if self.sequential_loading:
            self._load_visionary()
            print("\n[Step 1] Visionary analyzing image...")

            if image.mode != "RGB":
                image = image.convert("RGB")

            manifest = self.visionary.analyze(image)
            manifest_text = self.visionary.format_manifest_as_text(manifest)

            self._unload_visionary()
            self._load_inquisitor()
            self._init_orchestrator()

            print("\n[Step 2] Running adversarial debate...")
            debate_result = self.debate_orchestrator.run_debate_text_only(
                manifest_text=manifest_text,
            )

            self._unload_inquisitor()
        else:
            self._load_visionary()
            self._load_inquisitor()
            self._init_orchestrator()

            print("\n[Step 1] Running adversarial debate...")
            debate_result = self.debate_orchestrator.run_debate(image)
            manifest = debate_result.get("manifest", {})
            manifest_text = debate_result.get("manifest_text", "")

        print("\n[Step 3] Running NLI scoring...")
        nli_result = self.nli_judge.score_debate(
            manifest_text=manifest_text,
            debate_rounds=debate_result["debate_rounds"],
        )

        print("\n[Step 4] Aggregating verdict...")
        final_result = self.verdict_aggregator.aggregate(
            debate_result=debate_result,
            nli_result=nli_result,
        )

        final_result["manifest"] = manifest
        final_result["debate_rounds"] = debate_result["debate_rounds"]

        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        print(f"  Label:      {final_result['verdict']}")
        print(f"  Display:    {final_result['display_label']}")
        print(f"  Confidence: {final_result['confidence']}")
        print(f"  Manipulation Risk: {final_result['manipulation_risk']}")
        print(f"  AEGIS Score: {final_result['aegis_score']}")
        print(f"  Evidence:   {final_result['evidence_region']}")
        print(f"  Debate:     {final_result['num_debate_rounds']} rounds")
        print(f"  NLI Score:  {final_result['nli_score']}")

        return final_result

    def unload(self):
        """Free all GPU memory."""
        self._unload_visionary()
        self._unload_inquisitor()
