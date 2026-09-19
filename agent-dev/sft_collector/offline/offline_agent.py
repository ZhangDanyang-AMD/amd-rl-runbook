"""Offline Agent — Pipeline A: extract SFT data from saved session.v3.jsonl trajectories.

Reads saved trajectory files, reconstructs file states and context,
runs golden path engine, formats to GEAK schema, and writes to run directory.
"""



import hashlib
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set

from ..core.concept_extractor import ConceptExtractor
from ..core.context_accumulator import ContextAccumulator
from ..core.formatter import Formatter
from ..core.golden_path_engine import GoldenPathEngine
from ..core.independent_verifier import IndependentVerifier
from ..core.quality_gates import (
    GateResult,
    check_analysis_trajectory,
    check_common,
    check_concept_snapshot,
    check_field_completeness,
    check_opus_kernel,
)
from ..core.schema import (
    EventKind,
    SFTSample,
    Segment,
    VerifyStatus,
)
from ..core.shadow_copy import ShadowCopyManager
from ..core.writer import Writer
from .state_reconstructor import StateReconstructor, correlate_tool_calls_results
from .trajectory_parser import (
    find_trajectory_files,
    parse_events,
    segment_by_present,
)

logger = logging.getLogger(__name__)


class OfflineExtractionConfig:
    """Configuration for offline extraction."""

    def __init__(
        self,
        collection_mode: str = "direction_conditioned",
        harness_config: Optional[Dict[str, Any]] = None,
        workspace_dir: str = "",
        compiler_cmd: str = "opus-compile",
        test_runner_cmd: str = "opus-test",
        benchmark_cmd: Optional[str] = "opus-bench",
        verify_timeout: int = 300,
        skip_verify: bool = False,
        extract_concepts: bool = True,
    ) -> None:
        self.collection_mode = collection_mode
        self.harness_config = harness_config or {}
        self.workspace_dir = workspace_dir
        self.compiler_cmd = compiler_cmd
        self.test_runner_cmd = test_runner_cmd
        self.benchmark_cmd = benchmark_cmd
        self.verify_timeout = verify_timeout
        self.skip_verify = skip_verify
        self.extract_concepts = extract_concepts


class OfflineAgent:
    """Processes saved trajectory files and extracts SFT data."""

    def __init__(self, run_dir: str, config: OfflineExtractionConfig) -> None:
        self.run_dir = run_dir
        self.config = config
        self.writer = Writer(run_dir)
        self.formatter = Formatter(run_id=os.path.basename(run_dir))
        self.golden_engine = GoldenPathEngine()
        self.concept_extractor = ConceptExtractor()
        self.verifier = IndependentVerifier(
            compiler_cmd=config.compiler_cmd,
            test_runner_cmd=config.test_runner_cmd,
            benchmark_cmd=config.benchmark_cmd,
            timeout=config.verify_timeout,
        )
        self._concept_hashes: Set[str] = set()
        self._stats = {
            "trajectories_processed": 0,
            "segments_processed": 0,
            "analysis_samples": 0,
            "opus_samples": 0,
            "concept_samples": 0,
            "dead_ends": 0,
            "rejected": 0,
            "pending": 0,
        }

    def process_trajectory(self, jsonl_path: str) -> Dict[str, int]:
        """Process a single trajectory file and extract SFT data.

        Returns counts of extracted samples by type.
        """
        logger.info("Processing trajectory: %s", jsonl_path)
        counts = {"analysis": 0, "opus": 0, "concept": 0, "dead_end": 0, "rejected": 0}

        # 1. Parse events
        events = parse_events(jsonl_path)
        if not events:
            logger.warning("No events found in %s", jsonl_path)
            return counts

        # 2. Correlate tool_call → tool_result pairs
        events = correlate_tool_calls_results(events)

        # 3. Segment by present()
        segments = segment_by_present(events)

        # 4. Build reconstructor with full event history
        reconstructor = StateReconstructor(
            collection_mode=self.config.collection_mode,
            harness_config=self.config.harness_config,
        )

        # 5. Process each segment
        for seg_idx, segment in enumerate(segments):
            reconstructor.replay_segment(segment)
            shadow = reconstructor.shadow
            context = reconstructor.context

            # 5a. Analysis trajectory extraction
            if segment.deliverable:
                c = self._extract_analysis(segment, context)
                counts["analysis"] += c["ok"]
                counts["dead_end"] += c["dead"]
                counts["rejected"] += c["reject"]

            # 5b. OPUS kernel extraction
            for filepath in shadow.tracked_files():
                result = shadow.collect(filepath)
                if result:
                    parent, candidate, diff = result
                    c = self._extract_opus(
                        filepath, parent, candidate, diff, context
                    )
                    counts["opus"] += c["ok"]
                    counts["rejected"] += c["reject"]
                    shadow.clear_file(filepath)

            # 5c. Concept snapshots
            if self.config.extract_concepts:
                for ev in segment.events:
                    if ev.kind == EventKind.REASONING:
                        text = ev.data.get("text", "")
                        c = self._extract_concept(text, context)
                        counts["concept"] += c

        self._stats["trajectories_processed"] += 1
        self._stats["segments_processed"] += len(segments)
        self._stats["analysis_samples"] += counts["analysis"]
        self._stats["opus_samples"] += counts["opus"]
        self._stats["concept_samples"] += counts["concept"]
        self._stats["dead_ends"] += counts["dead_end"]
        self._stats["rejected"] += counts["rejected"]

        logger.info(
            "Trajectory %s: %d analysis, %d opus, %d concept, %d dead_end, %d rejected",
            os.path.basename(jsonl_path),
            counts["analysis"], counts["opus"], counts["concept"],
            counts["dead_end"], counts["rejected"],
        )
        return counts

    def process_batch(self, trajectory_dir: str) -> Dict[str, int]:
        """Process all trajectory files in a directory.

        Returns aggregate counts.
        """
        files = find_trajectory_files(trajectory_dir)
        logger.info("Found %d trajectory files in %s", len(files), trajectory_dir)

        totals = {"analysis": 0, "opus": 0, "concept": 0, "dead_end": 0, "rejected": 0}
        for fpath in files:
            try:
                counts = self.process_trajectory(fpath)
                for k in totals:
                    totals[k] += counts.get(k, 0)
            except Exception:
                logger.exception("Failed to process %s", fpath)

        # Write collector metadata
        self.writer.write_collector_meta({
            "version": "1.0.0",
            "extraction_method": "offline_agent",
            "collection_mode": self.config.collection_mode,
            "trajectory_dir": trajectory_dir,
            "trajectory_count": len(files),
            "stats": self._stats,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        logger.info("Batch complete: %s", totals)
        return totals

    # ---- internal extraction methods ----

    def _extract_analysis(
        self, segment: Segment, context: ContextAccumulator
    ) -> Dict[str, int]:
        """Extract analysis trajectory sample from a segment."""
        result = {"ok": 0, "dead": 0, "reject": 0}

        ctx_snap = context.snapshot()
        golden = self.golden_engine.classify(segment)

        sample = self.formatter.format_analysis_trajectory(
            golden=golden,
            context=ctx_snap,
            deliverable=segment.deliverable or "",
        )

        # Quality gates
        gate_common = check_common(sample)
        gate_analysis = check_analysis_trajectory(sample)

        if not gate_common or not gate_analysis:
            reasons = gate_common.reasons + gate_analysis.reasons
            self.writer.write_rejected(sample, reasons)
            result["reject"] = 1
            return result

        self.writer.write_sample(sample)
        result["ok"] = 1

        # Write dead ends as RL negatives
        if golden.dead_ends:
            for de in golden.dead_ends:
                de_id = f"de-{sample.sample_id}-{de.event_index}"
                self.writer.write_dead_end(de_id, {
                    "source_sample_id": sample.sample_id,
                    "event_index": de.event_index,
                    "tool_call": de.tool_call,
                    "tool_result": de.tool_result[:2000],
                    "reasoning": de.related_reasoning[:2000],
                })
                result["dead"] += 1

        return result

    def _extract_opus(
        self,
        filepath: str,
        parent: str,
        candidate: str,
        diff: str,
        context: ContextAccumulator,
    ) -> Dict[str, int]:
        """Extract OPUS kernel sample."""
        result = {"ok": 0, "reject": 0}
        ctx_snap = context.snapshot()

        # Run independent verify unless skipped
        verify_receipt = None
        if not self.config.skip_verify:
            ws_config = {
                "workspace_dir": self.config.workspace_dir,
                "kernel_file": os.path.basename(filepath),
                "gpu_identity": ctx_snap.architecture.gpu_sku,
            }
            diff_hash = hashlib.sha256(diff.encode()).hexdigest()[:12]
            verify_receipt = self.verifier.verify(
                parent, candidate, diff, ws_config, self.run_dir, "opus-verify-{}".format(diff_hash)
            )

        sample = self.formatter.format_opus_kernel(
            parent_source=parent,
            candidate_source=candidate,
            diff=diff,
            context=ctx_snap,
            verify_receipt=verify_receipt,
        )

        # Quality gates
        gate_common = check_common(sample)
        gate_opus = check_opus_kernel(sample, verify_receipt, skip_verify=self.config.skip_verify)

        if verify_receipt and verify_receipt.status == VerifyStatus.PENDING:
            self.writer.write_pending(sample)
            self._stats["pending"] += 1
            return result

        if not gate_common or not gate_opus:
            reasons = gate_common.reasons + gate_opus.reasons
            self.writer.write_rejected(sample, reasons)
            result["reject"] = 1
            return result

        self.writer.write_sample(sample, parent, candidate)
        result["ok"] = 1
        return result

    def _extract_concept(
        self, reasoning_text: str, context: ContextAccumulator
    ) -> int:
        """Extract concept snapshot. Returns 1 if extracted, 0 otherwise."""
        snapshot = self.concept_extractor.extract(reasoning_text)
        if snapshot is None:
            return 0
        if snapshot.has_unverified_constants:
            return 0

        ctx_snap = context.snapshot()
        sample = self.formatter.format_concept_snapshot(
            concept_text=snapshot.concept_text,
            inferred_question=snapshot.inferred_question,
            concept_count=snapshot.concept_count,
            has_unverified_constants=snapshot.has_unverified_constants,
            context=ctx_snap,
        )

        gate = check_concept_snapshot(sample, self._concept_hashes)
        if not gate:
            self.writer.write_rejected(sample, gate.reasons)
            return 0

        self._concept_hashes.add(sample.content_hash())
        self.writer.write_sample(sample)
        return 1
