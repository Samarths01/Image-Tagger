"""
orchestrator.py — Pipeline coordinator.

Manages the full execution flow:
  Vision → Enrichment → Taxonomy → Validation

Handles retries, routing decisions, and pipeline observability.
"""

from typing import Any

from .enrichment_agent import EnrichmentAgent
from .taxonomy_agent import TaxonomyAgent
from .validation_agent import ValidationAgent
from .vision_agent import VisionAgent

INGEST = "ingest"
REVIEW = "review"
ESCALATE = "escalate"


class Orchestrator:
    """
    Coordinates the 4-agent product tagging pipeline.
    """

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self.vision = VisionAgent()
        self.enrichment = EnrichmentAgent()
        self.taxonomy = TaxonomyAgent()
        self.validation = ValidationAgent()

    def process(
        self,
        image_input: str,
        product_text: dict[str, Any] | None = None,
        product_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full pipeline for a single product.

        Args:
            image_input:  Local file path or public URL to product image
            product_text: Optional dict with title, description, specifications
            product_id:   Optional identifier for tracking

        Returns:
            Pipeline result with routing_decision, final_output, and pipeline_trace
        """
        trace = []
        review_flags: list[str] = []
        retries = 0

        vision_output, vision_step = self._run_vision(image_input, trace)
        trace.append(vision_step)

        if vision_output["confidence"] == "low":
            review_flags.append(
                "Vision Agent reported low confidence — recommend human review of image quality"
            )

        enriched, enrich_step = self._run_enrichment(vision_output, product_text, trace)
        trace.append(enrich_step)

        if enriched.get("text_input_quality") == "absent":
            review_flags.append("No product text available — tags based on image only")

        if enriched.get("conflicts"):
            review_flags.append(
                f"{len(enriched['conflicts'])} attribute conflict(s) detected and resolved — verify accuracy"
            )

        taxonomy_output, taxonomy_step, retries = self._run_taxonomy_with_retry(
            enriched, trace
        )
        trace.append(taxonomy_step)

        if taxonomy_output.get("mapping_notes"):
            review_flags.append(f"Taxonomy mapping note: {taxonomy_output['mapping_notes']}")

        validation_result, validation_step = self._run_validation(taxonomy_output, trace)
        trace.append(validation_step)

        routing, routing_reason = self._decide_routing(
            validation_result, review_flags, retries
        )

        return {
            "product_id": product_id,
            "routing_decision": routing,
            "routing_reason": routing_reason,
            "review_flags": review_flags,
            "final_output": validation_result.get("validated_output", {}),
            "pipeline_summary": {
                "total_steps": len(trace),
                "retries": retries,
                "vision_confidence": vision_output.get("confidence"),
                "text_input_quality": enriched.get("text_input_quality"),
                "validation_status": validation_result.get("status"),
                "conflicts_found": len(enriched.get("conflicts", [])),
                "processing_notes": "; ".join(review_flags) if review_flags else None,
            },
            "pipeline_trace": trace,
        }

    def _run_vision(self, image_input: str, trace: list) -> tuple[dict, dict]:
        output = self.vision.analyze(image_input)
        step = self._trace_step(1, "Vision Agent", "success", output)

        if output.get("confidence") == "low" and self.max_retries > 0:
            output = self.vision.analyze(
                image_input,
                hint="Please focus particularly on color, material, and brand details.",
            )
            step = self._trace_step(1, "Vision Agent", "retried", output)

        return output, step

    def _run_enrichment(
        self, vision_output: dict, product_text: dict | None, trace: list
    ) -> tuple[dict, dict]:
        output = self.enrichment.fuse(vision_output, product_text)
        step = self._trace_step(2, "Enrichment Agent", "success", output)
        return output, step

    def _run_taxonomy_with_retry(
        self, enriched: dict, trace: list
    ) -> tuple[dict, dict, int]:
        retries = 0
        taxonomy_output = self.taxonomy.map(enriched)
        validation_result = self.validation.check(taxonomy_output)

        while (
            not self.validation.passed(validation_result)
            and retries < self.max_retries
        ):
            hints = self.validation.get_fail_hints(validation_result)
            retry_hint = " ".join(hints.values())

            taxonomy_output = self.taxonomy.map(enriched, retry_hint=retry_hint)
            validation_result = self.validation.check(taxonomy_output)
            retries += 1

        status = "retried" if retries > 0 else "success"
        step = self._trace_step(3, "Taxonomy Agent", status, taxonomy_output)
        return taxonomy_output, step, retries

    def _run_validation(self, taxonomy_output: dict, trace: list) -> tuple[dict, dict]:
        output = self.validation.check(taxonomy_output)
        step = self._trace_step(4, "Validation Agent", "success", output)
        return output, step

    def _decide_routing(
        self,
        validation_result: dict,
        review_flags: list[str],
        retries: int,
    ) -> tuple[str, str]:
        status = validation_result.get("status")
        has_conflicts = bool(review_flags)

        if status == "fail":
            if retries >= self.max_retries:
                return (
                    ESCALATE,
                    f"Validation failed after {retries} retries. Manual review required.",
                )
            return (ESCALATE, "Validation failed. Pipeline halted.")

        if status == "warn" or has_conflicts:
            return (
                REVIEW,
                "Product is usable but has quality flags requiring async review.",
            )

        return (INGEST, "All validation checks passed. No conflicts detected.")

    @staticmethod
    def _trace_step(step_num: int, agent_name: str, status: str, output: dict) -> dict:
        summary_parts = []

        if agent_name == "Vision Agent":
            cat = output.get("product_identity", {}).get("category", "unknown")
            conf = output.get("confidence", "unknown")
            summary_parts.append(f"category={cat}, confidence={conf}")
        elif agent_name == "Enrichment Agent":
            quality = output.get("text_input_quality", "unknown")
            conflicts = len(output.get("conflicts", []))
            summary_parts.append(f"text_quality={quality}, conflicts={conflicts}")
        elif agent_name == "Taxonomy Agent":
            dept = output.get("taxonomy", {}).get("department", "unknown")
            tag_count = len(output.get("search_tags", []))
            summary_parts.append(f"department={dept}, search_tags={tag_count}")
        elif agent_name == "Validation Agent":
            v_status = output.get("status", "unknown")
            fails = output.get("issue_count", {}).get("fail", 0)
            warns = output.get("issue_count", {}).get("warn", 0)
            summary_parts.append(f"status={v_status}, fails={fails}, warns={warns}")

        return {
            "step": step_num,
            "agent": agent_name,
            "status": status,
            "output_summary": ", ".join(summary_parts) or "completed",
        }
