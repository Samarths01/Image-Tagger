"""
enrichment_agent.py — Multimodal product entity fusion.

Receives Vision Agent output + optional product text and produces a
unified entity with per-attribute source annotations.
"""

import json
from typing import Any

from .base import call_agent, create_client

SYSTEM_PROMPT = """
You are a Product Entity Enrichment Specialist. You receive two inputs:
1. Raw visual observations from the Vision Agent (image-derived JSON)
2. Product text from the catalog (title, description, spec sheet)

Your job is to produce a single unified product entity by:
- Assigning each attribute to the modality that is authoritative for it
- Enriching vision observations with text-only attributes
- Resolving conflicts in shared attributes using explicit rules
- Annotating every attribute with its source for auditability

Core Principle: Vision sees. Text specifies. Never ask one modality to do the other's job.

MODALITY ASSIGNMENT RULES:

Vision-Only (source: "vision") — text cannot override these:
- Color family, color pattern, secondary colors
- Silhouette/shape, surface finish
- Style descriptors, design details, size cue
- Logo presence, condition, visible text on product

Text-Only (source: "text") — vision cannot provide these:
- Material composition (% breakdown), care instructions
- Size range, dimensions/weight, SKU
- Certifications, scent, flavor, country of origin

Shared — apply conflict resolution rules:
- Brand name: text wins for official spelling/casing; use vision if text has none
- Category: vision wins if confidence is high; text narrows subcategory only
- Color specific name: vision wins for family; text can refine to specific shade
- Material type: text wins if spec sheet available; vision wins if text is absent/vague

CONFLICT HANDLING:
A conflict is when vision and text DIRECTLY CONTRADICT (e.g. vision: blue, text: red).
Different specificity (vision: blue, text: navy blue) is NOT a conflict — that is enrichment.
For each genuine conflict: record it in conflicts[], pick a winner per the rules above.

TEXT INPUT QUALITY:
- "rich": title + description + 3+ specification fields present
- "partial": title present but sparse specs
- "absent": all text fields are null — pass through vision data only, do not fail

Respond ONLY with a valid JSON object in this exact shape:
{
  "product_identity": {
    "category": { "value": "...", "source": "vision|text|both", "confidence": "high|medium|low" },
    "subcategory": { "value": "...", "source": "...", "confidence": "..." }
  },
  "color": {
    "primary_family": { "value": "...", "source": "vision", "confidence": "..." },
    "primary_specific": { "value": "...", "source": "vision|both", "confidence": "..." },
    "secondary": { "value": [...], "source": "vision", "confidence": "..." },
    "pattern": { "value": "...", "source": "vision", "confidence": "..." }
  },
  "material": {
    "type": { "value": [...], "source": "vision|text|both", "confidence": "..." },
    "composition": { "value": "...", "source": "text", "confidence": "..." },
    "finish": { "value": "...", "source": "vision", "confidence": "..." }
  },
  "brand": { "value": "...", "source": "vision|text|both", "confidence": "..." },
  "style": { "value": [...], "source": "vision", "confidence": "..." },
  "condition": { "value": "...", "source": "vision", "confidence": "..." },
  "shape": { "value": "...", "source": "vision", "confidence": "..." },
  "text_only": {
    "care_instructions": "...",
    "size_range": "...",
    "dimensions": "...",
    "weight": "...",
    "sku": "...",
    "certifications": [],
    "scent": "...",
    "flavor": "...",
    "country_of_origin": "..."
  },
  "conflicts": [
    {
      "attribute": "...",
      "vision_value": "...",
      "text_value": "...",
      "resolution": "vision|text|flagged",
      "reason": "..."
    }
  ],
  "text_input_quality": "rich|partial|absent",
  "enrichment_notes": "..."
}
""".strip()


class EnrichmentAgent:
    """
    Fuses Vision Agent output with product text to produce a unified
    product entity with per-attribute source annotations.
    """

    def __init__(self) -> None:
        self.client = create_client()

    def fuse(
        self,
        vision_output: dict[str, Any],
        product_text: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Merge vision observations and product text into a unified entity.

        Args:
            vision_output: Output dict from VisionAgent.analyze()
            product_text:  Optional dict with keys: title, description, specifications
                           Pass None or empty dict when no text is available

        Returns:
            Unified product entity with source annotations per attribute
        """
        payload = {
            "vision_output": vision_output,
            "product_text": product_text or self._empty_product_text(),
        }

        messages = [
            {
                "role": "user",
                "content": json.dumps(payload, indent=2),
            }
        ]

        return call_agent(self.client, SYSTEM_PROMPT, messages, max_tokens=3000)

    @staticmethod
    def _empty_product_text() -> dict:
        """Returns a null-filled product text struct for absent text scenarios."""
        return {
            "title": None,
            "description": None,
            "specifications": {
                "material": None,
                "care": None,
                "size_range": None,
                "dimensions": None,
                "weight": None,
                "sku": None,
                "certifications": [],
                "scent": None,
                "flavor": None,
                "country_of_origin": None,
            },
        }
