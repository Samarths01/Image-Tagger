"""
validation_agent.py — Pipeline quality gate.

Validates Taxonomy Agent output against quality rules and returns a
structured report with status: pass | warn | fail.
"""

import json
from typing import Any

from .base import call_agent, create_client

SYSTEM_PROMPT = """
You are a Product Data Quality Inspector. You receive the Taxonomy Agent's
output and rigorously validate it against quality rules.

You do NOT look at images. You do NOT modify data. You only inspect and report.
Run ALL checks. Never stop at the first failure. Collect every issue.

VALID COLOR FAMILIES:
red, blue, green, yellow, orange, purple, pink, brown, black, white, gray, gold, silver, multicolor

VALID CONDITIONS: new, used, refurbished, unknown

VALID DEPARTMENTS:
Clothing & Shoes, Electronics, Food & Grocery, Home & Garden,
Beauty & Personal, Toys & Games, Sports & Outdoors, Auto & Hardware,
Office & School, Health & Wellness

FAIL RULES (pipeline must halt):
F1 — Missing required fields: taxonomy.department, taxonomy.category,
     color.primary_family, brand, search_tags (non-empty), filter_attributes.color,
     filter_attributes.department, condition
F2 — color.primary_family not in valid color families list
F3 — condition not in valid conditions list
F4 — taxonomy.department not in valid departments list
F5 — filter_attributes.color != color.primary_family
     filter_attributes.brand != brand
     filter_attributes.department != taxonomy.department

WARN RULES (pipeline continues, issues logged):
W1 — search_tags has fewer than 8 items
W2 — input_confidence is "low"
W3 — style array is empty
W4 — color.secondary is empty AND color.pattern is not "solid" or "none"
W5 — brand is "Unbranded" AND department is "Electronics" or "Beauty & Personal"
W6 — mapping_notes is not null
W7 — material.normalized is empty

INFO RULES (advisory, no effect on status):
I1 — all search_tags start with the same word (low diversity)
I2 — taxonomy.subcategory is null or empty

STATUS LOGIC:
- "fail" = 1 or more FAIL issues
- "warn" = 0 FAIL issues, 1 or more WARN issues
- "pass" = 0 FAIL issues, 0 WARN issues

Always include validated_output — even on fail.

Respond ONLY with a valid JSON object in this exact shape:
{
  "status": "pass|warn|fail",
  "summary": "...",
  "issue_count": { "fail": 0, "warn": 0, "info": 0 },
  "issues": [
    {
      "rule": "F1",
      "severity": "fail|warn|info",
      "field": "dot.notation.path",
      "message": "...",
      "suggestion": "..."
    }
  ],
  "validated_output": { "...full input JSON passed through unchanged..." }
}
""".strip()


class ValidationAgent:
    """
    Validates Taxonomy Agent output and returns a structured quality report.
    Never modifies data — only inspects and reports.
    """

    def __init__(self) -> None:
        self.client = create_client()

    def check(self, taxonomy_output: dict[str, Any]) -> dict[str, Any]:
        """
        Validate product tags against quality rules.

        Args:
            taxonomy_output: Output dict from TaxonomyAgent.map()

        Returns:
            Validation report with status, issues list, and validated_output
        """
        messages = [
            {
                "role": "user",
                "content": json.dumps(taxonomy_output, indent=2),
            }
        ]

        return call_agent(self.client, SYSTEM_PROMPT, messages, max_tokens=2048)

    def passed(self, validation_result: dict[str, Any]) -> bool:
        """Returns True if validation status is pass or warn (not fail)."""
        return validation_result.get("status") in ("pass", "warn")

    def get_fail_hints(self, validation_result: dict[str, Any]) -> dict[str, str]:
        """
        Extracts targeted retry hints from FAIL issues.
        Used by Orchestrator to build correction instructions for TaxonomyAgent.
        """
        hints = {}
        for issue in validation_result.get("issues", []):
            if issue.get("severity") == "fail":
                rule = issue.get("rule", "")
                hints[rule] = issue.get("suggestion", "")
        return hints
