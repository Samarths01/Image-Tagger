"""
main.py — CLI entry point for the Image Tagger pipeline.

Usage:
    # With a local image file:
    python main.py /path/to/product.jpg

    # With a public image URL:
    python main.py https://example.com/product.jpg

    # With product text (title + specs as JSON string):
    python main.py /path/to/product.jpg '{"title": "...", "specifications": {...}}'

Set ANTHROPIC_API_KEY in your .env file before running.
See .env.example for the required format.
"""

import json
import sys
from pathlib import Path

from agents import Orchestrator

SAMPLE_PRODUCT_TEXT = {
    "title": "Women's Tropical Print Sleeveless Midi Dress with Pockets",
    "description": (
        "A flowy A-line dress perfect for summer. "
        "Features side pockets and decorative button detail at neckline."
    ),
    "specifications": {
        "material": "95% Cotton, 5% Spandex",
        "care": "Machine wash cold, tumble dry low",
        "size_range": "XS - 3XL",
        "dimensions": None,
        "weight": None,
        "sku": None,
        "certifications": [],
        "scent": None,
        "flavor": None,
        "country_of_origin": None,
    },
}


def print_result(result: dict) -> None:
    """Print a human-readable summary of the pipeline result."""
    w = 62
    print("\n" + "═" * w)
    print(f"  {'IMAGE TAGGER PIPELINE':^{w - 4}}")
    print("═" * w)
    print(f"  Product ID   : {result.get('product_id', 'N/A')}")
    print(f"  Decision     : {result['routing_decision'].upper()}")
    print(f"  Reason       : {result['routing_reason']}")
    print("─" * w)

    s = result["pipeline_summary"]
    print(f"  Vision Confidence  : {s['vision_confidence']}")
    print(f"  Text Quality       : {s['text_input_quality']}")
    print(f"  Validation Status  : {s['validation_status']}")
    print(f"  Conflicts Found    : {s['conflicts_found']}")
    print(f"  Retries Used       : {s['retries']}")

    if result["review_flags"]:
        print("\n  ⚠️  Review Flags:")
        for flag in result["review_flags"]:
            print(f"     • {flag}")

    print("\n  Pipeline Trace:")
    icons = {"success": "✅", "retried": "🔄", "failed": "❌"}
    for step in result["pipeline_trace"]:
        icon = icons.get(step["status"], "•")
        print(f"    {icon} Step {step['step']}: {step['agent']}")
        print(f"         {step['output_summary']}")

    routing = result["routing_decision"]
    tags = result.get("final_output", {})

    if routing in ("ingest", "review") and tags:
        print("\n  Generated Tags:")
        taxonomy = tags.get("taxonomy", {})
        print(f"    Department  : {taxonomy.get('department')}")
        print(f"    Category    : {taxonomy.get('category')} › {taxonomy.get('subcategory')}")
        print(f"    Brand       : {tags.get('brand')}")
        color = tags.get("color", {})
        print(f"    Color       : {color.get('primary_family')} ({color.get('primary_specific')})")
        print(f"    Pattern     : {color.get('pattern')}")
        print(f"    Condition   : {tags.get('condition')}")
        print(f"    Style       : {', '.join(tags.get('style', []))}")
        search_tags = tags.get("search_tags", [])
        print(f"    Search Tags : {', '.join(search_tags[:6])}")
        if len(search_tags) > 6:
            print(f"                  + {len(search_tags) - 6} more")

    output_path = Path("output.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Full output → {output_path.resolve()}")
    print("═" * w + "\n")


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print("\nUsage:")
        print("  python main.py <image_path_or_url> [product_text_json]")
        print("\nExamples:")
        print("  python main.py /path/to/product.jpg")
        print("  python main.py https://example.com/product.jpg")
        print("  python main.py dress.jpg '{\"title\": \"Blue dress\", \"specifications\": {}}'")
        sys.exit(0)

    image_input = args[0]

    product_text = SAMPLE_PRODUCT_TEXT
    if len(args) >= 2:
        try:
            product_text = json.loads(args[1])
        except json.JSONDecodeError:
            print("⚠️  Could not parse product_text argument as JSON — using default sample text.")

    print(f"\n  🔍 Processing: {image_input}")
    print(f"  📄 Text quality: {'provided' if product_text else 'none'}\n")

    orchestrator = Orchestrator(max_retries=2)

    result = orchestrator.process(
        product_id="ITEM-TEST-001",
        image_input=image_input,
        product_text=product_text,
    )

    print_result(result)


if __name__ == "__main__":
    main()
