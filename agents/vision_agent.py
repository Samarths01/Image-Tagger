"""
vision_agent.py — Visual product attribute extractor.

Receives a product image (local path or URL) and returns structured
visual observations. First agent in the pipeline.
"""

import base64
import mimetypes
from pathlib import Path
from typing import Any

from .base import call_agent, create_client

SYSTEM_PROMPT = """
You are a Product Vision Analyst. Your sole job is to examine a product
image and extract precise, factual visual observations. You are the first
agent in a multi-agent product tagging pipeline. Your output feeds directly
into an Enrichment Agent, so accuracy and structure are critical.

Never invent attributes. Only report what is visually present.

Analyze the provided product image and extract the following attributes:

1. Product Identity
   - category: Broad product type (footwear, apparel, electronics, grocery, furniture, beauty)
   - subcategory: More specific type (running shoe, wireless earbuds, moisturizer)

2. Color Profile
   - primary_color: Dominant color using standard names (red, navy blue, off-white)
   - secondary_colors: List of other visible colors
   - color_pattern: Pattern if present (solid, striped, floral, tropical leaf print, none)

3. Material & Texture
   - materials: Visible materials (leather, mesh, plastic, cotton, metal)
   - finish: Surface finish (matte, glossy, satin, textured, transparent)

4. Shape & Form
   - shape: General form (rectangular, cylindrical, irregular, A-line, flat)
   - size_indicator: Any visible size cues (compact, large, travel-size, midi length, unknown)

5. Branding & Text
   - brand: Brand name if visible, else null
   - visible_text: Any other text on the product or packaging (list all)
   - logos: Describe any logos or symbols present

6. Condition & Style
   - condition: Apparent state (new, packaged, worn, assembled)
   - style_descriptors: Visual style words (minimalist, sporty, vintage, premium, casual)

Respond ONLY with a valid JSON object in this exact shape:
{
  "product_identity": { "category": "...", "subcategory": "..." },
  "color_profile": { "primary_color": "...", "secondary_colors": [...], "color_pattern": "..." },
  "material_texture": { "materials": [...], "finish": "..." },
  "shape_form": { "shape": "...", "size_indicator": "..." },
  "branding": { "brand": "...", "visible_text": [...], "logos": "..." },
  "condition_style": { "condition": "...", "style_descriptors": [...] },
  "confidence": "high | medium | low",
  "notes": "any ambiguities or null"
}
""".strip()


class VisionAgent:
    """
    Analyzes product images and returns structured visual observations.
    Handles both local file paths and remote URLs transparently.
    """

    def __init__(self) -> None:
        self.client = create_client()

    def analyze(self, image_input: str, hint: str | None = None) -> dict[str, Any]:
        """
        Analyze a product image and extract visual attributes.

        Args:
            image_input: Local file path OR public URL to the product image
            hint:        Optional instruction added to the user message for retries

        Returns:
            Structured visual observations dict matching the output contract
        """
        image_content = self._build_image_content(image_input)

        user_text = "Analyze this product image and report all visual attributes."
        if hint:
            user_text += f" {hint}"

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        return call_agent(self.client, SYSTEM_PROMPT, messages)

    def _build_image_content(self, image_input: str) -> dict:
        """Builds the correct image content block for URL or local file."""
        if image_input.startswith("http://") or image_input.startswith("https://"):
            return {
                "type": "image",
                "source": {"type": "url", "url": image_input},
            }
        return self._encode_local_image(image_input)

    def _encode_local_image(self, file_path: str) -> dict:
        """Base64-encodes a local image file for the API."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        media_type, _ = mimetypes.guess_type(file_path)
        supported = {"image/jpeg", "image/png", "image/gif", "image/webp"}

        if media_type not in supported:
            raise ValueError(
                f"Unsupported image type '{media_type}'. "
                f"Supported: {', '.join(supported)}"
            )

        image_data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        }
