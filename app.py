"""
app.py — Streamlit web UI for the Image Tagger pipeline.

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io → New app → select repo + app.py
    3. Add ANTHROPIC_API_KEY in Settings → Secrets

Set ANTHROPIC_API_KEY in your .env file (or Streamlit secrets) before running.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from agents import Orchestrator

st.set_page_config(
    page_title="Image Tagger",
    page_icon="🏷️",
    layout="wide",
)

st.title("Image Tagger")
st.caption(
    "Multi-agent pipeline: Vision → Enrichment → Taxonomy → Validation. "
    "Upload a product image and (optionally) provide product text to generate catalog tags."
)

# ── Sidebar: optional product text ───────────────────────────────────────────
with st.sidebar:
    st.header("Optional Product Text")
    st.caption("Leave blank to run on image alone.")

    title = st.text_input("Title", value="")
    description = st.text_area("Description", value="", height=80)

    st.markdown("**Specifications**")
    material = st.text_input("Material", value="")
    care = st.text_input("Care", value="")
    size_range = st.text_input("Size Range", value="")
    sku = st.text_input("SKU", value="")
    country = st.text_input("Country of Origin", value="")

    st.markdown("---")
    max_retries = st.slider("Max Retries", min_value=0, max_value=3, value=2)

# ── Main: image upload ───────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Product image",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        st.image(uploaded, use_container_width=True)

    run_button = st.button(
        "Analyze Image",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )


def build_product_text() -> dict | None:
    """Assemble product text from sidebar inputs. Return None if all empty."""
    has_any = any([title, description, material, care, size_range, sku, country])
    if not has_any:
        return None

    return {
        "title": title or None,
        "description": description or None,
        "specifications": {
            "material": material or None,
            "care": care or None,
            "size_range": size_range or None,
            "sku": sku or None,
            "country_of_origin": country or None,
            "dimensions": None,
            "weight": None,
            "certifications": [],
            "scent": None,
            "flavor": None,
        },
    }


def routing_badge(decision: str) -> str:
    """Map routing decision to a colored markdown badge."""
    colors = {
        "ingest": ("green", "✅ INGEST"),
        "review": ("orange", "⚠️ REVIEW"),
        "escalate": ("red", "❌ ESCALATE"),
    }
    color, label = colors.get(decision, ("gray", decision.upper()))
    return f":{color}-background[**{label}**]"


def render_result(result: dict) -> None:
    """Render the pipeline result in a structured layout."""
    st.markdown(f"### Decision: {routing_badge(result['routing_decision'])}")
    st.caption(result["routing_reason"])

    s = result["pipeline_summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vision Confidence", s.get("vision_confidence") or "—")
    m2.metric("Text Quality", s.get("text_input_quality") or "—")
    m3.metric("Validation", s.get("validation_status") or "—")
    m4.metric("Retries", s.get("retries", 0))

    if result.get("review_flags"):
        with st.expander(f"⚠️ Review Flags ({len(result['review_flags'])})", expanded=True):
            for flag in result["review_flags"]:
                st.warning(flag)

    tags = result.get("final_output") or {}
    if tags and result["routing_decision"] in ("ingest", "review"):
        st.subheader("Generated Tags")
        taxonomy = tags.get("taxonomy", {})
        color = tags.get("color", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Department:** {taxonomy.get('department', '—')}")
            st.markdown(
                f"**Category:** {taxonomy.get('category', '—')} › "
                f"{taxonomy.get('subcategory', '—')}"
            )
            st.markdown(f"**Brand:** {tags.get('brand', '—')}")
            st.markdown(f"**Condition:** {tags.get('condition', '—')}")
        with c2:
            st.markdown(
                f"**Color:** {color.get('primary_family', '—')} "
                f"({color.get('primary_specific', '—')})"
            )
            st.markdown(f"**Pattern:** {color.get('pattern', '—')}")
            style_list = tags.get("style", [])
            st.markdown(f"**Style:** {', '.join(style_list) if style_list else '—'}")

        search_tags = tags.get("search_tags", [])
        if search_tags:
            st.markdown("**Search Tags:**")
            st.write(" ".join(f"`{t}`" for t in search_tags))

    with st.expander("🔍 Pipeline Trace", expanded=False):
        icons = {"success": "✅", "retried": "🔄", "failed": "❌"}
        for step in result.get("pipeline_trace", []):
            icon = icons.get(step["status"], "•")
            st.markdown(f"{icon} **Step {step['step']}: {step['agent']}**")
            st.caption(step["output_summary"])

    with st.expander("📄 Raw JSON Output", expanded=False):
        st.json(result)

    st.download_button(
        "Download Full Output (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"{result.get('product_id', 'output')}.json",
        mime="application/json",
    )


# ── Run pipeline on button click ─────────────────────────────────────────────
with col_output:
    if run_button and uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        product_text = build_product_text()

        with st.status("Running pipeline...", expanded=True) as status:
            try:
                st.write("Initializing orchestrator...")
                orchestrator = Orchestrator(max_retries=max_retries)

                st.write("Running 4-agent pipeline (Vision → Enrichment → Taxonomy → Validation)...")
                result = orchestrator.process(
                    product_id=f"ITEM-{uploaded.name}",
                    image_input=tmp_path,
                    product_text=product_text,
                )
                status.update(label="Pipeline complete", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Pipeline failed", state="error", expanded=True)
                st.error(f"Error: {e}")
                result = None
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        if result:
            render_result(result)
    elif not run_button:
        st.info("Upload an image and click **Analyze Image** to run the pipeline.")
