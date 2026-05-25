# Image Tagger

A multi-agent AI pipeline that analyzes product images and generates structured catalog tags for e-commerce — search keywords, taxonomy classification, color, material, style, and more.

## Architecture

```
Image + Text → [Vision] → [Enrichment] → [Taxonomy] → [Validation] → Route Decision
                  ↑________________ Retry w/ hints (Orchestrator) ___↑
```

| Agent | Role |
|-------|------|
| Vision | Extracts raw visual attributes from the image |
| Enrichment | Fuses vision + product text, resolves conflicts |
| Taxonomy | Maps to standardized catalog vocabulary |
| Validation | QA gate — 14 rules, outputs `pass/warn/fail` |
| Orchestrator | Coordinates pipeline, retry logic, routing decisions |

**Routing decisions:**
- `ingest` — all checks pass, ready for catalog
- `review` — warnings/conflicts, async human review needed
- `escalate` — validation failures after max retries

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Samarths01/Image-Tagger.git
cd Image-Tagger
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Or run the CLI

```bash
python main.py /path/to/product.jpg
python main.py https://example.com/product.jpg
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo + `app.py`
4. In **Settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
5. Click **Deploy** — you'll get a shareable public URL

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | Your Anthropic API key |

## Project Structure

```
Image-Tagger/
├── agents/
│   ├── base.py              # Shared client + call utilities
│   ├── vision_agent.py      # Image attribute extractor
│   ├── enrichment_agent.py  # Vision + text fusion
│   ├── taxonomy_agent.py    # Catalog tag mapper
│   ├── validation_agent.py  # Quality gate
│   └── orchestrator.py      # Pipeline coordinator
├── app.py                   # Streamlit web UI
├── main.py                  # CLI entry point
├── requirements.txt
├── .env.example
└── .streamlit/
    └── secrets.toml.example
```
