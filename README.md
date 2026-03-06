# MEDICAL-DIAGNOSIS--Domain-Specific-RAG

CliniSight AI is a domain-specific medical diagnosis assistant focused on high-accuracy, evidence-backed outputs.
It extracts symptoms from clinical text, generates differential diagnosis suggestions, fetches supporting PubMed papers, and summarizes evidence.

## Features

- Symptom extraction with negation handling and normalization
- Differential diagnosis generation (structured, robust prompt pipeline)
- PubMed evidence retrieval with metadata
- Research summarization over retrieved papers
- FastAPI backend + simple web frontend
- MCP tool integration for MCP-compatible clients

## Project Structure

- `app.py` — FastAPI app and web route
- `mcp_tool.py` — MCP entrypoint
- `functions/symptom_extractor.py` — symptom extraction
- `functions/diagnosis_symptoms.py` — diagnosis reasoning
- `functions/pubmed_articles.py` — PubMed retrieval
- `functions/summerize_pubmed.py` — evidence summarization
- `static/index.html` — frontend UI

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in project root:

```env
OPENAI_API_KEY=your_openai_api_key
NCBI_API_KEY=your_ncbi_api_key
```

## Run

```bash
python app.py
```

Open:

- `http://127.0.0.1:8080` (frontend)
- `http://127.0.0.1:8080/docs` (FastAPI docs)

## API

`POST /diagnosis`

Request:

```json
{ "description": "persistent dry cough, fever, night sweats" }
```

Response includes:

- `symptom`
- `diagnosis`
- `pubmed_articles`
- `pubmed_summary`

## MCP

```bash
mcp install mcp_tool.py
uv run mcp dev mcp_tool.py
```

## Disclaimer

For educational/research support only. Not a replacement for professional medical advice.
