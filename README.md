# 🏥 CliniSight AI — Medical Diagnosis Assistant

An AI-powered clinical decision-support tool that extracts symptoms from natural language, generates differential diagnoses, retrieves PubMed evidence, and summarizes the literature — all through a clean web UI or MCP tool interface.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [MCP Tool](#mcp-tool)
- [Tech Stack](#tech-stack)

---

## 🏗️ Architecture

```
User Input (symptoms text)
        │
        ▼
┌─────────────────────┐
│  extract_symptoms()  │  ← GPT-4o: clinical NLP + negation handling
└────────┬────────────┘
         │  list[str]
         ▼
┌─────────────────────┐
│   get_diagnosis()    │  ← GPT-4o: structured differential diagnosis
└────────┬────────────┘
         │  str (formatted report)
         ▼
┌──────────────────────────────┐
│ fetch_pubmed_articles()      │  ← PubMed E-utilities API
└────────┬─────────────────────┘
         │  list[dict]
         ▼
┌─────────────────────┐
│  summarize_text()    │  ← GPT-4o: evidence synthesis
└────────┬────────────┘
         │
         ▼
   JSON Response → Web UI
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Symptom Extraction** | LLM-powered with negation detection, medical synonym normalization, and regex fallback |
| **Differential Diagnosis** | Structured report with likelihood rankings, key findings, and red flags |
| **PubMed Evidence** | Smart boolean query builder, retry with exponential backoff, NCBI API key support |
| **Research Summary** | Concise synthesis of retrieved articles with clinical relevance |
| **Web Frontend** | Clean single-page UI with progress indicators and expandable article cards |
| **MCP Tool** | Use via Claude Desktop or any MCP-compatible client |
| **Model Fallback** | Automatic fallback chain: `gpt-4o → gpt-4-turbo → gpt-4` |
| **Error Resilience** | Input sanitization, graceful degradation, structured logging |

---

## 📁 Project Structure

```
Medical Diagnosis/
├── app.py                          # FastAPI server + frontend route
├── mcp_tool.py                     # MCP server (stdio transport)
├── requirements.txt                # Python dependencies
├── .env                            # API keys (not committed)
├── .gitignore
├── static/
│   └── index.html                  # Web frontend (single-page)
├── functions/
│   ├── symptom_extractor.py        # GPT-4o symptom extraction
│   ├── diagnosis_symptoms.py       # Differential diagnosis generation
│   ├── pubmed_articles.py          # PubMed search & retrieval
│   └── summerize_pubmed.py         # Article summarization
└── demo.ipynb                      # PubMed pipeline demo notebook
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/clinisight-ai.git
cd clinisight-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
NCBI_API_KEY=your_ncbi_api_key_here    # optional, increases PubMed rate limit
```

> Get your OpenAI key at [platform.openai.com](https://platform.openai.com/api-keys)
> Get a free NCBI key at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/)

---

## 🚀 Usage

### Web UI

```bash
python app.py
```

Open **http://127.0.0.1:8080** in your browser. Enter a symptom description and click **Analyze Symptoms**.

### API (cURL)

```bash
curl -X POST http://127.0.0.1:8080/diagnosis \
  -H "Content-Type: application/json" \
  -d '{"description": "persistent dry cough for 2 weeks, mild fever, night sweats, weight loss"}'
```

---

## 📡 API Reference

### `POST /diagnosis`

**Request Body:**

```json
{
  "description": "patient symptom text here"
}
```

**Response:**

```json
{
  "symptom": ["dry cough", "fever", "night sweats", "weight loss"],
  "diagnosis": "Differential Diagnosis Report...",
  "pubmed_articles": [
    {
      "title": "Article Title",
      "authors": "Author A, Author B",
      "journal": "Journal Name",
      "publication_date": "2024-01-15",
      "abstract": "...",
      "article_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    }
  ],
  "pubmed_summary": "Summary of the research evidence..."
}
```

---

## 🔧 MCP Tool

CliniSight AI can also run as an [MCP](https://modelcontextprotocol.io/) tool for use with Claude Desktop or any MCP-compatible client.

```bash
# Install into Claude Desktop
mcp install mcp_tool.py

# Development mode
uv run mcp dev mcp_tool.py
```

---

## 🛠️ Tech Stack

- **LLM:** OpenAI GPT-4o (with fallback chain)
- **Backend:** FastAPI + Uvicorn
- **Evidence:** PubMed E-utilities (esearch + efetch)
- **XML Parsing:** BeautifulSoup + lxml
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
- **MCP:** FastMCP (stdio transport)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

MIT
