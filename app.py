from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from functions.symptom_extractor import extract_symptoms
from functions.diagnosis_symptoms import get_diagnosis
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata, format_articles_as_text
from functions.summerize_pubmed import summarize_text


app = FastAPI()

# ── Serve the frontend ──────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


class SymptomInput(BaseModel):
    description:str

@app.post("/diagnosis")
def diagnosis(data:SymptomInput):
    symptom = extract_symptoms(data.description)
    diagnosis_result = get_diagnosis(symptom)
    pubmed_articles = fetch_pubmed_articles_with_metadata(" ".join(symptom))
    articles_text = format_articles_as_text(pubmed_articles)
    summary = summarize_text(articles_text[:3000])

    return {
        "symptom":symptom,
        "diagnosis":diagnosis_result,
        "pubmed_articles": pubmed_articles,
        "pubmed_summary" :summary
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host = "127.0.0.1", port=8080, reload = True)