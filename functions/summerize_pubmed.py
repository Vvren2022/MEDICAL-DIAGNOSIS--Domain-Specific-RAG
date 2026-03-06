import os
import logging

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt — detailed instructions for medical literature summarisation
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert medical research summarizer. Your task is to
produce a concise, clinically useful summary from PubMed article data.

RULES:
1. Summarize the KEY FINDINGS from each article — focus on diagnosis,
   treatment outcomes, and clinical relevance.
2. Preserve important study details: sample size, methodology, results.
3. Clearly attribute findings to their source article by title.
4. If no article data is provided or the text says "No relevant PubMed
   articles were found", state that no evidence was available.
5. Do NOT invent or hallucinate findings not present in the input.
6. Maintain a neutral, evidence-based tone.
7. End with a brief "Clinical Relevance" statement explaining how
   these findings might inform patient care.
"""

_USER_PROMPT_TEMPLATE = """Summarize the following PubMed research articles for a
clinician reviewing a patient case:

{text}
"""


def summarize_text(text: str) -> str:
    """Summarize PubMed article text for clinical use.

    Handles:
    - Empty / missing input gracefully
    - Model fallback chain (gpt-4o → gpt-4-turbo → gpt-4)
    - Error recovery — never crashes the pipeline
    """
    # ── Guard: empty input ────────────────────────────────────────────
    if not text or not text.strip():
        return "No PubMed articles were available to summarize."

    no_evidence_markers = [
        "no relevant pubmed articles",
        "no pubmed articles were found",
    ]
    if any(marker in text.lower() for marker in no_evidence_markers):
        return (
            "No relevant PubMed research articles were found for the given "
            "symptoms. Please consult a healthcare professional for evidence-"
            "based guidance."
        )

    # ── Build messages ────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=text)},
    ]

    # ── Call LLM with model fallback ──────────────────────────────────
    models = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Summarize with %s failed: %s", model, exc)
            continue

    logger.error("All summarization models failed.")
    return (
        "Unable to generate a research summary at this time due to a "
        "service error. The raw PubMed article data is still available "
        "in the response for manual review."
    )