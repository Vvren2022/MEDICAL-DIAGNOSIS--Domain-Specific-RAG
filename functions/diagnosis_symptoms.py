import os
import re
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to act as a board-certified diagnostician
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a board-certified diagnostic physician with expertise
across internal medicine, emergency medicine, and general practice.

Your task: given a list of patient-reported symptoms, provide a **structured
differential diagnosis** with treatment guidance.

RULES — follow every one precisely:

1. **Differential Diagnosis** — Provide the top 3-5 most likely conditions
   ranked by probability. For each condition include:
   • Condition name
   • Likelihood (high / moderate / low)
   • Brief clinical reasoning (why these symptoms suggest this condition)

2. **Most Likely Diagnosis** — State the single most probable diagnosis
   with a short explanation.

3. **Recommended Actions** — Suggest:
   • Immediate self-care steps the patient can take
   • Diagnostic tests or examinations a clinician should order
   • When to seek emergency care (red-flag signs)

4. **Suggested Treatment** — For the most likely diagnosis, suggest
   evidence-based treatment options (both pharmacological and
   non-pharmacological where applicable).

5. **Confidence & Uncertainty** — Explicitly state your confidence level.
   If the symptoms are ambiguous, say so. NEVER present a speculative
   diagnosis as certain. Use language like "most likely", "consider",
   "cannot rule out" appropriately.

6. **Safety** — Always include a disclaimer that this is AI-generated
   guidance, not a substitute for professional medical evaluation.

7. **Output Format** — Return a JSON object with this exact structure:
{
  "differential_diagnosis": [
    {
      "condition": "...",
      "likelihood": "high|moderate|low",
      "reasoning": "..."
    }
  ],
  "most_likely_diagnosis": {
    "condition": "...",
    "explanation": "..."
  },
  "recommended_actions": {
    "self_care": ["..."],
    "diagnostic_tests": ["..."],
    "seek_emergency_if": ["..."]
  },
  "suggested_treatment": {
    "pharmacological": ["..."],
    "non_pharmacological": ["..."]
  },
  "confidence_level": "high|moderate|low",
  "disclaimer": "..."
}

8. If the symptom list is empty or nonsensical, return:
{
  "error": "Insufficient symptom information to provide a diagnosis.",
  "disclaimer": "Please consult a healthcare professional."
}
"""

_USER_PROMPT_TEMPLATE = """Analyze the following patient symptoms and provide a
structured differential diagnosis with treatment guidance.

Patient symptoms: {symptoms}

Remember: rank conditions by likelihood, express uncertainty honestly, and
include safety disclaimers.
"""


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------
_VALID_SYMPTOM_PATTERN = re.compile(r"^[a-zA-Z\s\-',/()]+$")


def _sanitize_symptoms(symptoms: list[str]) -> list[str]:
    """Remove empty, duplicate, and potentially malicious symptom entries."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for s in symptoms:
        if not isinstance(s, str):
            continue
        s = s.strip().lower()
        # Skip empty, too-long, or suspicious entries
        if not s or len(s) > 100 or not _VALID_SYMPTOM_PATTERN.match(s):
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# Core LLM call with model fallback
# ---------------------------------------------------------------------------
def _call_llm(prompt_messages: list[dict]) -> str:
    """Call the LLM with automatic model fallback chain."""
    models_with_json = ["gpt-4o", "gpt-4-turbo"]
    response = None

    for model in models_with_json:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=prompt_messages,
            )
            break
        except Exception:
            continue

    # Last resort: gpt-4 without JSON mode (prompt still requests JSON)
    if response is None:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=prompt_messages,
        )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Format structured JSON into readable text for downstream consumers
# ---------------------------------------------------------------------------
def _format_diagnosis_text(data: dict) -> str:
    """Convert the structured JSON diagnosis into clean readable text."""
    if "error" in data:
        return f"{data['error']}\n\n{data.get('disclaimer', '')}"

    lines: list[str] = []

    # Most likely diagnosis
    mld = data.get("most_likely_diagnosis", {})
    if mld:
        lines.append(f"## Most Likely Diagnosis: {mld.get('condition', 'Unknown')}")
        lines.append(mld.get("explanation", ""))
        lines.append("")

    # Differential diagnosis
    ddx = data.get("differential_diagnosis", [])
    if ddx:
        lines.append("## Differential Diagnosis")
        for i, dx in enumerate(ddx, 1):
            likelihood = dx.get("likelihood", "unknown").upper()
            lines.append(f"  {i}. **{dx.get('condition', '')}** [{likelihood}]")
            lines.append(f"     {dx.get('reasoning', '')}")
        lines.append("")

    # Recommended actions
    actions = data.get("recommended_actions", {})
    if actions:
        lines.append("## Recommended Actions")
        for label, key in [
            ("Self-Care", "self_care"),
            ("Diagnostic Tests", "diagnostic_tests"),
            ("Seek Emergency Care If", "seek_emergency_if"),
        ]:
            items = actions.get(key, [])
            if items:
                lines.append(f"  **{label}:**")
                for item in items:
                    lines.append(f"    - {item}")
        lines.append("")

    # Suggested treatment
    treatment = data.get("suggested_treatment", {})
    if treatment:
        lines.append("## Suggested Treatment")
        for label, key in [
            ("Pharmacological", "pharmacological"),
            ("Non-Pharmacological", "non_pharmacological"),
        ]:
            items = treatment.get(key, [])
            if items:
                lines.append(f"  **{label}:**")
                for item in items:
                    lines.append(f"    - {item}")
        lines.append("")

    # Confidence
    confidence = data.get("confidence_level", "")
    if confidence:
        lines.append(f"**Confidence Level:** {confidence.upper()}")
        lines.append("")

    # Disclaimer
    disclaimer = data.get("disclaimer", "")
    if disclaimer:
        lines.append(f"*{disclaimer}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API — same signature: get_diagnosis(symptoms) -> str
# ---------------------------------------------------------------------------
def get_diagnosis(symptoms: list[str]) -> str:
    """Generate a structured differential diagnosis from a list of symptoms.

    Uses GPT-4 with:
    - Input sanitisation to reject garbage/injection
    - Structured prompt for differential diagnosis + treatment
    - Model fallback chain (gpt-4o → gpt-4-turbo → gpt-4)
    - Confidence calibration and safety disclaimers
    - Graceful error handling — never crashes the pipeline

    Returns:
        A formatted diagnosis string. On failure, returns a safe error message.
    """
    # ── Guard: empty / invalid input ──────────────────────────────────────
    clean = _sanitize_symptoms(symptoms) if symptoms else []
    if not clean:
        return (
            "Insufficient symptom information to provide a diagnosis.\n\n"
            "*This is AI-generated guidance and is not a substitute for "
            "professional medical evaluation. Please consult a healthcare "
            "professional for proper diagnosis and treatment.*"
        )

    # ── Build prompt ──────────────────────────────────────────────────────
    symptom_str = ", ".join(clean)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(symptoms=symptom_str)},
    ]

    # ── Call LLM ──────────────────────────────────────────────────────────
    try:
        raw = _call_llm(messages)
        data = json.loads(raw)
        return _format_diagnosis_text(data)
    except json.JSONDecodeError:
        # LLM returned valid text but not valid JSON — return it as-is
        # (this can happen with plain gpt-4 fallback)
        logger.warning("Diagnosis LLM returned non-JSON response; using raw text.")
        return raw  # type: ignore[possibly-undefined]
    except Exception as exc:
        logger.error("Diagnosis LLM call failed: %s", exc)
        return (
            "Unable to generate a diagnosis at this time due to a service error.\n"
            "Please try again later or consult a healthcare professional.\n\n"
            "*This is AI-generated guidance and is not a substitute for "
            "professional medical evaluation.*"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ["headache", "fever"],
        ["severe chest pain", "difficulty breathing", "dizziness"],
        [],
        ["nausea", "vomiting", "abdominal pain"],
    ]
    for symptoms in test_cases:
        print(f"Symptoms: {symptoms}")
        print(get_diagnosis(symptoms))
        print("-" * 70)


