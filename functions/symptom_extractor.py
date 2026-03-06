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
# System prompt — instructs GPT-4 to behave as a clinical NLP specialist
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a clinical NLP specialist. Your ONLY job is to extract
medical symptoms from patient-reported text.

RULES — follow every one precisely:
1. Return ONLY symptoms the patient **affirms** having.
2. **Exclude** any symptom that is negated, denied, or absent.
   Negation cues include but are not limited to:
   "no", "not", "without", "denies", "deny", "absent", "negative for",
   "rules out", "ruled out", "free of", "lack of", "doesn't have",
   "does not have", "never had", "resolved".
3. Normalise each symptom to its common clinical English name
   (e.g. "pyrexia" → "fever", "cephalgia" → "headache",
    "dyspnea" → "difficulty breathing", "emesis" → "vomiting").
4. Preserve meaningful qualifiers when they add clinical value
   (e.g. "severe chest pain", "intermittent dizziness").
5. Do NOT invent symptoms that are not mentioned in the text.
6. Do NOT include diagnoses, treatments, or body parts by themselves.
7. Return the result as a JSON object with a single key "symptoms"
   whose value is an array of lowercase strings.
   Example: {"symptoms": ["fever", "severe chest pain", "nausea"]}
8. If no affirmed symptoms are found, return: {"symptoms": []}
"""

_USER_PROMPT_TEMPLATE = """Extract all affirmed symptoms from the following patient text.
Remember: exclude anything the patient denies or negates.

Patient text:
\"\"\"
{text}
\"\"\"
"""


# ---------------------------------------------------------------------------
# Primary extraction — LLM-powered (GPT-4)
# ---------------------------------------------------------------------------
def _extract_symptoms_llm(text: str) -> list[str]:
    """Use GPT-4 to extract affirmed symptoms accurately."""
    # Try models in order: gpt-4o supports JSON mode natively;
    # fall back to gpt-4-turbo, then plain gpt-4 (no JSON mode).
    models_with_json_mode = ["gpt-4o", "gpt-4-turbo"]
    response = None

    for model in models_with_json_mode:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=text)},
                ],
            )
            break
        except Exception:
            continue

    # Last resort: gpt-4 without response_format (prompt still asks for JSON)
    if response is None:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=text)},
            ],
        )

    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)

    symptoms = data.get("symptoms", [])
    # Normalise: lowercase, strip whitespace, deduplicate while preserving order
    seen = set()
    cleaned: list[str] = []
    for s in symptoms:
        s_clean = s.strip().lower()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            cleaned.append(s_clean)
    return cleaned


# ---------------------------------------------------------------------------
# Fallback extraction — enhanced regex (used when LLM call fails)
# ---------------------------------------------------------------------------
_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|denies?|deny|absent|negative\s+for|rules?\s+out|"
    r"free\s+of|lack\s+of|doesn'?t\s+have|does\s+not\s+have|never\s+had|resolved)\b",
    re.IGNORECASE,
)

_SYMPTOM_KEYWORDS: list[str] = [
    # General / constitutional
    "fever", "chills", "fatigue", "malaise", "weakness", "weight loss",
    "weight gain", "night sweats", "loss of appetite", "dehydration",
    # Pain
    "headache", "migraine", "chest pain", "abdominal pain", "back pain",
    "joint pain", "muscle pain", "neck pain", "pelvic pain", "sore throat",
    "pain", "body ache", "cramps",
    # Neurological
    "dizziness", "vertigo", "numbness", "tingling", "tremor", "seizure",
    "confusion", "memory loss", "fainting", "lightheadedness", "blurred vision",
    # Respiratory
    "cough", "shortness of breath", "difficulty breathing", "wheezing",
    "nasal congestion", "runny nose", "sneezing", "sputum",
    # Gastrointestinal
    "nausea", "vomiting", "diarrhea", "constipation", "bloating",
    "heartburn", "acid reflux", "indigestion", "blood in stool",
    # Cardiovascular
    "palpitations", "rapid heartbeat", "swelling", "edema",
    # Dermatological
    "rash", "itching", "hives", "bruising", "skin discoloration",
    # Musculoskeletal
    "stiffness", "swollen joints", "limited range of motion",
    # Psychological
    "anxiety", "depression", "insomnia", "irritability",
    # ENT / Eyes
    "ear pain", "hearing loss", "tinnitus", "eye pain", "red eyes",
    # Urinary
    "frequent urination", "painful urination", "blood in urine",
    "urinary incontinence",
]

# Sort by length descending so longer phrases match first
_SYMPTOM_KEYWORDS.sort(key=len, reverse=True)

_SYMPTOM_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _SYMPTOM_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _extract_symptoms_regex(text: str) -> list[str]:
    """Enhanced regex fallback with basic negation handling."""
    text_lower = text.lower()
    # Split into simple clauses on punctuation / conjunctions
    clauses = re.split(r"[.,;!?]|\band\b|\bbut\b|\bhowever\b", text_lower)

    affirmed: list[str] = []
    seen: set[str] = set()

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        is_negated = bool(_NEGATION_PATTERN.search(clause))
        for match in _SYMPTOM_REGEX.finditer(clause):
            symptom = match.group(0).strip().lower()
            if not is_negated and symptom not in seen:
                seen.add(symptom)
                affirmed.append(symptom)

    return affirmed


# ---------------------------------------------------------------------------
# Public API — same signature as before: extract_symptoms(text) -> list[str]
# ---------------------------------------------------------------------------
def extract_symptoms(text: str) -> list[str]:
    """Extract affirmed medical symptoms from patient-reported text.

    Uses GPT-4 for high-accuracy clinical NLP extraction.
    Falls back to an enhanced regex approach if the LLM call fails.
    """
    if not text or not text.strip():
        return []

    try:
        return _extract_symptoms_llm(text)
    except Exception as exc:
        logger.warning("LLM symptom extraction failed (%s). Using regex fallback.", exc)
        return _extract_symptoms_regex(text)


# ---------------------------------------------------------------------------
# Quick smoke-test (only runs when executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # "I have a headache and fever, but no nausea or fatigue.",
        # "No headache but nausea present.",
        # "I deny any pain or fatigue.",
        # "Patient presents with pyrexia and cephalgia.",
        # "I've been having trouble breathing and my chest feels tight.",
        # "Severe stabbing chest pain with intermittent dizziness.",
        # "I don't have any symptoms, just a general feeling of malaise.",
        # "Resolved symptoms include fever and chills, but I still have fatigue.",
        "Earlier today I had a severe headache and nausea, but after taking medication the headache completely disappeared. However, about two hours later I started feeling what might be a mild headache again, although I'm not sure if it's actually pain or just fatigue.",
        "I don't think I can say I don't have a headache, but it's not really painful either, just a strange pressure feeling."
    ]
    for t in test_cases:
        result = extract_symptoms(t)
        print(f"Input:  {t}")
        print(f"Output: {result}\n")