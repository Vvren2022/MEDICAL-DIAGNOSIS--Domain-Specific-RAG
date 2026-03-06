import os
import re
import time
import logging
import warnings
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Optional NCBI API key — raises rate limit from 3/sec to 10/sec
_NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

_HEADERS = {"User-Agent": "ClinisightAI/1.0 (medical-diagnosis-tool)"}
_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5  # seconds


# ---------------------------------------------------------------------------
# Query builder — turns raw symptom text into a proper PubMed query
# ---------------------------------------------------------------------------
def _build_pubmed_query(raw_query: str) -> str:
    """Convert a raw symptom string into a proper PubMed boolean query.

    Example: 'fever headache nausea' →
             '(fever AND headache AND nausea) AND (diagnosis OR treatment OR clinical)'
    """
    raw_query = raw_query.strip()
    if not raw_query:
        return ""

    # Split on common delimiters and strip junk
    tokens = re.split(r"[,;]+|\band\b|\bor\b", raw_query, flags=re.IGNORECASE)
    terms = [t.strip().lower() for t in tokens if t.strip()]

    if not terms:
        return ""

    # Wrap multi-word terms in quotes for exact matching
    quoted = [f'"{t}"' if " " in t else t for t in terms]

    # Join with AND + add a clinical context filter
    symptom_clause = " AND ".join(quoted)
    return f"({symptom_clause}) AND (diagnosis OR treatment OR clinical)"


# ---------------------------------------------------------------------------
# Retry-aware HTTP GET
# ---------------------------------------------------------------------------
def _get_with_retry(url: str, params: dict, timeout: int = 15) -> requests.Response:
    """GET with exponential backoff and retry for transient failures."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, headers=_HEADERS, timeout=timeout
            )
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "PubMed request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, _MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------
def _parse_article(article_xml, pmid: str) -> dict:
    """Parse a single <PubmedArticle> XML element into a clean dict."""
    title_tag = article_xml.find("articletitle")
    abstract_tag = article_xml.find("abstract")
    date_tag = article_xml.find("pubdate")
    author_tags = article_xml.find_all("author")

    # Title
    title = title_tag.get_text(strip=True) if title_tag else "No title"

    # Abstract
    abstract = (
        abstract_tag.get_text(separator=" ", strip=True)
        if abstract_tag
        else "No abstract available"
    )

    # Authors
    authors: list[str] = []
    for author in author_tags:
        last = author.find("lastname")
        fore = author.find("forename")
        if last and fore:
            authors.append(f"{fore.get_text()} {last.get_text()}")
        elif last:
            authors.append(last.get_text())
    if not authors:
        authors = ["No authors listed"]

    # Publication date
    pub_date = "No date"
    if date_tag:
        year = date_tag.find("year")
        month = date_tag.find("month")
        if year and month:
            pub_date = f"{month.get_text()} {year.get_text()}"
        elif year:
            pub_date = year.get_text()

    # URL
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publication_date": pub_date,
        "article_url": url,
    }


# ---------------------------------------------------------------------------
# Format articles into clean text for downstream LLM summarisation
# ---------------------------------------------------------------------------
def format_articles_as_text(articles: list[dict]) -> str:
    """Convert a list of article dicts into a clean, LLM-friendly text block.

    This is intended to be passed to summarize_text() so the LLM receives
    properly formatted article content rather than raw Python dicts.
    """
    if not articles:
        return "No relevant PubMed articles were found for the given symptoms."

    parts: list[str] = []
    for i, art in enumerate(articles, 1):
        authors_str = ", ".join(art.get("authors", ["Unknown"]))
        parts.append(
            f"--- Article {i} ---\n"
            f"Title: {art.get('title', 'N/A')}\n"
            f"Authors: {authors_str}\n"
            f"Date: {art.get('publication_date', 'N/A')}\n"
            f"URL: {art.get('article_url', 'N/A')}\n"
            f"Abstract: {art.get('abstract', 'N/A')}\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API — fetch_pubmed_articles_with_metadata(query, ...) -> list[dict]
# ---------------------------------------------------------------------------
def fetch_pubmed_articles_with_metadata(
    query: str,
    max_results: int = 3,
) -> list[dict]:
    """Search PubMed and return structured article metadata.

    Improvements over original:
    - Smart boolean query construction for higher relevance
    - Retry with exponential backoff for transient failures
    - Input validation (empty / garbage queries)
    - NCBI API key support (optional, via NCBI_API_KEY env var)
    - PMID extracted from XML (not zip-aligned with search IDs)
    - Structured logging instead of print()
    - No misleading mock data — returns empty list on failure

    Returns:
        list[dict] — each dict has keys: title, abstract, authors,
                     publication_date, article_url
    """
    # ── Input validation ──────────────────────────────────────────────
    if not query or not query.strip():
        logger.info("Empty query — skipping PubMed search.")
        return []

    pubmed_query = _build_pubmed_query(query)
    if not pubmed_query:
        logger.info("Query produced no valid search terms.")
        return []

    logger.info("PubMed query: %s", pubmed_query)

    # ── Step 1: Search for article IDs ────────────────────────────────
    search_params: dict = {
        "db": "pubmed",
        "term": pubmed_query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if _NCBI_API_KEY:
        search_params["api_key"] = _NCBI_API_KEY

    try:
        search_resp = _get_with_retry(_SEARCH_URL, search_params)
        search_data = search_resp.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        logger.info("PubMed returned %d article IDs: %s", len(id_list), id_list)

        if not id_list:
            logger.warning("No PubMed articles found for query: %s", pubmed_query)
            return []

    except Exception as exc:
        logger.error("PubMed search failed: %s", exc)
        return []

    # ── Step 2: Fetch full article metadata ───────────────────────────
    fetch_params: dict = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
    }
    if _NCBI_API_KEY:
        fetch_params["api_key"] = _NCBI_API_KEY

    try:
        fetch_resp = _get_with_retry(_FETCH_URL, fetch_params)
        soup = BeautifulSoup(fetch_resp.text, "lxml")
        articles_xml = soup.find_all("pubmedarticle")
        logger.info("Parsed %d articles from XML.", len(articles_xml))

    except Exception as exc:
        logger.error("PubMed fetch failed: %s", exc)
        return []

    # ── Step 3: Parse each article — extract PMID from XML itself ─────
    articles: list[dict] = []
    for article_xml in articles_xml:
        # Extract PMID from the XML element (not from zip with search IDs)
        pmid_tag = article_xml.find("pmid")
        pmid = pmid_tag.get_text(strip=True) if pmid_tag else "unknown"
        try:
            parsed = _parse_article(article_xml, pmid)
            articles.append(parsed)
        except Exception as exc:
            logger.warning("Failed to parse article PMID=%s: %s", pmid, exc)
            continue

    logger.info("Successfully parsed %d/%d articles.", len(articles), len(articles_xml))
    return articles


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_queries = [
        "fever headache",
        "severe chest pain difficulty breathing",
        "",
        "nausea vomiting abdominal pain",
    ]
    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: '{q}'")
        print("=" * 70)
        results = fetch_pubmed_articles_with_metadata(q)
        print(f"Articles returned: {len(results)}")
        for art in results:
            print(f"  - {art['title']}")
            print(f"    {art['article_url']}")
        # Show formatted text output
        print(f"\nFormatted text (first 300 chars):")
        print(format_articles_as_text(results)[:300])
