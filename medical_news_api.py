"""
Medical News Credibility + AI-Generation Detector (Production-ready)
- Scrapes article text with multiple fallbacks
- Caches scraped text in SQLite with TTL
- Extracts & verifies DOIs
- Detects medical entities using biomedical NER
- Runs optional fake-news classifier (generic; weak for medical domain)
- Runs Groq LLM analysis (NO Ollama):
    - short medical summary (which disease + what they said)
    - misinformation detection
    - false info -> true remedy mapping
    - AI-generated likelihood
- Supports GROQ_API_KEY rotation/fallback from .env:
    GROQ_API_KEY=key1,key2,key3

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

# If you rely on MKL/OMP stacks and see duplicate warnings:
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import json
import time
import sqlite3
import hashlib
import logging
import asyncio
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from starlette.concurrency import run_in_threadpool
import httpx

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ------------------------------
# Logging
# ------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medical-news-detector")

# Reduce HF logs (optional)
try:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass


# ------------------------------
# .env / Config
# ------------------------------
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "medical_news_cache.sqlite3")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 3600)))  # 7 days

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))
DEFAULT_GROQ_MAX_KEY_RETRIES = int(os.getenv("GROQ_MAX_KEY_RETRIES", "3"))

DEFAULT_SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT_SECONDS", "20"))
DEFAULT_HTTP_UA = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; MedicalNewsDetector/1.0)")

# Biomedical NER (medical topic/entity detection)
BIOBERT_NER_MODEL = os.getenv("BIO_NER_MODEL", "d4data/biomedical-ner-all")

# Generic fake-news classifier (optional; can be slow/large)
FAKE_NEWS_MODEL_ID = os.getenv("FAKE_NEWS_MODEL", "jy46604790/Fake-News-Bert-Detect")


# ------------------------------
# GROQ API Key rotation
# ------------------------------
def _parse_groq_keys() -> List[str]:
    raw = (os.getenv("GROQ_API_KEY") or "").strip()
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("Missing GROQ_API_KEY in environment (.env). Provide one or multiple keys separated by commas.")
    return keys


GROQ_KEYS: List[str] = _parse_groq_keys()
_groq_key_i = 0
_groq_key_lock = asyncio.Lock()


async def _next_groq_key() -> str:
    global _groq_key_i
    async with _groq_key_lock:
        key = GROQ_KEYS[_groq_key_i % len(GROQ_KEYS)]
        _groq_key_i += 1
        return key


# ------------------------------
# SQLITE CACHE
# ------------------------------
def _now_ts() -> int:
    return int(time.time())


def _connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _connect_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                url_hash TEXT,
                final_url TEXT,
                domain TEXT,
                scraper_used TEXT,
                text TEXT,
                text_length INTEGER,
                created_at_ts INTEGER,
                updated_at_ts INTEGER
            )
        """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
        conn.commit()


def url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def domain_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        return (p.netloc or "").lower()
    except Exception:
        return ""


def cache_get(
    url: str, db_path: str = DB_PATH, ttl_seconds: int = CACHE_TTL_SECONDS
) -> Optional[Dict[str, Any]]:
    h = url_hash(url)
    with _connect_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT url, final_url, domain, scraper_used, text, text_length, updated_at_ts
            FROM articles
            WHERE url_hash = ?
        """,
            (h,),
        )
        row = cur.fetchone()

    if not row:
        return None

    (cached_url, final_url, domain, scraper_used, text, text_length, updated_at_ts) = row
    age = _now_ts() - int(updated_at_ts or 0)
    if age > ttl_seconds:
        return None

    return {
        "url": cached_url,
        "final_url": final_url,
        "domain": domain,
        "scraper_used": scraper_used,
        "text": text,
        "text_length_chars": text_length,
        "cached": True,
        "cache_age_seconds": age,
    }


def cache_upsert(url: str, final_url: str, scraper_used: str, text: str, db_path: str = DB_PATH) -> None:
    h = url_hash(url)
    domain = domain_from_url(final_url or url)
    ts = _now_ts()
    with _connect_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO articles (url, url_hash, final_url, domain, scraper_used, text, text_length, created_at_ts, updated_at_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                final_url=excluded.final_url,
                domain=excluded.domain,
                scraper_used=excluded.scraper_used,
                text=excluded.text,
                text_length=excluded.text_length,
                updated_at_ts=excluded.updated_at_ts
        """,
            (url, h, final_url, domain, scraper_used, text, len(text), ts, ts),
        )
        conn.commit()


# ------------------------------
# TEXT UTILITIES
# ------------------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[\.\?\!])\s+", clean_text(text))


# ------------------------------
# SCRAPERS (blocking; run in threadpool)
# ------------------------------
def _requests_get(url: str) -> requests.Response:
    return requests.get(
        url,
        headers={"User-Agent": DEFAULT_HTTP_UA},
        timeout=DEFAULT_SCRAPE_TIMEOUT,
        allow_redirects=True,
    )


def scrape_with_trafilatura(url: str) -> str:
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError("trafilatura fetch failed")
    extracted = trafilatura.extract(downloaded)
    if not extracted:
        raise RuntimeError("trafilatura extract empty")
    return clean_text(extracted)


def scrape_with_readability(url: str) -> str:
    from readability import Document

    r = _requests_get(url)
    r.raise_for_status()
    doc = Document(r.text)
    html = doc.summary()
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    return clean_text(text)


def scrape_with_newspaper3k(url: str) -> str:
    from newspaper import Article

    a = Article(url)
    a.download()
    a.parse()
    if not a.text:
        raise RuntimeError("newspaper3k extracted empty")
    return clean_text(a.text)


def scrape_with_bs4(url: str) -> str:
    r = _requests_get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return clean_text(text)


def get_final_url_blocking(url: str) -> str:
    try:
        r = _requests_get(url)
        return r.url
    except Exception:
        return url


def scrape_url_blocking(url: str) -> Dict[str, Any]:
    scrapers = [
        ("trafilatura", scrape_with_trafilatura),
        ("readability", scrape_with_readability),
        ("newspaper3k", scrape_with_newspaper3k),
        ("bs4", scrape_with_bs4),
    ]

    final_url = get_final_url_blocking(url)
    errors = []

    for name, fn in scrapers:
        try:
            text = fn(final_url)
            if len(text) > 400:
                return {"ok": True, "scraper": name, "text": text, "final_url": final_url}
            raise RuntimeError(f"Text too short ({len(text)} chars)")
        except Exception as e:
            errors.append({"scraper": name, "error": str(e)})

    return {"ok": False, "errors": errors, "final_url": final_url}


async def scrape_url_async(url: str) -> Dict[str, Any]:
    return await run_in_threadpool(scrape_url_blocking, url)


# ------------------------------
# DOI EXTRACTION + VERIFY (async)
# ------------------------------
def extract_dois(text: str) -> List[str]:
    doi_pattern = r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"
    dois = re.findall(doi_pattern, text, flags=re.IGNORECASE)
    return sorted(set(d.lower().rstrip(").,;]") for d in dois))


async def verify_doi_async(doi: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    try:
        r = await client.get(
            f"https://doi.org/{doi}",
            follow_redirects=True,
            timeout=15,
            headers={"User-Agent": DEFAULT_HTTP_UA},
        )
        return {"doi": doi, "status": r.status_code, "resolved_url": str(r.url)}
    except Exception as e:
        return {"doi": doi, "error": str(e)}


async def _gather_limited(callables, limit: int = 8):
    """
    Run many awaitables with concurrency limit.
    callables: list of 0-arg functions that return awaitables.
    """
    sem = asyncio.Semaphore(limit)

    async def _run_one(fn):
        async with sem:
            return await fn()

    return await asyncio.gather(*[_run_one(fn) for fn in callables])


# ------------------------------
# Model loaders (lazy, thread-safe)
# ------------------------------
_ner_lock = asyncio.Lock()
_ner_pipeline = None

_fake_lock = asyncio.Lock()
_fake_pipeline = None


async def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline
    async with _ner_lock:
        if _ner_pipeline is not None:
            return _ner_pipeline

        def _load():
            logger.info("Loading biomedical NER model: %s", BIOBERT_NER_MODEL)
            tok = AutoTokenizer.from_pretrained(BIOBERT_NER_MODEL)
            mod = AutoModelForTokenClassification.from_pretrained(BIOBERT_NER_MODEL)
            return pipeline("ner", model=mod, tokenizer=tok, aggregation_strategy="simple")

        _ner_pipeline = await run_in_threadpool(_load)
        return _ner_pipeline


async def get_fake_news_pipeline():
    global _fake_pipeline
    if _fake_pipeline is not None:
        return _fake_pipeline
    async with _fake_lock:
        if _fake_pipeline is not None:
            return _fake_pipeline

        def _load():
            logger.info("Loading fake-news classifier model: %s", FAKE_NEWS_MODEL_ID)
            return pipeline("text-classification", model=FAKE_NEWS_MODEL_ID, tokenizer=FAKE_NEWS_MODEL_ID)

        _fake_pipeline = await run_in_threadpool(_load)
        return _fake_pipeline


# ------------------------------
# Medical entity extraction (Bio NER)
# ------------------------------
async def extract_medical_keywords_async(text: str) -> Dict[str, Any]:
    """
    Uses biomedical NER to extract entities like diseases, drugs, symptoms, treatments, etc.
    """
    try:
        ner = await get_ner_pipeline()
        snippet = text[:2000]

        def _run():
            return ner(snippet)

        entities = await run_in_threadpool(_run)

        keywords = []
        detailed = []
        for e in entities:
            word = (e.get("word") or "").strip()
            if len(word) < 3:
                continue
            keywords.append(word)
            detailed.append(
                {
                    "term": word,
                    "entity_type": e.get("entity_group"),
                    "confidence": float(e.get("score", 0.0)),
                }
            )

        keywords = list(set(keywords))
        label = "MEDICAL_RELATED" if len(keywords) >= 3 else "NOT_SURE_MEDICAL"

        return {
            "label": label,
            "keyword_count": len(keywords),
            "medical_keywords": keywords[:50],
            "entities": detailed[:50],
            "model": BIOBERT_NER_MODEL,
        }
    except Exception as e:
        return {"label": "UNKNOWN", "error": str(e), "model": BIOBERT_NER_MODEL}


# ------------------------------
# Fake news classifier (optional; blocking -> threadpool)
# ------------------------------
async def run_fake_news_classifier_async(text: str) -> Dict[str, Any]:
    try:
        clf = await get_fake_news_pipeline()
        snippet = text[:3000]

        def _run():
            return clf(snippet, truncation=True)[0]

        result = await run_in_threadpool(_run)
        label_map = {"LABEL_0": "FAKE", "LABEL_1": "REAL"}
        return {
            "model": FAKE_NEWS_MODEL_ID,
            "label_raw": result.get("label"),
            "label": label_map.get(result.get("label"), result.get("label")),
            "score": float(result.get("score", 0.0)),
            "note": "Generic model. Can misclassify medical/science or crime reports.",
        }
    except Exception as e:
        return {"enabled": False, "error": str(e), "note": "Fake-news model failed to load/run."}


# ------------------------------
# Evidence highlight helpers
# ------------------------------
AI_PHRASE_PATTERNS = [
    r"\bas an ai language model\b",
    r"\bin conclusion\b",
    r"\boverall\b",
    r"\bit is important to note\b",
    r"\bdelve into\b",
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\bthis article explores\b",
    r"\bcutting-edge\b",
    r"\bgame-changer\b",
]

SUSPICIOUS_MEDICAL_PATTERNS = [
    r"\bmiracle cure\b",
    r"\binstant cure\b",
    r"\bguaranteed\b",
    r"\b100%\b",
    r"\bno side effects\b",
    r"\bsecret (?:herb|formula|treatment)\b",
    r"\bdoctors don'?t want you to know\b",
    r"\bclinically proven\b",
    r"\bnew study shows\b",
    r"\bstudies show\b",
    r"\bresearchers (?:say|claim|confirm)\b",
]

MEDICAL_RISK_PATTERNS = [
    r"\bcures? cancer\b",
    r"\breverses diabetes\b",
    r"\btreats autism\b",
    r"\bdetox\b",
    r"\bparasite cleanse\b",
    r"\banti-?vax\b|\bvaccine hoax\b",
]


def find_spans(patterns: List[str], text: str, flags=re.IGNORECASE) -> List[Dict[str, Any]]:
    spans = []
    for p in patterns:
        for m in re.finditer(p, text, flags):
            spans.append({"pattern": p, "match": m.group(0), "start": m.start(), "end": m.end()})
    spans.sort(key=lambda x: (x["start"], x["end"]))
    return spans


def highlight_text(text: str, spans: List[Dict[str, Any]], window: int = 90, cap: int = 50) -> List[Dict[str, Any]]:
    out = []
    for s in spans[:cap]:
        start = max(0, s["start"] - window)
        end = min(len(text), s["end"] + window)
        snippet = text[start:end]

        rel_s = s["start"] - start
        rel_e = s["end"] - start
        marked = snippet[:rel_s] + "<<<" + snippet[rel_s:rel_e] + ">>>" + snippet[rel_e:]

        out.append({"match": s["match"], "pattern": s["pattern"], "snippet": marked})
    return out


# ------------------------------
# AI text heuristics + likeness
# ------------------------------
def ai_text_heuristics(text: str) -> Dict[str, Any]:
    t = text.lower()
    signals = []

    words = re.findall(r"\b\w+\b", t)
    if len(words) < 150:
        signals.append("Very short article")

    if words:
        ratio = len(set(words)) / len(words)
        if ratio < 0.35:
            signals.append("High repetition (low vocabulary variety)")

    if "as an ai language model" in t:
        signals.append("AI artifact phrase detected")

    generic = ["in conclusion", "overall", "it is important to note"]
    if sum(p in t for p in generic) >= 2:
        signals.append("Multiple generic filler phrases")

    spans = find_spans(AI_PHRASE_PATTERNS, text)
    evidence = highlight_text(text, spans)

    return {
        "signals": signals,
        "flag": len(signals) >= 2,
        "evidence_snippets": evidence,
        "evidence_spans": spans[:200],
    }


def ai_likeness_score(text: str) -> Dict[str, Any]:
    try:
        import statistics
    except Exception:
        statistics = None

    sentences = split_sentences(text)
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {"score": 0, "label": "UNKNOWN", "reasons": ["Empty text"]}

    uniq_ratio = len(set(words)) / len(words)

    sent_lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences if s.strip()]
    var = None
    if statistics and len(sent_lens) >= 2:
        try:
            var = float(statistics.pvariance(sent_lens))
        except Exception:
            var = None

    boiler_hits = [p for p in AI_PHRASE_PATTERNS if re.search(p, text, re.I)]

    score = 0
    reasons = []

    if uniq_ratio < 0.38:
        score += 1
        reasons.append("Low vocabulary variety")

    if len(boiler_hits) >= 2:
        score += 1
        reasons.append("Multiple boilerplate phrases")

    if var is not None and var < 20:
        score += 1
        reasons.append("Sentence lengths unusually uniform")

    if re.search(r"\bas an ai language model\b", text, re.I):
        score += 3
        reasons.append("Explicit AI artifact phrase")

    if score >= 4:
        label = "HIGH_AI_LIKENESS"
    elif score >= 2:
        label = "MEDIUM_AI_LIKENESS"
    else:
        label = "LOW_AI_LIKENESS"

    return {
        "score": score,
        "label": label,
        "reasons": reasons,
        "uniq_ratio": float(uniq_ratio),
        "sentence_len_variance": var,
        "boilerplate_patterns_hit": boiler_hits[:20],
    }


# ------------------------------
# Medical misinfo heuristics
# ------------------------------
def medical_misinfo_heuristics(text: str) -> Dict[str, Any]:
    spans = find_spans(SUSPICIOUS_MEDICAL_PATTERNS + MEDICAL_RISK_PATTERNS, text)
    evidence = highlight_text(text, spans)
    signals = []
    if len(spans) >= 2:
        signals.append("Multiple suspicious medical-claim phrases")
    if any(re.search(p, text, re.I) for p in MEDICAL_RISK_PATTERNS):
        signals.append("High-risk medical claim pattern detected")
    return {
        "signals": signals,
        "flag": len(signals) >= 1,
        "evidence_snippets": evidence,
        "evidence_spans": spans[:200],
    }


# ------------------------------
# Text mistakes checks
# ------------------------------
def text_mistakes_checks(text: str) -> Dict[str, Any]:
    issues = []

    if re.search(r"[!?]{3,}", text):
        issues.append("Excessive punctuation (!!! / ???)")

    if re.search(r"\.{3,}", text):
        issues.append("Repeated dots / ellipsis spam (...)")

    caps_words = re.findall(r"\b[A-Z]{6,}\b", text)
    if len(caps_words) >= 10:
        issues.append("Many long ALL-CAPS words (spammy style)")

    if text.count("(") != text.count(")"):
        issues.append("Unbalanced parentheses")

    if text.count("[") != text.count("]"):
        issues.append("Unbalanced brackets")

    if "doi" in text.lower() and not extract_dois(text):
        issues.append("Mentions DOI but no valid DOI pattern found")

    sentences = split_sentences(text)
    long_sents = [s for s in sentences if len(re.findall(r"\b\w+\b", s)) >= 45]
    if len(long_sents) >= 5:
        issues.append("Many very long sentences (hard to read / suspicious style)")

    return {"issues": issues, "flag": len(issues) >= 2}


# ------------------------------
# GROQ ANALYSIS (NO OLLAMA)
# ------------------------------
async def groq_analyze_async(
    text: str,
    model: str = DEFAULT_GROQ_MODEL,
    timeout: int = DEFAULT_GROQ_TIMEOUT,
    max_retries_keys: int = DEFAULT_GROQ_MAX_KEY_RETRIES,
) -> Dict[str, Any]:
    """
    Calls Groq LLM and returns strict JSON analysis:
    - which disease + what they said summary
    - misinformation detection
    - false claims + correct remedy
    - ai_generated likelihood
    """
    prompt = f"""
Respond ONLY with valid JSON. No markdown.

Return JSON with keys:
medical_topic (boolean)
diseases (array of strings)
short_summary (string)                 // 1-3 lines: which disease + what they said
is_medical_misinformation (boolean)
false_claims (array of objects) where each object has:
  - false_info (string)               // what is false in the article
  - why_false (string)
  - true_remedy (string)              // correct remedy / what to do instead (safe, evidence-based)
  - urgency (string)                  // LOW|MEDIUM|HIGH (HIGH if dangerous advice)
what_to_verify (array of strings)      // what the reader should verify from trusted sources
ai_generated_likelihood (string)       // LOW|MEDIUM|HIGH

Rules:
- If no clear disease/condition, keep diseases empty but still summarize what the article claims.
- If no misinformation, set is_medical_misinformation=false and false_claims=[].
- Remedies must be general safe guidance, not personalized medical advice. Suggest consulting a clinician when appropriate.

Article:
{text[:7000]}
""".strip()

    url = "https://api.groq.com/openai/v1/chat/completions"

    attempts = min(max_retries_keys, max(1, len(GROQ_KEYS)))
    last_err: Optional[str] = None

    for _ in range(attempts):
        api_key = await _next_groq_key()
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json=payload)

            # Rotate on auth/rate/temporary server issues
            if r.status_code in (401, 403):
                last_err = f"Groq auth error with one key: HTTP {r.status_code}"
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"Groq temporary error: HTTP {r.status_code} | {r.text[:300]}"
                continue

            r.raise_for_status()
            data = r.json()
            content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            content = (content or "").strip()

            try:
                parsed = json.loads(content)
                return {"enabled": True, "provider": "groq", "model": model, "analysis": parsed}
            except Exception:
                return {
                    "enabled": True,
                    "provider": "groq",
                    "model": model,
                    "raw_response": content[:6000],
                    "warning": "Groq did not return strict JSON",
                }
        except Exception as e:
            last_err = str(e)
            continue

    return {"enabled": False, "provider": "groq", "model": model, "error": last_err or "Unknown Groq error"}


# ------------------------------
# FINAL VERDICT
# ------------------------------
def final_verdict(part_a: Dict[str, Any]) -> Dict[str, Any]:
    cred_score = 0
    notes = []

    med = part_a.get("medical_relevance", {})
    is_medical = isinstance(med, dict) and med.get("label") == "MEDICAL_RELATED"
    notes.append("Medical-related content detected." if is_medical else "Not strongly medical-related by NER threshold (may still be medical).")

    doi_checks = part_a.get("doi_verification", [])
    doi_ok = any(isinstance(c, dict) and c.get("status") in (200, 301, 302) for c in doi_checks)
    if doi_ok:
        cred_score -= 2
        notes.append("Resolvable DOI found (credibility +).")
    else:
        notes.append("No resolvable DOI found (neutral for many news sites).")

    mm = part_a.get("medical_misinfo_heuristics", {})
    if isinstance(mm, dict) and mm.get("flag") is True:
        cred_score += 2
        notes.append("Heuristics flagged suspicious medical-claim patterns.")

    groq = part_a.get("groq_report", {})
    analysis = groq.get("analysis", {}) if isinstance(groq, dict) else {}
    if isinstance(analysis, dict) and analysis.get("is_medical_misinformation") is True:
        cred_score += 3
        notes.append("Groq: likely medical misinformation.")
    elif isinstance(analysis, dict) and analysis.get("is_medical_misinformation") is False:
        cred_score -= 1
        notes.append("Groq: not medical misinformation.")

    clf = part_a.get("fake_news_classifier", {})
    if isinstance(clf, dict) and clf.get("label") == "FAKE":
        cred_score += 1
        notes.append("Generic fake-news classifier voted FAKE (weak for medical domain).")
    elif isinstance(clf, dict) and clf.get("label") == "REAL":
        cred_score -= 1
        notes.append("Generic fake-news classifier voted REAL (weak for medical domain).")

    mistakes = part_a.get("text_mistakes_checks", {})
    if isinstance(mistakes, dict) and mistakes.get("flag") is True:
        cred_score += 1
        notes.append("Text mistakes checks flagged spammy/sloppy patterns.")

    if cred_score >= 4:
        credibility = "HIGH_RISK_MEDICAL_MISINFORMATION"
    elif cred_score >= 2:
        credibility = "POSSIBLE_MEDICAL_MISINFORMATION"
    else:
        credibility = "LOW_EVIDENCE_OF_MEDICAL_MISINFORMATION"

    ai = part_a.get("ai_likeness", {})
    ai_score = int(ai.get("score", 0)) if isinstance(ai, dict) else 0

    groq_ai = None
    if isinstance(analysis, dict):
        groq_ai = analysis.get("ai_generated_likelihood")
    if isinstance(groq_ai, str):
        if groq_ai.upper() == "HIGH":
            ai_score += 2
        elif groq_ai.upper() == "MEDIUM":
            ai_score += 1

    if ai_score >= 5:
        ai_generation = "LIKELY_AI_GENERATED"
    elif ai_score >= 3:
        ai_generation = "POSSIBLY_AI_GENERATED"
    else:
        ai_generation = "UNLIKELY_AI_GENERATED"

    return {
        "medical_misinformation_risk": credibility,
        "risk_score": cred_score,
        "ai_generation": ai_generation,
        "ai_score": ai_score,
        "notes": notes,
    }


# ------------------------------
# MAIN ANALYSIS (ASYNC, WITH CACHE)
# ------------------------------
async def analyze_async(
    url: str,
    enable_fake_news_classifier: bool = True,
    enable_groq: bool = True,
    groq_model: str = DEFAULT_GROQ_MODEL,
    use_cache: bool = True,
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    db_path: str = DB_PATH,
    include_scraped_text_in_output: bool = True,
) -> Dict[str, Any]:
    init_db(db_path)

    if use_cache:
        cached = await run_in_threadpool(cache_get, url, db_path, cache_ttl_seconds)
        if cached and cached.get("text"):
            text = cached["text"]
            final_url = cached.get("final_url", url)
            part_a = await _run_analysis_pipeline_async(
                url=url,
                final_url=final_url,
                domain=cached.get("domain", domain_from_url(final_url)),
                scraper_used=cached.get("scraper_used", "cache"),
                text=text,
                enable_fake_news_classifier=enable_fake_news_classifier,
                enable_groq=enable_groq,
                groq_model=groq_model,
                include_scraped_text_in_output=include_scraped_text_in_output,
                from_cache=True,
                cache_age_seconds=cached.get("cache_age_seconds"),
            )
            return {"ok": True, "part_a": part_a}

    scrape_result = await scrape_url_async(url)
    if not scrape_result["ok"]:
        return {"ok": False, "scrape_errors": scrape_result["errors"], "final_url": scrape_result.get("final_url")}

    text = scrape_result["text"]
    final_url = scrape_result.get("final_url", url)
    scraper_used = scrape_result.get("scraper", "unknown")
    domain = domain_from_url(final_url)

    await run_in_threadpool(cache_upsert, url, final_url, scraper_used, text, db_path)

    part_a = await _run_analysis_pipeline_async(
        url=url,
        final_url=final_url,
        domain=domain,
        scraper_used=scraper_used,
        text=text,
        enable_fake_news_classifier=enable_fake_news_classifier,
        enable_groq=enable_groq,
        groq_model=groq_model,
        include_scraped_text_in_output=include_scraped_text_in_output,
        from_cache=False,
        cache_age_seconds=None,
    )

    return {"ok": True, "part_a": part_a}


async def _run_analysis_pipeline_async(
    url: str,
    final_url: str,
    domain: str,
    scraper_used: str,
    text: str,
    enable_fake_news_classifier: bool,
    enable_groq: bool,
    groq_model: str,
    include_scraped_text_in_output: bool,
    from_cache: bool,
    cache_age_seconds: Optional[int],
) -> Dict[str, Any]:
    dois = extract_dois(text)

    async with httpx.AsyncClient(timeout=20) as client:
        doi_checks = await _gather_limited([lambda d=d: verify_doi_async(d, client) for d in dois], limit=8)

    med_rel = await extract_medical_keywords_async(text)

    part_a: Dict[str, Any] = {
        "input_url": url,
        "final_url": final_url,
        "source_domain": domain,
        "scraper_used": scraper_used,
        "from_cache": from_cache,
        "cache_age_seconds": cache_age_seconds,
        "text_length_chars": len(text),
        "scraped_text_preview": text[:1200] + ("..." if len(text) > 1200 else ""),
    }

    if include_scraped_text_in_output:
        part_a["scraped_text_full"] = text

    part_a["medical_relevance"] = med_rel
    part_a["doi_found"] = dois
    part_a["doi_verification"] = doi_checks

    if enable_fake_news_classifier:
        part_a["fake_news_classifier"] = await run_fake_news_classifier_async(text)
    else:
        part_a["fake_news_classifier"] = {"enabled": False, "note": "Set enable_fake_news_classifier=True to run transformers model."}

    part_a["ai_text_heuristics"] = ai_text_heuristics(text)
    part_a["ai_likeness"] = ai_likeness_score(text)
    part_a["medical_misinfo_heuristics"] = medical_misinfo_heuristics(text)
    part_a["text_mistakes_checks"] = text_mistakes_checks(text)

    # Groq analysis for:
    #  1) short summary (disease + what they said)
    #  2) false info -> true remedy list
    if enable_groq:
        part_a["groq_report"] = await groq_analyze_async(text, model=groq_model)
    else:
        part_a["groq_report"] = {"enabled": False, "note": "Set enable_groq=True to run Groq LLM analysis."}

    groq_analysis = {}
    if isinstance(part_a.get("groq_report"), dict):
        groq_analysis = part_a["groq_report"].get("analysis") or {}

    part_a["ai_short_summary"] = {
        "diseases": groq_analysis.get("diseases", []),
        "short_summary": groq_analysis.get("short_summary", ""),
        "medical_topic": groq_analysis.get("medical_topic"),
    }

    part_a["false_info_and_remedies"] = groq_analysis.get("false_claims", [])
    part_a["what_to_verify"] = groq_analysis.get("what_to_verify", [])

    part_a["final_verdict"] = final_verdict(part_a)
    part_a["generated_at"] = datetime.now().isoformat(timespec="seconds")

    return part_a


# ------------------------------
# FASTAPI APP
# ------------------------------
app = FastAPI(title="Medical News Credibility + AI-Generation Detector", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: HttpUrl

    use_cache: bool = True
    cache_ttl_seconds: int = CACHE_TTL_SECONDS
    include_full_text: bool = True

    enable_fake_news_classifier: bool = True

    enable_groq: bool = True
    groq_model: str = DEFAULT_GROQ_MODEL


class AnalyzeResponse(BaseModel):
    ok: bool
    part_a: Optional[Dict[str, Any]] = None
    scrape_errors: Optional[List[Dict[str, Any]]] = None
    final_url: Optional[str] = None


@app.on_event("startup")
async def _startup():
    # Ensure DB exists
    await run_in_threadpool(init_db, DB_PATH)
    logger.info("Startup complete. DB ready at %s", DB_PATH)


@app.get("/health")
async def health():
    return {"ok": True, "service": app.title, "ts": datetime.now().isoformat(timespec="seconds")}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(req: AnalyzeRequest):
    try:
        result = await analyze_async(
            url=str(req.url),
            enable_fake_news_classifier=req.enable_fake_news_classifier,
            enable_groq=req.enable_groq,
            groq_model=req.groq_model,
            use_cache=req.use_cache,
            cache_ttl_seconds=req.cache_ttl_seconds,
            db_path=DB_PATH,
            include_scraped_text_in_output=req.include_full_text,
        )
        if not result.get("ok"):
            return AnalyzeResponse(ok=False, scrape_errors=result.get("scrape_errors"), final_url=result.get("final_url"))
        return AnalyzeResponse(ok=True, part_a=result.get("part_a"))
    except Exception as e:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=str(e))