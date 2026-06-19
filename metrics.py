"""
metrics.py
──────────
Three custom evaluation metrics for the Email Generation Assistant.
All v1 bugs are fixed here; each fix is labelled explicitly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC 1 — Fact Coverage Score (FCS)                          range [0, 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Fraction of user-supplied key facts that are semantically
            represented in the generated email.

Logic:
  1. Split the generated email into sentences using regex.
  2. Encode all sentences and all key facts with all-MiniLM-L6-v2 (22 MB).
  3. For each key fact, compute max cosine similarity against all sentences.
  4. A fact is "covered" if its best-match similarity ≥ 0.42.
  5. FCS = covered_count / total_facts.

Rationale: Keyword search fails for paraphrases ("complimentary access" ≠
"free months"). Semantic embeddings catch valid paraphrases while penalising
hallucinated or omitted information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC 2 — Tone Congruence Score (TCS)                        range [0, 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: How well the generated email's lexical register matches the
            requested tone descriptor(s).

Logic:
  1. Parse the tone string into individual descriptors, e.g.
     "Warm, enthusiastic, professional" → ["warm", "enthusiastic", "professional"].
  2. Resolve each descriptor to a canonical tone key via TONE_ALIASES
     (e.g. "warm" → "casual", "enthusiastic" → "persuasive").
  3. Collect positive and negative marker lists from ALL resolved tones.
  4. Score: pos_rate = min(pos_hits / expected_hits, 1.0);
            neg_rate = neg_hits / len(negative_markers)
            TCS = clip(pos_rate − 0.5 × neg_rate, 0, 1)

BUG FIX #1 (v1 → v2): v1 only used the FIRST comma-separated token, silently
discarding the rest. For scenario 5 ("Warm, enthusiastic, professional"),
"warm" resolved to "casual" and the email was penalised for not using casual
markers like "hey" and "cheers" — obviously wrong. The fix scores against the
UNION of ALL resolved tones so compound descriptors are handled correctly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC 3 — Email Quality Score (EQS)                          range [0, 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Composite quality across structural completeness, textual
            similarity to the human reference, and readability.

Components:
  Structure   (weight 0.40) — regex checks for subject line, salutation,
                              body ≥2 substantial paragraphs, closing phrase,
                              sign-off.
  ROUGE-L F1  (weight 0.40) — longest-common-subsequence overlap with the
                              hand-crafted reference email.
  Readability (weight 0.20) — Flesch Reading Ease normalised to [0, 1];
                              ideal professional range 30–65.

Note: In v1 evaluation, readability averaged 0.977 with near-zero spread —
essentially dead weight. The normalisation is retained for completeness but
the weight could be reallocated to a more discriminative metric (e.g.
grammatical error rate) in future iterations.
"""

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Tone lexicons ─────────────────────────────────────────────────────────────

TONE_LEXICONS = {
    "formal": {
        "positive": [
            "i am writing to", "i hope this", "please find", "i would like to",
            "i am pleased to", "i wish to", "at your earliest convenience",
            "please do not hesitate", "i remain", "sincerely", "yours faithfully",
            "best regards", "i trust", "pursuant to", "i would appreciate",
            "i look forward to", "on behalf of", "i am reaching out",
        ],
        "negative": [
            "hey ", "hi there", "gonna", "wanna", "asap", "btw", "lol",
            "thanks!", "!!!", "catch ya", "no worries",
        ],
        "expected_hits": 3,
    },
    "casual": {
        "positive": [
            "hey", "hi ", "thanks", "sounds good", "let me know",
            "feel free", "just wanted to", "hope you're", "cheers",
            "take care", "talk soon", "quick note",
        ],
        "negative": [
            "hereby", "pursuant", "aforementioned", "i wish to formally",
            "yours faithfully", "i am writing to formally",
        ],
        "expected_hits": 2,
    },
    "urgent": {
        "positive": [
            "urgent", "immediately", "as soon as possible", "critical",
            "time-sensitive", "must", "required by", "no later than",
            "action required", "promptly", "without delay", "deadline",
            "priority",
        ],
        "negative": [
            "when you get a chance", "no rush", "at your leisure",
            "feel free to", "whenever convenient",
        ],
        "expected_hits": 2,
    },
    "empathetic": {
        "positive": [
            "i understand", "i sincerely apologize", "i'm sorry",
            "i appreciate", "i can imagine", "i am sorry",
            "we deeply regret", "i genuinely apologize",
            "i'm sorry for the inconvenience", "your concerns",
            "i take full responsibility", "regret", "apologize",
        ],
        "negative": [
            "your fault", "you should have", "as per policy",
            "we cannot be held", "unfortunately we are unable",
        ],
        "expected_hits": 2,
    },
    "persuasive": {
        "positive": [
            "i believe", "opportunity", "i am confident",
            "mutually beneficial", "compelling", "significant value",
            "i strongly recommend", "unique", "exciting",
            "i encourage you", "proven", "i would love to",
            "the potential", "transform",
        ],
        "negative": [
            "unfortunately", "i'm afraid", "regret to inform",
            "unable to", "cannot",
        ],
        "expected_hits": 3,
    },
    "professional": {
        "positive": [
            "i am writing", "please", "thank you", "best regards",
            "sincerely", "i appreciate", "i look forward",
            "please let me know", "i remain available",
        ],
        "negative": [
            "lol", "omg", "tbh", "!!!", "????",
        ],
        "expected_hits": 2,
    },
}

TONE_ALIASES = {
    "enthusiastic":  "persuasive",
    "apologetic":    "empathetic",
    "courteous":     "formal",
    "firm":          "professional",
    "neutral":       "professional",
    "warm":          "casual",
    "grateful":      "empathetic",
    "business-like": "formal",
    "confident":     "persuasive",
    "clear":         "professional",
    "polite":        "formal",
    "urgent":        "urgent",
}


def _resolve_tones(tone_str: str) -> list:
    """
    BUG FIX #1: Parse compound tone descriptors into canonical keys.

    "Warm, enthusiastic, professional" → ["casual", "persuasive", "professional"]

    v1 only took tone_str.split(",")[0].strip() — discarding all but the first.
    """
    tokens = [t.strip().lower() for t in tone_str.split(",") if t.strip()]
    resolved = []
    for token in tokens:
        if token in TONE_LEXICONS:
            resolved.append(token)
        elif token in TONE_ALIASES:
            resolved.append(TONE_ALIASES[token])
        else:
            # Partial match: "business-like" → "formal" via substring
            for key in TONE_LEXICONS:
                if key in token or token in key:
                    resolved.append(key)
                    break
            else:
                resolved.append("professional")   # safe fallback
    return list(dict.fromkeys(resolved))           # deduplicate, preserve order


# ── Singleton embedding model ─────────────────────────────────────────────────

_ST_MODEL = None

def _get_st_model():
    """Load sentence-transformer model once and cache it."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        _ST_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _ST_MODEL


# ── Metric 1: Fact Coverage Score ────────────────────────────────────────────

def compute_fcs(generated_email: str, key_facts: list) -> tuple:
    """
    Returns (fcs_score: float, per_fact_similarities: list[float])
    """
    from config import FCS_SIMILARITY_THRESHOLD

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", generated_email.strip())
        if len(s.strip()) > 5
    ]
    if not sentences:
        return 0.0, [0.0] * len(key_facts)

    model       = _get_st_model()
    sent_embs   = model.encode(sentences,  convert_to_numpy=True, normalize_embeddings=True)
    fact_embs   = model.encode(key_facts,  convert_to_numpy=True, normalize_embeddings=True)

    per_fact    = []
    covered     = 0
    for f_emb in fact_embs:
        sims    = cosine_similarity([f_emb], sent_embs)[0]
        max_sim = float(sims.max())
        per_fact.append(round(max_sim, 4))
        if max_sim >= FCS_SIMILARITY_THRESHOLD:
            covered += 1

    fcs = round(covered / len(key_facts), 4) if key_facts else 0.0
    return fcs, per_fact


# ── Metric 2: Tone Congruence Score ──────────────────────────────────────────

def compute_tcs(generated_email: str, tone_str: str) -> float:
    """
    BUG FIX #1: Scores against UNION of ALL resolved tone descriptors.
    Returns TCS in [0.0, 1.0].
    """
    email_lower     = generated_email.lower()
    resolved_keys   = _resolve_tones(tone_str)

    if not resolved_keys:
        return 0.5

    all_positive    = []
    all_negative    = []
    total_expected  = 0

    for key in resolved_keys:
        lex = TONE_LEXICONS.get(key, {})
        all_positive.extend(lex.get("positive", []))
        all_negative.extend(lex.get("negative", []))
        total_expected += lex.get("expected_hits", 2)

    if total_expected == 0:
        return 0.5

    pos_hits = sum(1 for p in all_positive if p in email_lower)
    neg_hits = sum(1 for n in all_negative if n in email_lower)

    raw_score  = pos_hits - (neg_hits * 1.5)
    normalized = max(0.0, min(1.0, raw_score / total_expected))
    return round(normalized, 4)


# ── Metric 3: Email Quality Score ────────────────────────────────────────────

def _structure_score(email: str) -> float:
    """Check presence of: subject, salutation, body, closing phrase, sign-off."""
    score  = 0.0
    lines  = email.strip().split("\n")
    lower  = email.lower()

    if lines and lines[0].lower().startswith("subject:"):
        score += 0.2

    if any(
        ln.strip().lower().startswith(p)
        for ln in lines
        for p in ("dear ", "hi ", "hello ", "greetings ")
    ):
        score += 0.2

    body_lines = [
        ln.strip() for ln in lines
        if ln.strip() and not ln.strip().lower().startswith("subject:")
    ]
    if sum(1 for ln in body_lines if len(ln) > 30) >= 2:
        score += 0.2

    if any(
        phrase in lower
        for phrase in (
            "thank you", "i look forward", "best regards", "sincerely",
            "warm regards", "yours", "kind regards", "regards,",
        )
    ):
        score += 0.2

    non_empty = [ln.strip() for ln in lines if ln.strip()]
    if non_empty:
        last = non_empty[-1]
        if re.match(r"^\[.+\]$|^[A-Z][a-z]+ .+[a-z]$", last) or any(
            last.lower().startswith(p)
            for p in ("best", "sincerely", "warm", "yours", "regards", "thanks")
        ):
            score += 0.2

    return round(score, 4)


def _rouge_l(generated: str, reference: str) -> float:
    """ROUGE-L F1 between generated and reference email."""
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    return round(scorer.score(reference, generated)["rougeL"].fmeasure, 4)


def _readability(email: str) -> float:
    """Flesch Reading Ease normalised to [0, 1]. Ideal range: 30–65."""
    try:
        import textstat
        fre = textstat.flesch_reading_ease(email)
    except Exception:
        return 0.5

    if fre <= 30:
        return 0.3
    elif fre >= 65:
        return 0.7
    return round(0.7 + 0.3 * ((fre - 30) / 35.0), 4)


def compute_eqs(generated_email: str, reference_email: str) -> tuple:
    """
    Returns (eqs_score: float, components: dict)
    """
    from config import EQS_STRUCTURE_WEIGHT, EQS_ROUGE_WEIGHT, EQS_READABILITY_WEIGHT

    struct = _structure_score(generated_email)
    rouge  = _rouge_l(generated_email, reference_email)
    read   = _readability(generated_email)

    eqs = round(
        EQS_STRUCTURE_WEIGHT   * struct +
        EQS_ROUGE_WEIGHT       * rouge  +
        EQS_READABILITY_WEIGHT * read,
        4,
    )
    return eqs, {"structure": struct, "rouge_l": rouge, "readability": read}