"""
config.py
─────────
Central configuration for the Email Generation Assistant.
All tunable constants live here. No magic numbers elsewhere.

Model Strategy
──────────────
Both models are served LOCALLY via Ollama (GGUF quantized).
This eliminates HuggingFace Inference API rate limits entirely and
runs fine on a free Colab T4 (15 GB VRAM) or any CPU-only machine.

  Model A — qwen2.5:7b   (~4.4 GB, Q4_K_M GGUF)
            Qwen 2.5 7B Instruct — SOTA open-source chat model,
            exceptional instruction following and tone calibration.

  Model B — mistral:7b   (~4.1 GB, Q4_K_M GGUF)
            Mistral 7B Instruct v0.3 — strong architecture-family
            counterpart for a meaningful head-to-head comparison.

Both are LiteLLM-routed through Ollama's OpenAI-compatible local API,
so CrewAI's agent framework works without any API key.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Ollama configuration ──────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_API_BASE  = f"{OLLAMA_BASE_URL}/v1"          # OpenAI-compatible endpoint
OLLAMA_SERVE_WAIT = 8                               # seconds after starting server

# ── Model registry ────────────────────────────────────────────────────────────
# Keys are display names used in reports and tables.
# Values are Ollama model tags.
MODELS = {
    "Model-A (Qwen2.5-7B)":  "qwen2.5:7b",
    "Model-B (Mistral-7B)":   "mistral:7b",
}

# LiteLLM prefix for Ollama routing (CrewAI reads this)
def litellm_id(ollama_tag: str) -> str:
    """Return the LiteLLM model string for a given Ollama tag."""
    return f"ollama/{ollama_tag}"

# ── LLM generation parameters ─────────────────────────────────────────────────
LLM_TEMPERATURE     = 0.65     # slightly lower for consistency with 7B models
LLM_MAX_TOKENS      = 1200     # headroom for long emails + thinking blocks
LLM_TOP_P           = 0.90
LLM_TIMEOUT         = 180      # seconds per CrewAI task

# ── CrewAI / pipeline ─────────────────────────────────────────────────────────
MAX_RETRIES         = 3
RETRY_BASE_DELAY    = 6        # seconds; doubles on each retry
INTER_SCENARIO_WAIT = 1        # seconds between scenarios (no rate limit needed locally)

# ── Evaluation ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL          = "all-MiniLM-L6-v2"   # 22 MB, runs on CPU
FCS_SIMILARITY_THRESHOLD = 0.42
EQS_STRUCTURE_WEIGHT     = 0.40
EQS_ROUGE_WEIGHT         = 0.40
EQS_READABILITY_WEIGHT   = 0.20

# ── Output files ──────────────────────────────────────────────────────────────
CSV_PATH          = RESULTS_DIR / "evaluation_scores.csv"
JSON_PATH         = RESULTS_DIR / "evaluation_report.json"
EMAILS_PATH       = RESULTS_DIR / "generated_emails.json"
ANALYSIS_MD_PATH  = RESULTS_DIR / "comparative_analysis.md"

# ── LangSmith (optional observability) ────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = "email-assistant-eval"