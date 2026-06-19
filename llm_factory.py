"""
llm_factory.py
──────────────
Creates LiteLLM-compatible LLM identifiers for CrewAI agents backed by
locally-downloaded Ollama models.

Why Ollama instead of HuggingFace Inference API?
  • No monthly rate limits — unlimited local inference
  • Models run as GGUF (Q4_K_M by default): 4.4 GB for Qwen2.5-7B,
    4.1 GB for Mistral-7B — both fit comfortably in Colab T4 (15 GB VRAM)
  • Ollama exposes an OpenAI-compatible REST API at localhost:11434/v1
  • LiteLLM routes CrewAI agent calls to Ollama transparently using the
    string prefix "ollama/<model-tag>"

CrewAI ↔ LiteLLM ↔ Ollama ↔ local GGUF model
"""

import os
import time

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_API_BASE,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)


def _ollama_healthy() -> bool:
    """Ping Ollama's health endpoint."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_ollama(timeout: int = 30) -> None:
    """Block until Ollama server is responsive (called after starting it)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _ollama_healthy():
            return
        time.sleep(1)
    raise RuntimeError(
        f"Ollama server not reachable after {timeout}s. "
        "Run `ollama serve` manually or re-run app.py."
    )


def create_llm(ollama_tag: str) -> str:
    """
    Return a LiteLLM-compatible model identifier for a local Ollama model.

    CrewAI's Agent class accepts this string directly and routes all
    calls through LiteLLM, which forwards them to Ollama's local API.

    LiteLLM env vars set here ensure correct routing and parameters:
      OLLAMA_API_BASE  — tells LiteLLM where Ollama is listening
      LLM_TEMPERATURE  — generation temperature
      LLM_MAX_TOKENS   — max tokens per completion

    Parameters
    ----------
    ollama_tag : str
        Ollama model tag, e.g. "qwen2.5:7b" or "mistral:7b"

    Returns
    -------
    str
        LiteLLM model string, e.g. "ollama/qwen2.5:7b"
    """
    # Set LiteLLM routing env vars
    os.environ["OLLAMA_API_BASE"]  = OLLAMA_API_BASE
    os.environ["LLM_TEMPERATURE"]  = str(LLM_TEMPERATURE)
    os.environ["LLM_MAX_TOKENS"]   = str(LLM_MAX_TOKENS)
    os.environ["LLM_REQUEST_TIMEOUT"] = str(LLM_TIMEOUT)

    return f"ollama/{ollama_tag}"


def verify_model_available(ollama_tag: str) -> bool:
    """
    Check that a specific model is present in the Ollama cache.

    Returns True if found, False otherwise.
    Raises RuntimeError if Ollama itself is not reachable.
    """
    if not _ollama_healthy():
        raise RuntimeError(
            "Ollama server is not running. "
            "Call install.start_ollama_server() first."
        )

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        models = r.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        # Ollama tags can appear as "qwen2.5:7b" or "qwen2.5:7b-instruct-q4_K_M"
        base_name = ollama_tag.split(":")[0]
        return any(base_name in name for name in model_names)
    except Exception as e:
        raise RuntimeError(f"Failed to query Ollama model list: {e}")


def list_cached_models() -> list:
    """Return a list of all models currently cached in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []