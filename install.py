"""
install.py
──────────
Environment bootstrap — called automatically by app.py on every run.
Safe to re-run: all operations are idempotent.

Steps:
  1. pip-install Python dependencies from requirements.txt
  2. Download NLTK punkt tokeniser data
  3. Install Ollama (if not present)
  4. Start Ollama server (if not already running)
  5. Pull both LLM models (skip if already cached)
"""

import os
import platform
import shutil
import subprocess
import sys
import time

import requests


# ── pip dependencies ──────────────────────────────────────────────────────────

def install_python_packages() -> None:
    """Install all Python packages from requirements.txt."""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_file):
        print("[install] WARNING: requirements.txt not found — skipping pip install.")
        return

    print("[install] Installing Python dependencies …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file,
         "--no-cache-dir", "-q"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"[install] pip WARNING:\n{result.stderr[-1500:]}")
    else:
        print("[install] ✓ Python packages ready.")


# ── NLTK ──────────────────────────────────────────────────────────────────────

def setup_nltk() -> None:
    """Download NLTK punkt tokeniser (required by metrics.py)."""
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            print(f"[install] Downloading NLTK '{pkg}' …")
            nltk.download(pkg, quiet=True)
    print("[install] ✓ NLTK data ready.")


# ── Ollama install ────────────────────────────────────────────────────────────

def _ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def _zstd_installed() -> bool:
    return shutil.which("zstd") is not None


def ensure_zstd() -> None:
    """
    Ollama's Linux install script ships its binary as a .tar.zst archive and
    shells out to `zstd` to unpack it. Stock Colab/Ubuntu images don't have
    zstd installed, which makes `install.sh` fail with a non-zero exit code
    (caught upstream as a CalledProcessError). Install it proactively so the
    Ollama install step doesn't fail on fresh environments.
    """
    if _zstd_installed():
        return

    print("[install] 'zstd' not found — installing via apt (required by Ollama's installer) …")
    try:
        subprocess.run(
            ["apt-get", "update", "-qq"],
            check=True, timeout=120,
        )
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "zstd"],
            check=True, timeout=120,
        )
        print("[install] ✓ zstd installed.")
    except Exception as exc:
        print(
            f"[install] WARNING: automatic zstd install failed ({exc!r}). "
            "If the Ollama install step below fails, run manually:\n"
            "    !apt-get install -y zstd"
        )


def install_ollama() -> None:
    """
    Install Ollama if not already present.

    • Linux  (Colab / Ubuntu) — official install script via curl
    • macOS                   — Homebrew
    • Windows                 — print manual instruction and exit
    """
    if _ollama_installed():
        print("[install] ✓ Ollama already installed.")
        return

    system = platform.system()
    print(f"[install] Ollama not found — installing for {system} …")

    if system == "Linux":
        ensure_zstd()
        subprocess.run(
            ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            check=True, timeout=300,
        )
    elif system == "Darwin":
        subprocess.run(["brew", "install", "ollama"], check=True, timeout=300)
    elif system == "Windows":
        print(
            "\n[install] MANUAL STEP REQUIRED:\n"
            "  Download and install Ollama from https://ollama.com/download/windows\n"
            "  Then re-run:  python app.py\n"
        )
        sys.exit(1)
    else:
        raise RuntimeError(f"Unsupported OS for automatic Ollama install: {system}")

    if not _ollama_installed():
        raise RuntimeError(
            "Ollama installation appears to have failed. "
            "Install manually from https://ollama.com"
        )
    print("[install] ✓ Ollama installed.")


# ── Ollama server ─────────────────────────────────────────────────────────────

def _ollama_running() -> bool:
    """Check if Ollama's HTTP API is reachable."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_ollama_server(wait_seconds: int = 8) -> None:
    """Start `ollama serve` as a background process if not already running."""
    if _ollama_running():
        print("[install] ✓ Ollama server already running.")
        return

    print("[install] Starting Ollama server …")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Poll until the server is up
    deadline = time.time() + wait_seconds + 10
    while time.time() < deadline:
        if _ollama_running():
            print("[install] ✓ Ollama server started.")
            return
        time.sleep(1)

    raise RuntimeError(
        "Ollama server did not start within the timeout window. "
        "Try running `ollama serve` manually in a separate terminal."
    )


# ── Model pull ────────────────────────────────────────────────────────────────

def _model_cached(tag: str) -> bool:
    """Return True if the Ollama model is already in the local cache."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = r.json().get("models", [])
        return any(m.get("name", "").startswith(tag.split(":")[0]) for m in models)
    except Exception:
        return False


def pull_model(tag: str) -> None:
    """
    Pull an Ollama model if not already cached.

    Uses `ollama pull` which streams progress to stdout so the user
    can see download progress for large models (4-5 GB each).
    """
    if _model_cached(tag):
        print(f"[install] ✓ Model '{tag}' already cached.")
        return

    print(f"[install] Pulling '{tag}' (~4-5 GB, please wait) …")
    result = subprocess.run(
        ["ollama", "pull", tag],
        timeout=1800,   # 30 minutes — large model download
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull Ollama model '{tag}'.")
    print(f"[install] ✓ Model '{tag}' ready.")


def pull_all_models(model_tags: list) -> None:
    """Pull all required Ollama models."""
    for tag in model_tags:
        pull_model(tag)


# ── LangSmith (optional) ──────────────────────────────────────────────────────

def configure_langsmith() -> None:
    """Enable LangSmith tracing if LANGCHAIN_API_KEY is set."""
    from config import LANGSMITH_API_KEY, LANGSMITH_PROJECT
    if LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"]  = "true"
        os.environ["LANGCHAIN_API_KEY"]      = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"]      = LANGSMITH_PROJECT
        print(f"[install] ✓ LangSmith tracing enabled → project '{LANGSMITH_PROJECT}'.")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("[install] ℹ  LangSmith tracing disabled (set LANGCHAIN_API_KEY to enable).")


# ── Top-level bootstrap ───────────────────────────────────────────────────────

def bootstrap(model_tags: list) -> None:
    """
    Run the complete environment bootstrap.
    Called once at startup by app.py.
    """
    print("\n" + "═" * 60)
    print("  ENVIRONMENT BOOTSTRAP")
    print("═" * 60)

    install_python_packages()
    setup_nltk()
    configure_langsmith()
    install_ollama()
    start_ollama_server()
    pull_all_models(model_tags)

    print("═" * 60)
    print("  ✓ Environment ready — starting pipeline.\n")
