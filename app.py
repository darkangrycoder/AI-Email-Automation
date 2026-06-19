#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            EMAIL GENERATION ASSISTANT  ·  Agentic Architecture             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Entry point for the AI Engineer screening assessment.

Architecture
────────────
  LLM backend  : Ollama (local GGUF models — no API key required)
  Agent layer  : CrewAI 4-agent sequential pipeline
                 Planner → Writer → Critic → Refiner
  Prompt method: R-FSCoT (Role + 3-shot Few-Shot + Structured Chain-of-Thought)
  Evaluation   : 3 custom metrics (FCS, TCS, EQS) across 10 scenarios × 2 models
  Observability: LangSmith (optional — set LANGCHAIN_API_KEY env var)

Models (both served locally via Ollama, ~4 GB each)
────────────────────────────────────────────────────
  Model A — Qwen/Qwen2.5-7B  (qwen2.5:7b)
            SOTA open-source 7B, exceptional instruction following
  Model B — Mistral-7B-Instruct-v0.3 (mistral:7b)
            Strong architecture-family counterpart for head-to-head comparison

Usage
─────
  python app.py                    # full pipeline (generate + evaluate)
  python app.py --skip-generation  # re-evaluate from saved emails
  python app.py --demo             # interactive single-email demo (no eval)
  python app.py --model-a qwen2.5:7b --model-b mistral:7b  # override models

VRAM requirement: ~4.5 GB per model (Q4_K_M GGUF), well within Colab T4 (15 GB).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
  ███████╗███╗   ███╗ █████╗ ██╗██╗         █████╗ ██╗
  ██╔════╝████╗ ████║██╔══██╗██║██║        ██╔══██╗██║
  █████╗  ██╔████╔██║███████║██║██║        ███████║██║
  ██╔══╝  ██║╚██╔╝██║██╔══██║██║██║        ██╔══██║██║
  ███████╗██║ ╚═╝ ██║██║  ██║██║███████╗   ██║  ██║██║
  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝  ╚═╝╚═╝
  Email Generation Assistant  ·  Agentic Architecture
  CrewAI · Ollama · R-FSCoT · FCS / TCS / EQS Evaluation
"""


def print_banner() -> None:
    print(BANNER)
    print(f"  {'─'*58}")
    print(f"  Started : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  {'─'*58}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Email Generation Assistant — Agentic Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip LLM generation and re-run evaluation from saved emails.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Interactive demo: generate a single email and exit.",
    )
    parser.add_argument(
        "--model-a",
        default=None,
        metavar="OLLAMA_TAG",
        help="Override Model A (default: qwen2.5:7b).",
    )
    parser.add_argument(
        "--model-b",
        default=None,
        metavar="OLLAMA_TAG",
        help="Override Model B (default: mistral:7b).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip Ollama model pull (assume models already cached).",
    )
    return parser.parse_args()


# ── Demo mode ─────────────────────────────────────────────────────────────────

def run_demo(model_tag: str) -> None:
    """Generate a single email interactively."""
    from llm_factory import create_llm
    from agents import build_agents, generate_email

    print("\n" + "═" * 60)
    print("  DEMO MODE — Interactive Email Generation")
    print("═" * 60)
    intent    = input("  Intent (e.g. 'Follow up after interview'): ").strip()
    raw_facts = input("  Key Facts (comma-separated): ").strip()
    tone      = input("  Tone (e.g. 'Professional, warm'): ").strip()
    print()

    facts   = [f.strip() for f in raw_facts.split(",") if f.strip()]
    llm_id  = create_llm(model_tag)
    agents  = build_agents(llm_id)
    scenario = {"id": 0, "intent": intent, "key_facts": facts, "tone": tone}

    print(f"  Running 4-agent pipeline with {model_tag}…\n")
    t0    = time.time()
    email = generate_email(*agents, scenario)
    elapsed = time.time() - t0

    print("\n" + "─" * 60)
    print(email)
    print("─" * 60)
    print(f"\n  Generated in {elapsed:.1f}s\n")


# ── Skip-generation mode ──────────────────────────────────────────────────────

def load_saved_emails() -> dict:
    """Load previously generated emails from the JSON cache."""
    from config import EMAILS_PATH
    path = Path(EMAILS_PATH)
    if not path.exists():
        print(f"[app] ERROR: {path} not found. Run without --skip-generation first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[app] Loaded saved emails from {path}  ({len(data)} models)")
    return data


# ── Full generation pipeline ──────────────────────────────────────────────────

def run_generation(models: dict) -> dict:
    """
    Generate emails for all scenarios with all models.

    Parameters
    ----------
    models : dict  { display_name: ollama_tag }

    Returns
    -------
    dict  { display_name: { scenario_id: email_str } }
    """
    from llm_factory import create_llm
    from agents import generate_all
    from scenarios import SCENARIOS

    all_emails = {}
    for display_name, ollama_tag in models.items():
        llm_id = create_llm(ollama_tag)
        emails = generate_all(display_name, llm_id, SCENARIOS)
        all_emails[display_name] = emails

    return all_emails


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print_banner()
    args = parse_args()

    # ── Step 0: Resolve models ─────────────────────────────────────────────
    from config import MODELS as DEFAULT_MODELS
    models = dict(DEFAULT_MODELS)
    if args.model_a:
        key_a        = list(models.keys())[0]
        models[key_a] = args.model_a
    if args.model_b:
        key_b        = list(models.keys())[1]
        models[key_b] = args.model_b

    model_tags = list(models.values())

    print("[app] Models configured:")
    for name, tag in models.items():
        print(f"       {name:30s}  ←  {tag}")
    print()

    # ── Step 1: Bootstrap environment ─────────────────────────────────────
    print("[app] STEP 1/6  Environment bootstrap")
    from install import bootstrap
    bootstrap([] if args.no_download else model_tags)

    # ── Step 2: Demo mode ──────────────────────────────────────────────────
    if args.demo:
        print("[app] STEP 2/6  Demo mode selected")
        demo_tag = model_tags[0]
        run_demo(demo_tag)
        print("[app] Demo complete. Exiting.")
        return

    # ── Step 3: Generate or load emails ───────────────────────────────────
    print("\n[app] STEP 2/6  Email generation")
    if args.skip_generation:
        print("[app] --skip-generation active: loading saved emails.")
        all_emails = load_saved_emails()
    else:
        all_emails = run_generation(models)

    # ── Step 4: Save emails ────────────────────────────────────────────────
    print("\n[app] STEP 3/6  Saving generated emails")
    from reporter import save_emails
    save_emails(all_emails)

    # ── Step 5: Evaluate ───────────────────────────────────────────────────
    print("\n[app] STEP 4/6  Running evaluation metrics (FCS · TCS · EQS)")
    from evaluator import evaluate_all
    df = evaluate_all(all_emails)

    # ── Step 6: Save reports ───────────────────────────────────────────────
    print("\n[app] STEP 5/6  Generating reports")
    from reporter import save_csv, save_json, save_analysis_md
    save_csv(df)
    report = save_json(df)
    save_analysis_md(report, df)

    # ── Step 7: Print summary ──────────────────────────────────────────────
    print("\n[app] STEP 6/6  Summary")
    from reporter import print_summary
    print_summary(df)

    # ── Colab download helper ──────────────────────────────────────────────
    from config import CSV_PATH, JSON_PATH, EMAILS_PATH, ANALYSIS_MD_PATH
    output_files = [CSV_PATH, JSON_PATH, EMAILS_PATH, ANALYSIS_MD_PATH]

    print("✓ Pipeline complete. Output files:")
    for p in output_files:
        print(f"    {p}")

    try:
        from google.colab import files as colab_files
        print("\n[app] Detected Colab environment — downloading result files…")
        for p in output_files:
            if Path(p).exists():
                colab_files.download(str(p))
        print("[app] Downloads initiated.")
    except ImportError:
        print(
            "\n[app] Not running in Colab — files are saved locally in ./results/"
        )


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()