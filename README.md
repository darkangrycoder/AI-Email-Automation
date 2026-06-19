# Email Generation Assistant

An agentic pipeline that generates context-aware, send-ready professional emails using a 4-agent CrewAI architecture, with a custom evaluation suite for comparing model quality head-to-head.

Built as an AI Engineer screening assessment, fully self-contained and runnable end-to-end on a free Colab T4 instance — no API keys required.

```
python app.py
```

## How it works

A single LLM call tends to produce generic, structurally inconsistent emails. This pipeline splits the work across four specialised agents, each with its own role, goal, and prompt strategy, run sequentially with CrewAI:

```
PlannerAgent  ──►  WriterAgent  ──►  CriticAgent  ──►  RefinerAgent
    │                  │                  │                  │
Blueprint          Draft email       Structured          Final email
(5-step CoT)        (R-FSCoT)         critique          (send-ready)
```

- **Planner** thinks structurally — produces a blueprint (subject line options, paragraph plan, vocabulary to use/avoid, risk notes) via explicit 5-step chain-of-thought.
- **Writer** thinks fluently — drafts the full email from the blueprint using role-playing + few-shot priming.
- **Critic** thinks analytically — scores the draft against a four-point rubric (factual completeness, tone precision, structural integrity, reader experience) and flags concrete, quote-level fixes.
- **Refiner** integrates — receives both the draft and the critique together and produces the final, send-ready email.

This catch-and-fix loop mirrors a professional editorial workflow, and CrewAI's task context-passing means the Refiner always sees the original draft *and* the critique side by side.

### Prompting strategy — R-FSCoT

Each agent is prompted with a composite of three techniques, tuned for 7B-class open models:

1. **Role-Playing Priming** — agents are given rich, authoritative personas (e.g. a 20-year veteran executive email writer) rather than generic instructions.
2. **Few-Shot In-Context Learning** — three worked examples spanning formal/operational, empathetic/customer-service, and persuasive/business-dev contexts.
3. **Structured Chain-of-Thought** — the Planner's task is decomposed into named STEP 1–5 reasoning before any output, and the Writer/Refiner include explicit self-check instructions.

### Models

Both models are served **locally via Ollama** (GGUF, quantized) — this removes HuggingFace Inference API rate limits entirely and runs comfortably on a free Colab T4 (15 GB VRAM) or any CPU-only machine.

| | Model | Tag | Size |
|---|---|---|---|
| Model A | Qwen2.5-7B-Instruct | `qwen2.5:7b` | ~4.4 GB |
| Model B | Mistral-7B-Instruct-v0.3 | `mistral:7b` | ~4.1 GB |

CrewAI routes agent calls through LiteLLM, which forwards them to Ollama's OpenAI-compatible local API (`localhost:11434/v1`) — no API key needed for either model.

## Evaluation

Each generated email is scored against the original scenario on three custom metrics, computed automatically after generation:

| Metric | What it measures | Method |
|---|---|---|
| **FCS** — Fact Coverage Score | Fraction of key facts semantically present in the email | Sentence-transformer embeddings (`all-MiniLM-L6-v2`), cosine similarity ≥ 0.42 per fact |
| **TCS** — Tone Congruence Score | Lexical alignment with the requested tone(s) | Curated positive/negative vocabulary banks, scored across the **union** of all compound tone descriptors (e.g. "Warm, enthusiastic, professional") |
| **EQS** — Email Quality Score | Structural completeness + similarity to a hand-written reference + readability | Weighted composite: structure regex (0.40) + ROUGE-L F1 vs. reference (0.40) + normalised Flesch readability (0.20) |

A `composite` score (mean of FCS, TCS, EQS) is reported per scenario and aggregated per model.

Evaluation runs across **10 hand-crafted scenarios** spanning formal, empathetic, urgent, persuasive, warm, and neutral tones, each with 3–5 facts that must appear in the output and a gold-standard reference email.

### Known bug fixes baked into the metrics

- **Tone parsing**: an earlier version only read the first comma-separated tone token, silently dropping the rest (so "Warm, enthusiastic, professional" was scored only against "warm" markers). Fixed to score against the union of all resolved tones.
- **Thinking-tag leakage**: Qwen2.5 emits `<Thinking>...</Thinking>` (capital T), which a case-sensitive regex missed, leaking planning text into the final email. Tag stripping is now case-insensitive and covers `<think>`, `<reasoning>`, and `<reflection>` variants too.

## Results

Latest run — Qwen2.5-7B vs Mistral-7B, 10 scenarios:

| Metric | Qwen2.5-7B | Mistral-7B | Δ (A − B) |
|---|---|---|---|
| FCS | 0.960 ± 0.084 | 0.900 ± 0.141 | +0.060 |
| TCS | 0.669 ± 0.323 | 0.632 ± 0.249 | +0.037 |
| EQS | 0.645 ± 0.085 | 0.642 ± 0.056 | +0.004 |
| **Composite** | **0.758** ± 0.117 | **0.725** ± 0.116 | **+0.033** |

Qwen2.5-7B led on all three metrics, with the gap driven mainly by fact coverage. Full analysis — including failure-mode breakdown and production recommendation — is auto-generated at `results/comparative_analysis.md` after each run.

## Getting started

### Run on Colab (recommended)

Upload the project files to a Colab notebook and run:

```bash
!python app.py
```

The first run automatically:
1. Installs Python dependencies from `requirements.txt`
2. Downloads NLTK tokenizer data
3. Installs and starts Ollama
4. Pulls both models (~8.5 GB total, one-time)
5. Generates emails for all 10 scenarios × 2 models
6. Runs all three evaluation metrics
7. Saves CSV/JSON/Markdown reports and downloads them automatically

### Run locally

```bash
git clone https://github.com/darkangrycoder/AI-Email-Automation.git
cd email-generation-assistant
python app.py
```

Requires Ollama-compatible hardware (works on CPU, faster with GPU). Same bootstrap sequence runs automatically.

### CLI options

```bash
python app.py                                          # full pipeline: generate + evaluate
python app.py --skip-generation                        # re-evaluate from previously saved emails
python app.py --demo                                   # interactive single-email demo, no evaluation
python app.py --model-a qwen2.5:14b --model-b llama3:8b # override either model
python app.py --no-download                             # skip Ollama pull, assume models already cached
```

### Demo mode

```bash
python app.py --demo
```

Prompts for an intent, comma-separated key facts, and a tone, then runs the full 4-agent pipeline once and prints the final email.

## Project structure

```
.
├── app.py                  # entry point — orchestrates the full pipeline
├── agents.py                # CrewAI agent + task definitions, pipeline execution
├── prompts.py                # R-FSCoT prompt templates for all four agents
├── scenarios.py               # 10 evaluation scenarios with key facts + reference emails
├── llm_factory.py              # Ollama/LiteLLM model routing
├── metrics.py                # FCS, TCS, EQS metric implementations
├── evaluator.py               # runs metrics across all (model × scenario) pairs
├── reporter.py                # serialises results to CSV/JSON/Markdown
├── install.py                 # environment bootstrap (pip, NLTK, Ollama, model pulls)
├── config.py                  # central configuration — models, paths, thresholds
├── requirements.txt
└── results/                   # generated on first run
    ├── generated_emails.json
    ├── evaluation_scores.csv
    ├── evaluation_report.json
    └── comparative_analysis.md
```

## Configuration

All tunable values live in `config.py`:

- `MODELS` — display name → Ollama tag mapping
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TOP_P` — generation parameters
- `FCS_SIMILARITY_THRESHOLD` — cosine similarity cutoff for fact coverage (default 0.42)
- `EQS_STRUCTURE_WEIGHT` / `EQS_ROUGE_WEIGHT` / `EQS_READABILITY_WEIGHT` — EQS sub-component weights
- `MAX_RETRIES`, `RETRY_BASE_DELAY` — retry behavior for failed generations

Optional [LangSmith](https://www.langchain.com/langsmith) tracing can be enabled by setting the `LANGCHAIN_API_KEY` environment variable before running.

## Limitations

- **Readability is nearly non-discriminative**: in current results it averages ~0.97 with minimal spread across both models — a future iteration should swap it for grammatical error rate or perplexity.
- **Evaluation scale**: 10 scenarios across 6 tone categories is enough to surface directional differences, not statistical significance. A production evaluation would need 50+ scenarios with blind human raters.
- **Latency**: the 4-agent pipeline makes 4+ LLM calls per email; on a Colab T4 this is ~15–45 seconds per email. For latency-sensitive production use, a dedicated GPU or a serving stack like vLLM/TGI would help.

## Requirements

See `requirements.txt`. Core dependencies:

- `crewai` / `crewai-tools` — agent orchestration
- `litellm` — Ollama routing
- `sentence-transformers`, `scikit-learn` — FCS embeddings
- `rouge-score`, `textstat`, `nltk` — EQS sub-metrics
- `pandas`, `numpy` — reporting

Ollama itself is installed automatically by `install.py` and does not need to be pre-installed.
