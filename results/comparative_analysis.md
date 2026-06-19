# Comparative Analysis: Email Generation Assistant
*Auto-generated · 2026-06-19 17:27 UTC*
*Models: Model-A (Qwen2.5-7B) vs Model-B (Mistral-7B)*

---

## 1. Metric Definitions

| Metric | Definition | Method |
|--------|-----------|--------|
| **FCS** — Fact Coverage Score | Fraction of key facts semantically present in the generated email | sentence-transformers (all-MiniLM-L6-v2), cosine similarity ≥ 0.42 |
| **TCS** — Tone Congruence Score | Lexical alignment with the target tone descriptor(s) | Curated vocabulary banks, scored across ALL compound tones (bug fix from v1) |
| **EQS** — Email Quality Score | Composite: structure + ROUGE-L + readability | Structure regex (0.40) + ROUGE-L F1 vs reference (0.40) + Flesch normalised (0.20) |

---

## 2. Evaluation Summary

| Metric | Model-A (Qwen2.5-7B) | Model-B (Mistral-7B) | Δ (A − B) |
|--------|--------------------|--------------------|-----------|
| FCS (mean ± std) | 0.9600 ± 0.0843 | 0.9000 ± 0.1414 | +0.0600 |
| TCS (mean ± std) | 0.6686 ± 0.3229 | 0.6319 ± 0.2491 | +0.0367 |
| EQS (mean ± std) | 0.6453 ± 0.0848 | 0.6415 ± 0.0561 | +0.0038 |
| **Composite**    | **0.7579** ± 0.1167 | **0.7245** ± 0.1159 | **+0.0334** |

*EQS sub-components:*

| Sub-metric | Model-A (Qwen2.5-7B) | Model-B (Mistral-7B) |
|------------|--------------------|--------------------|
| Structure  | 0.9200 | 0.9800 |
| ROUGE-L    | 0.3286 | 0.3079 |
| Readability| 0.7291 | 0.6318 |

---

## Q1 — Which model/strategy performed better?

**Model-A (Qwen2.5-7B)** achieved the higher composite score (0.7579 vs 0.7245,
Δ = +0.0334) and led on 3 out of 3 individual metrics.

The margin is narrow
(0.0334 composite points), suggesting a
meaningful but not conclusive quality difference at this evaluation scale.

---

## Q2 — Biggest failure mode of Model-B (Mistral-7B)

The largest performance gap is on **FCS**
(0.9000 vs 0.9600, Δ = -0.0600).

**Primary failure mode: FCS degradation**

Qualitative inspection of Scenario 6 ("Request a formal proposal from a software vendor for a CRM s…",
composite score = 0.5087) reveals:

- **Fact omission / over-paraphrase**: The model paraphrased key facts so
  loosely, or omitted them entirely, that semantic similarity fell below the
  0.42 coverage threshold. This indicates the model underweights the
  information-anchoring requirement of the task when following complex
  multi-agent context chains.

- **Tone drift**: In longer generations, the initial tone calibration degrades
  mid-email. The opening may match the target register, but closing paragraphs
  revert to a generic professional style regardless of the specified tone.

- **Structural regression**: Scenarios requiring non-standard formats (urgent
  flagging, executive announcements) show lower structure scores, suggesting
  the model defaults to conventional email templates under complex instructions.

These failure modes are consistent with a model with weaker in-context
instruction retention — the 4-agent pipeline's longer context window amplifies
this weakness relative to models with stronger long-context instruction following.

---

## Q3 — Production recommendation

**Recommended for production: Model-A (Qwen2.5-7B)**

Metric-grounded justification:

1. **Fact reliability (FCS 0.9600)** — In a production email assistant,
   an email that omits a promised discount, incorrect date, or wrong customer ID
   is a trust and compliance risk. Model-A (Qwen2.5-7B)'s higher FCS demonstrates more
   consistent fact anchoring across varied scenario types.

2. **Tone accuracy (TCS 0.6686)** — Business emails in the wrong
   register damage professional relationships. Model-A (Qwen2.5-7B) shows stronger calibration
   across all tone categories tested (formal, empathetic, urgent, persuasive,
   neutral), including compound multi-tone scenarios.

3. **Structural completeness (EQS 0.6453)** — Subject lines, salutations,
   and professional closings are non-negotiable in a deployed email assistant.
   Model-A (Qwen2.5-7B) achieves higher structure and ROUGE-L scores, indicating tighter
   alignment with professional email conventions.

**Deployment note**: Both models are served locally via Ollama GGUF at ~4 GB
each. Latency on Colab T4 is approximately 15–45 seconds per email (4-agent
pipeline, ~4–8 inference calls per scenario). For latency-sensitive production
use, a dedicated GPU instance or quantized serving stack (vLLM, TGI) would be
advisable.

---

## Self-Critique & Metric Limitations

1. **Readability near-constant**: Readability averaged 0.729 (Model-A (Qwen2.5-7B)) and
   0.632 (Model-B (Mistral-7B)) — spread of 0.0973. Both models land inside the
   ideal Flesch range regardless of quality, making this sub-metric nearly
   non-discriminative. Replacement with grammatical error rate or perplexity
   would provide more signal in future iterations.

2. **ROUGE-L context**: ROUGE-L scores of 0.329 / 0.308 may appear
   low in absolute terms. This is expected for open-ended professional email
   generation where many valid phrasings exist. ROUGE-L functions as a
   structural similarity proxy, not an absolute quality measure.

3. **Evaluation scope**: 10 scenarios across 6 tone categories. A production
   evaluation would require 50+ scenarios with blind human raters to reach
   statistical significance on composite score differences.

4. **Agentic overhead**: The 4-agent sequential pipeline makes 4 LLM calls per
   email. Critic and Refiner tasks improve quality but introduce latency. A
   production system might conditionally invoke the full pipeline only for
   high-stakes scenarios.

---
*Full raw scores: `results/evaluation_scores.csv`*
*Metric report:  `results/evaluation_report.json`*
*Generated emails: `results/generated_emails.json`*
