"""
evaluator.py
────────────
Runs all three custom metrics across all generated emails and
returns a structured pandas DataFrame.

Called by app.py after generation is complete.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from metrics import compute_fcs, compute_tcs, compute_eqs
from scenarios import SCENARIOS


def evaluate_one(
    generated_email: str,
    scenario: dict,
) -> dict:
    """
    Compute all three metrics for a single (model, scenario) pair.

    Parameters
    ----------
    generated_email : str
        The email produced by the generation pipeline.
    scenario : dict
        The corresponding entry from scenarios.SCENARIOS.

    Returns
    -------
    dict with keys: fcs, tcs, eqs, eqs_structure, eqs_rouge_l,
                    eqs_readability, composite
    """
    # Metric 1
    fcs, _per_fact = compute_fcs(generated_email, scenario["key_facts"])

    # Metric 2
    tcs = compute_tcs(generated_email, scenario["tone"])

    # Metric 3
    eqs, eqs_sub = compute_eqs(generated_email, scenario["reference_email"])

    composite = round((fcs + tcs + eqs) / 3, 4)

    return {
        "fcs":             fcs,
        "tcs":             tcs,
        "eqs":             eqs,
        "eqs_structure":   eqs_sub["structure"],
        "eqs_rouge_l":     eqs_sub["rouge_l"],
        "eqs_readability": eqs_sub["readability"],
        "composite":       composite,
    }


def evaluate_all(
    all_model_emails: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Evaluate all models across all scenarios.

    Parameters
    ----------
    all_model_emails : dict
        { model_display_name: { scenario_id: generated_email_str } }

    Returns
    -------
    pd.DataFrame
        One row per (model × scenario), with metric columns.
    """
    records: List[dict] = []

    for model_name, email_dict in all_model_emails.items():
        print(f"\n[eval] Scoring {model_name} …")

        for scenario in SCENARIOS:
            sid = scenario["id"]
            # JSON serialises int keys as strings — handle both
            email = email_dict.get(str(sid), email_dict.get(sid, ""))

            if not email or str(email).startswith("[ERROR]"):
                scores = {
                    "fcs": 0.0, "tcs": 0.0, "eqs": 0.0,
                    "eqs_structure": 0.0, "eqs_rouge_l": 0.0,
                    "eqs_readability": 0.0, "composite": 0.0,
                }
            else:
                scores = evaluate_one(email, scenario)

            records.append({
                "model":       model_name,
                "scenario_id": sid,
                "intent":      scenario["intent"],
                "tone":        scenario["tone"],
                "generated":   str(email)[:2000],   # truncate for CSV readability
                **scores,
            })

            print(
                f"  [{sid:02d}] FCS={scores['fcs']:.3f}  "
                f"TCS={scores['tcs']:.3f}  "
                f"EQS={scores['eqs']:.3f}  "
                f"→ composite={scores['composite']:.3f}"
            )

    return pd.DataFrame(records)


def model_summary(df: pd.DataFrame) -> dict:
    """
    Compute per-model aggregate statistics.

    Returns
    -------
    dict  { model_name: { metric: mean/std/min/max } }
    """
    summary = {}
    for model_name, grp in df.groupby("model"):
        summary[model_name] = {
            "FCS_mean":       round(grp["fcs"].mean(), 4),
            "FCS_std":        round(grp["fcs"].std(), 4),
            "TCS_mean":       round(grp["tcs"].mean(), 4),
            "TCS_std":        round(grp["tcs"].std(), 4),
            "EQS_mean":       round(grp["eqs"].mean(), 4),
            "EQS_std":        round(grp["eqs"].std(), 4),
            "composite_mean": round(grp["composite"].mean(), 4),
            "composite_std":  round(grp["composite"].std(), 4),
            # EQS sub-components
            "structure_mean": round(grp["eqs_structure"].mean(), 4),
            "rouge_l_mean":   round(grp["eqs_rouge_l"].mean(), 4),
            "readability_mean": round(grp["eqs_readability"].mean(), 4),
        }
    return summary