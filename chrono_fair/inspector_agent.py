"""
Week 3c — Inspector Agent (Claude-agent wrapper).

When the IntersectionalMonitor flags a cell, this module emits a structured
report that a downstream LLM-based agent (or a clinical governance committee)
can act on. The output schema is designed to be regulator-readable: it maps
each finding to the relevant FDA PCCP and EU AI Act clauses.

This file deliberately contains *no* LLM call inside its core path so that
CHRONO-Fair is fully reproducible. A separate `LLMNarrator` adapter can be
plugged in to convert the structured report into prose; we ship a
deterministic templated narrator that requires no external API.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import pandas as pd
import json
from datetime import datetime


@dataclass
class InspectionReport:
    cell: str
    alarm_time: str
    n_observed: int
    flip_rate: float
    baseline_rate: float
    e_value: float
    flip_hazard_ratio: float
    aleatoric_share: float
    epistemic_share: float
    cause: str
    suggested_action: str
    regulatory_mapping: Dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


_REG_MAP = {
    'epistemic': {
        'FDA PCCP': 'Section IV.B - Data Management Plan: trigger retrain '
                    'with augmented stratum sample.',
        'EU AI Act': 'Article 10(2)(f) - representativeness; Article 61 - '
                     'post-market monitoring corrective action.',
        'STANDING Together (NEJM AI 2025)': 'Recommendation 12 - re-survey '
                                              'underrepresented subgroups.',
    },
    'aleatoric': {
        'FDA PCCP': 'Section IV.C - Modification Protocol: pause and audit '
                    'label-generation pipeline.',
        'EU AI Act': 'Article 10(2)(g) - bias-aware data preparation; '
                     'Article 14 - human oversight escalation.',
        'STANDING Together (NEJM AI 2025)': 'Recommendation 7 - audit '
                                              'historical encoding bias.',
    },
    'mixed': {
        'FDA PCCP': 'Sections IV.B + IV.C combined.',
        'EU AI Act': 'Articles 10(2)(f)+(g); Article 61.',
        'STANDING Together (NEJM AI 2025)': 'Recommendations 7 + 12.',
    },
    'none': {},
}


def build_report(
    cell: str,
    cell_history_row: Dict[str, Any],
    decomposition_row: pd.Series,
    baseline_rate: float,
) -> InspectionReport:
    cause = recommend_action_label(decomposition_row)
    flip_rate = float(decomposition_row.get('flip_rate', 0.0))
    hazard_ratio = (
        flip_rate / baseline_rate if baseline_rate > 0 else float('nan')
    )
    return InspectionReport(
        cell=cell,
        alarm_time=str(datetime.utcnow().isoformat()),
        n_observed=int(cell_history_row.get('n', 0)),
        flip_rate=flip_rate,
        baseline_rate=float(baseline_rate),
        e_value=float(cell_history_row.get('E', 1.0)),
        flip_hazard_ratio=float(hazard_ratio),
        aleatoric_share=float(decomposition_row.get('aleatoric_share', 0)),
        epistemic_share=float(decomposition_row.get('epistemic_share', 0)),
        cause=cause,
        suggested_action=_action_for(cause, decomposition_row),
        regulatory_mapping=_REG_MAP.get(cause, {}),
    )


def recommend_action_label(row: pd.Series) -> str:
    """Same logic as decomposition.recommend_action but returns only the label."""
    flip_rate = float(row.get('flip_rate', 0.0))
    if flip_rate < 1e-3:
        return 'none'
    epis = float(row.get('epistemic_share', 0.0))
    alea = float(row.get('aleatoric_share', 0.0))
    if epis > 0.6:
        return 'epistemic'
    if alea > 0.6:
        return 'aleatoric'
    return 'mixed'


def _action_for(cause: str, row: pd.Series) -> str:
    n = int(row.get('n', 0))
    if cause == 'epistemic':
        return (f"Collect approximately {int(n * 1.5)} additional samples "
                f"from this stratum and trigger model retraining with "
                f"stratum-reweighted loss.")
    if cause == 'aleatoric':
        return ("Pause model influence on this stratum, audit the label "
                "and feature pipeline for systematic bias; model retraining "
                "alone will not resolve the gap.")
    if cause == 'mixed':
        return ("Joint mitigation: parallel data-pipeline audit and "
                "ensemble-stabilised retrain. Re-evaluate after 30 days.")
    return "No action required."


class LLMNarrator:
    """Deterministic templated narrator (no external LLM dependency)."""

    def narrate(self, report: InspectionReport) -> str:
        return (
            f"[CHRONO-Fair alert at {report.alarm_time}]\n"
            f"Cell '{report.cell}' (n={report.n_observed}) has crossed the "
            f"anytime-valid e-value threshold (E={report.e_value:.1f}). "
            f"Current flip-rate {report.flip_rate:.1%} versus baseline "
            f"{report.baseline_rate:.1%} (hazard ratio "
            f"{report.flip_hazard_ratio:.2f}).\n"
            f"Decomposition: aleatoric share {report.aleatoric_share:.0%}, "
            f"epistemic share {report.epistemic_share:.0%} -- root cause "
            f"is **{report.cause}**.\n"
            f"Recommended action: {report.suggested_action}\n"
            f"Regulatory mapping: "
            + "; ".join(f"{k}: {v}" for k, v in report.regulatory_mapping.items())
        )


__all__ = ['InspectionReport', 'build_report', 'LLMNarrator',
           'recommend_action_label']
