"""Tests for the Inspector Agent's Z-Inspection / ALTAI structural alignment.

These tests do not claim that CHRONO-Fair implements the full Z-Inspection
protocol of Zicari et al. (IEEE TTS, 2021). They check that the Inspector
emits ALTAI-aligned governance cards in the case-driven, regulator-mapped
style of that work.
"""
from __future__ import annotations
import pandas as pd

from chrono_fair.inspector_agent import (
    InspectionReport,
    build_report,
    _ALTAI_REQUIREMENTS,
    _ALTAI_MAP,
)


def _decomp_row(flip_rate: float, alea: float, epis: float) -> pd.Series:
    return pd.Series({
        'flip_rate': flip_rate,
        'aleatoric_share': alea,
        'epistemic_share': epis,
        'n': 200,
    })


def test_altai_requirements_complete():
    """ALTAI has exactly seven Trustworthy AI requirements."""
    assert len(_ALTAI_REQUIREMENTS) == 7
    assert set(_ALTAI_REQUIREMENTS.keys()) == {
        f'R{i}' for i in range(1, 8)}


def test_epistemic_card_includes_altai_mapping():
    """An epistemic alarm yields an ALTAI mapping touching R2/R5/R7."""
    row = _decomp_row(flip_rate=0.18, alea=0.20, epis=0.80)
    rep = build_report(
        cell='race=Black',
        cell_history_row={'n': 200, 'E': 25.0},
        decomposition_row=row,
        baseline_rate=0.05,
    )
    assert isinstance(rep, InspectionReport)
    assert rep.cause == 'epistemic'
    # ALTAI mapping is non-empty and references the expected requirements.
    keys = list(rep.altai_mapping.keys())
    assert any('R2' in k for k in keys)
    assert any('R5' in k for k in keys)
    assert any('R7' in k for k in keys)
    # Suggested action is propagated unchanged.
    assert rep.regulatory_mapping  # FDA / EU / STANDING mapping still present
    assert rep.altai_mapping is not rep.regulatory_mapping


def test_aleatoric_card_includes_pipeline_audit_requirements():
    """An aleatoric alarm references the data-pipeline requirements R1/R3/R5/R7."""
    row = _decomp_row(flip_rate=0.18, alea=0.80, epis=0.10)
    rep = build_report(
        cell='race=Black',
        cell_history_row={'n': 200, 'E': 25.0},
        decomposition_row=row,
        baseline_rate=0.05,
    )
    assert rep.cause == 'aleatoric'
    keys = list(rep.altai_mapping.keys())
    for req in ['R1', 'R3', 'R5', 'R7']:
        assert any(req in k for k in keys), (
            f'aleatoric card missing ALTAI {req}: {keys}')


def test_mixed_card_includes_joint_mitigation_requirements():
    row = _decomp_row(flip_rate=0.18, alea=0.40, epis=0.50)
    rep = build_report(
        cell='race=Black',
        cell_history_row={'n': 200, 'E': 25.0},
        decomposition_row=row,
        baseline_rate=0.05,
    )
    assert rep.cause == 'mixed'
    assert len(rep.altai_mapping) >= 4


def test_no_alarm_card_has_empty_altai_mapping():
    row = _decomp_row(flip_rate=0.0, alea=0.0, epis=0.0)
    rep = build_report(
        cell='race=Black',
        cell_history_row={'n': 200, 'E': 1.0},
        decomposition_row=row,
        baseline_rate=0.05,
    )
    assert rep.cause == 'none'
    assert rep.altai_mapping == {}


def test_report_serialises_with_altai_field():
    """The JSON serialisation must include the altai_mapping field."""
    row = _decomp_row(flip_rate=0.18, alea=0.10, epis=0.85)
    rep = build_report(
        cell='race=Black',
        cell_history_row={'n': 200, 'E': 25.0},
        decomposition_row=row,
        baseline_rate=0.05,
    )
    blob = rep.to_json()
    assert '"altai_mapping"' in blob
    assert '"regulatory_mapping"' in blob
