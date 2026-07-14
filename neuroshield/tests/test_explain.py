import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS, extract_features
from neuroshield.models.artifact import train_final_m1
from neuroshield.runtime.baseline import compute_baseline_from_events, zscore_features
from neuroshield.runtime.events_to_bundle import events_to_bundle
from neuroshield.runtime.explain import (
    FALLBACK_REASON,
    FORBIDDEN_TERMS,
    MOTION_PAUSED_REASON,
    POOR_SIGNAL_REASON,
    TEMPLATES,
    assert_no_clinical_claims,
    explain_abstention,
    explain_color_status,
    explain_status,
    extract_lr_coefficients,
    rank_reasons,
)
from neuroshield.runtime.quality_gate import (
    MOTION_PAUSED,
    POOR_SIGNAL,
    AbstentionResult,
    check_abstention,
)
from neuroshield.runtime.status import CALIBRATING, GREEN, RED, WAITING
from neuroshield.runtime.synthetic_source import generate_events


class TestRankReasons:
    def test_orders_by_abs_zscore_without_coefficients(self):
        z = {"eda_level": 0.2, "hr_mean_bpm": 3.5, "temp_mean_c": -0.1}
        reasons = rank_reasons(z)
        assert reasons[0].feature == "hr_mean_bpm"

    def test_dedupes_by_group(self):
        z = {"hr_mean_bpm": 4.0, "ibi_sd_ms": 3.9, "ibi_rmssd_ms": 3.8}  # all "pulse"
        reasons = rank_reasons(z)
        assert len(reasons) == 1
        assert reasons[0].group == "pulse"

    def test_at_most_max_reasons_and_one_per_group(self):
        z = {
            "hr_mean_bpm": 5.0,
            "ibi_sd_ms": 4.9,
            "eda_level": 4.5,
            "eda_slope": 4.4,
            "temp_mean_c": -4.0,
        }
        reasons = rank_reasons(z, max_reasons=3)
        assert len(reasons) == 3
        assert len({r.group for r in reasons}) == 3

    def test_direction_matches_zscore_sign(self):
        reasons = rank_reasons({"eda_level": 2.5})
        assert reasons[0].direction == "above"
        reasons = rank_reasons({"eda_level": -2.5})
        assert reasons[0].direction == "below"

    def test_nan_zscore_is_skipped(self):
        reasons = rank_reasons({"eda_level": float("nan"), "hr_mean_bpm": 2.0})
        assert len(reasons) == 1
        assert reasons[0].feature == "hr_mean_bpm"

    def test_quality_features_never_selected(self):
        z = {"ppg_quality": 10.0, "valid_fraction": -10.0, "hr_mean_bpm": 0.1}
        reasons = rank_reasons(z)
        assert all(r.feature not in ("ppg_quality", "valid_fraction") for r in reasons)

    def test_signed_contribution_reorders_vs_plain_zscore(self):
        z = {"hr_mean_bpm": 5.0, "eda_level": 1.0}
        # Without coefficients, hr_mean_bpm (larger |z|) ranks first.
        plain = rank_reasons(z)
        assert plain[0].feature == "hr_mean_bpm"
        # With a coefficient set that makes eda_level's contribution dominate, it ranks first.
        coefficients = {"hr_mean_bpm": 0.01, "eda_level": 10.0}
        weighted = rank_reasons(z, coefficients=coefficients)
        assert weighted[0].feature == "eda_level"


class TestExtractLrCoefficients:
    def test_returns_one_coefficient_per_feature_column(self):
        rng = np.random.default_rng(0)
        rows = []
        for label in (0, 1):
            for _ in range(30):
                row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}
                row["m1_label"] = label
                rows.append(row)
        df = pd.DataFrame(rows)
        pipeline = train_final_m1(df)
        coefficients = extract_lr_coefficients(pipeline, FEATURE_COLUMNS)
        assert set(coefficients.keys()) == set(FEATURE_COLUMNS)
        assert all(isinstance(v, float) for v in coefficients.values())


class TestExplainColorStatus:
    def test_returns_between_one_and_three_reasons(self):
        z = {"hr_mean_bpm": 3.0, "eda_level": 2.0, "temp_mean_c": -1.5}
        reasons = explain_color_status(z)
        assert 1 <= len(reasons) <= 3

    def test_falls_back_when_no_candidate_survives(self):
        z = {"hr_mean_bpm": float("nan"), "ppg_quality": 5.0}  # only non-explainable/NaN features
        reasons = explain_color_status(z)
        assert reasons == [FALLBACK_REASON]

    def test_every_reason_maps_to_a_real_feature_column(self):
        z = {col: 2.0 for col in FEATURE_COLUMNS}
        ranked = rank_reasons(z)
        for r in ranked:
            assert r.feature in FEATURE_COLUMNS


class TestExplainAbstention:
    def test_motion_paused_uses_fixed_template(self):
        abst = AbstentionResult(abstain=True, reason=MOTION_PAUSED, triggers=["motion_dynamic_rms=5>1"])
        assert explain_abstention(abst) == [MOTION_PAUSED_REASON]

    def test_poor_signal_uses_fixed_template(self):
        abst = AbstentionResult(abstain=True, reason=POOR_SIGNAL, triggers=["ppg_quality=0.1<0.7"])
        assert explain_abstention(abst) == [POOR_SIGNAL_REASON]


class TestExplainStatusDispatch:
    def test_color_state_requires_and_uses_zscores(self):
        reasons = explain_status(GREEN, z_scores={"hr_mean_bpm": 0.1, "eda_level": 0.1})
        assert len(reasons) >= 1

    def test_color_state_without_zscores_raises(self):
        with pytest.raises(ValueError):
            explain_status(RED)

    def test_motion_paused_requires_abstention(self):
        with pytest.raises(ValueError):
            explain_status(MOTION_PAUSED)

    def test_motion_paused_with_abstention_returns_fixed_text(self):
        abst = AbstentionResult(abstain=True, reason=MOTION_PAUSED, triggers=[])
        assert explain_status(MOTION_PAUSED, abstention=abst) == [MOTION_PAUSED_REASON]

    def test_non_physiological_states_return_empty(self):
        assert explain_status(WAITING) == []
        assert explain_status(CALIBRATING) == []


class TestNoClinicalClaims:
    def test_assert_raises_on_forbidden_term(self):
        with pytest.raises(ValueError):
            assert_no_clinical_claims("This looks like a panic attack starting.")

    def test_assert_passes_on_safe_text(self):
        assert_no_clinical_claims("Skin-response activity is above your quiet baseline.")

    def test_all_templates_are_clean(self):
        for group in TEMPLATES.values():
            for text in group.values():
                assert_no_clinical_claims(text)

    def test_fixed_reasons_are_clean(self):
        assert_no_clinical_claims(MOTION_PAUSED_REASON)
        assert_no_clinical_claims(POOR_SIGNAL_REASON)
        assert_no_clinical_claims(FALLBACK_REASON)

    def test_forbidden_terms_list_is_non_trivial(self):
        assert "panic" in FORBIDDEN_TERMS
        assert "diagnos" in FORBIDDEN_TERMS


class TestFullReplayExplanations:
    """Integration: every colored window in a real replay gets 1-3 grounded reasons, no exceptions."""

    def test_every_colored_window_has_valid_reasons(self):
        events = generate_events(duration_sec=300.0, seed=13, phases=[("quiet_baseline", 0.5), ("mild_stress_rise", 0.5)])
        quiet_events = [e for e in events if e["t_us"] < 100_000_000]
        baseline = compute_baseline_from_events(quiet_events, source="synthetic", subject_id="t15")

        bundle = events_to_bundle(events)
        features = extract_features(bundle)
        z = zscore_features(features, baseline)

        rng = np.random.default_rng(0)
        rows = []
        for label, shift in ((0, 0.0), (1, 3.0)):
            for _ in range(40):
                row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                row["m1_label"] = label
                rows.append(row)
        pipeline = train_final_m1(pd.DataFrame(rows))
        coefficients = extract_lr_coefficients(pipeline, FEATURE_COLUMNS)

        checked_any_color = False
        for _, row in z.iterrows():
            abst = check_abstention(row)
            if abst.abstain:
                reasons = explain_status(abst.reason, abstention=abst)
                assert reasons == [MOTION_PAUSED_REASON] or reasons == [POOR_SIGNAL_REASON]
                continue

            z_scores = {col: row[f"{col}_z"] for col in FEATURE_COLUMNS}
            reasons = explain_color_status(z_scores, coefficients)
            assert 1 <= len(reasons) <= 3
            for text in reasons:
                assert_no_clinical_claims(text)
            checked_any_color = True

        assert checked_any_color
