import pandas as pd
import pytest

from neuroshield.data.nurse_stress_loader import DEFAULT_ROOT, STRESS_ZIP_NAME
from neuroshield.models.nurse_insights import (
    DESCRIPTIVE_DISCLAIMER,
    context_cooccurrence,
    render_insights_markdown,
    save_insights,
)


def _events():
    # 3 high-stress (level 2) events, 2 low-stress (level 0). "Patient in Crisis" only in high;
    # "Documentation" only in low.
    return pd.DataFrame(
        {
            "stress_level": [2, 2, 2, 0, 0],
            "Patient in Crisis": [1, 1, 0, 0, 0],
            "Increased Workload": [1, 0, 1, 1, 0],
            "Documentation": [0, 0, 0, 1, 1],
        }
    )


class TestContextCooccurrence:
    def test_ranks_by_high_stress_rate(self):
        table = context_cooccurrence(_events(), context_columns=["Patient in Crisis", "Increased Workload", "Documentation"])
        # Patient in Crisis: 2/3 high; Workload: 2/3 high; Documentation: 0/3 high -> last
        assert table.iloc[-1]["context"] == "Documentation"
        assert table.iloc[-1]["rate_high_stress"] == 0.0

    def test_counts_are_correct(self):
        table = context_cooccurrence(_events(), context_columns=["Patient in Crisis"]).set_index("context")
        row = table.loc["Patient in Crisis"]
        assert row["n_high_stress"] == 2
        assert row["n_low_stress"] == 0
        assert row["rate_high_stress"] == round(2 / 3, 3)

    def test_lift_infinite_when_only_in_high(self):
        table = context_cooccurrence(_events(), context_columns=["Patient in Crisis"]).set_index("context")
        assert table.loc["Patient in Crisis"]["lift_high_vs_low"] == "inf"

    def test_documentation_more_common_in_low_stress(self):
        table = context_cooccurrence(_events(), context_columns=["Documentation"]).set_index("context")
        assert table.loc["Documentation"]["rate_low_stress"] > table.loc["Documentation"]["rate_high_stress"]

    def test_handles_string_and_numeric_flags(self):
        events = pd.DataFrame(
            {"stress_level": [2, 2, 0], "COVID related": ["1", "Yes", "0"]}
        )
        table = context_cooccurrence(events, context_columns=["COVID related"]).set_index("context")
        assert table.loc["COVID related"]["n_high_stress"] == 2  # "1" and "Yes" both count


class TestRenderAndSave:
    def test_markdown_includes_disclaimer_and_rows(self):
        table = context_cooccurrence(_events(), context_columns=["Patient in Crisis", "Documentation"])
        md = render_insights_markdown(table, n_high=3, n_low=2)
        assert DESCRIPTIVE_DISCLAIMER in md
        assert "Patient in Crisis" in md
        assert "descriptive" in md.lower()

    def test_save_writes_file(self, tmp_path):
        table = context_cooccurrence(_events(), context_columns=["Patient in Crisis"])
        path = tmp_path / "insights.md"
        save_insights(table, n_high=3, n_low=2, path=path)
        assert path.exists()
        assert "Context" in path.read_text()


@pytest.mark.skipif(
    not (DEFAULT_ROOT / STRESS_ZIP_NAME).exists(), reason="Nurse Stress dataset not present"
)
def test_real_survey_produces_ranked_table():
    from neuroshield.data.nurse_stress_loader import load_survey_events

    events = load_survey_events()
    table = context_cooccurrence(events)
    assert len(table) > 0
    assert "rate_high_stress" in table.columns
    # ranked descending
    rates = table["rate_high_stress"].tolist()
    assert rates == sorted(rates, reverse=True)
