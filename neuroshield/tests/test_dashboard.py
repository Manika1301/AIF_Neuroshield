"""Headless execution tests for app/dashboard.py using Streamlit's AppTest.

AppTest genuinely executes the Streamlit script and captures its rendered element tree -- unlike
curling a running `streamlit run` server (which only proves the static shell serves, since the
script itself only runs once a browser opens a WebSocket session). This is the most real
verification available without a browser automation tool in this environment; it does not
replace an actual visual check, which is noted in T17's summary.
"""

import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


class TestDashboardWithoutBackend:
    def test_shows_disconnected_error_without_crashing(self, monkeypatch):
        monkeypatch.setenv("NEUROSHIELD_API_URL", "http://127.0.0.1:8799")  # nothing listens here
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=15)

        assert not at.exception  # the script itself must not raise -- errors must be handled
        assert len(at.error) >= 1
        assert "Disconnected" in at.error[0].value


class TestDashboardWithLiveBackend:
    @staticmethod
    @pytest.fixture(scope="class")
    def live_backend_url(tmp_path_factory):
        import numpy as np
        import pandas as pd

        from neuroshield.api.engine import RuntimeEngine
        from neuroshield.api.main import app, get_engine
        from neuroshield.features.extract import FEATURE_COLUMNS
        from neuroshield.features.harmonize import harmonize_labels, pool_harmonized
        from neuroshield.features.personalize import add_personalized_features
        from neuroshield.models.multihead import save_multihead_artifact, train_final_multihead

        # Train and save a throwaway multi-head artifact to an isolated tmp path -- deliberately not
        # the real artifacts/models/ default, which a concurrent real training run owns.
        rng = np.random.default_rng(0)
        rows = []
        for i in range(3):
            for raw, shift in ((1, 0.0), (2, 3.0), (3, 1.5), (4, -1.0)):
                for _ in range(15):
                    row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}
                    row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                    row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                    row["subject_id"] = f"W{i}"
                    row["label"] = raw
                    row["valid_fraction"] = 1.0
                    rows.append(row)
        pooled, _ = harmonize_labels(add_personalized_features(pd.DataFrame(rows)), "wesad")
        pooled = pool_harmonized([pooled])
        model = train_final_multihead(pooled, random_state=0)

        tmp_dir = tmp_path_factory.mktemp("dashboard_model")
        model_path = tmp_dir / "m2.joblib"
        manifest_path = tmp_dir / "m2_manifest.json"
        save_multihead_artifact(model, pooled, model_path=model_path, manifest_path=manifest_path)

        test_engine = RuntimeEngine(model_path=model_path, manifest_path=manifest_path)
        app.dependency_overrides[get_engine] = lambda: test_engine

        config = uvicorn.Config(app, host="127.0.0.1", port=8732, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        yield "http://127.0.0.1:8732"
        server.should_exit = True
        thread.join(timeout=5)
        app.dependency_overrides.clear()

    def test_shows_waiting_state_with_no_session_started(self, monkeypatch, live_backend_url):
        monkeypatch.setenv("NEUROSHIELD_API_URL", live_backend_url)
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=15)

        assert not at.exception
        assert len(at.error) == 0  # a reachable, healthy backend must not show an error banner
