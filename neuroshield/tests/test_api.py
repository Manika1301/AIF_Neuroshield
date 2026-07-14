import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from neuroshield.api.engine import RuntimeEngine
from neuroshield.api.main import app, get_engine, run
from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.features.personalize import add_personalized_features
from neuroshield.features.harmonize import harmonize_labels, pool_harmonized
from neuroshield.models.multihead import save_multihead_artifact, train_final_multihead
from neuroshield.runtime.quality_gate import MOTION_PAUSED, POOR_SIGNAL
from neuroshield.runtime.status import CALIBRATING, WAITING
from neuroshield.runtime.synthetic_source import generate_events, write_ndjson


def _wesad_like(n_subjects=4, n_per_class=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        for raw, shift in ((1, 0.0), (2, 3.0), (3, 1.5), (4, -1.0)):
            for _ in range(n_per_class):
                row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                row["subject_id"] = f"W{i}"
                row["label"] = raw
                row["valid_fraction"] = 1.0
                rows.append(row)
    return add_personalized_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("model")
    pooled, _ = harmonize_labels(_wesad_like(), "wesad")
    pooled = pool_harmonized([pooled])
    model = train_final_multihead(pooled, random_state=0)
    model_path = tmp_dir / "m2.joblib"
    manifest_path = tmp_dir / "m2_manifest.json"
    metrics_path = tmp_dir / "m2_metrics.json"
    save_multihead_artifact(
        model, pooled, metrics_path=metrics_path, model_path=model_path, manifest_path=manifest_path
    )
    return model_path, manifest_path


@pytest.fixture
def replay_fixture_path(tmp_path):
    phases = [
        ("quiet_baseline", 0.30),
        ("mild_stress_rise", 0.20),
        ("motion_burst", 0.15),
        ("recovery", 0.20),
        ("sensor_fault", 0.15),
    ]
    events = generate_events(duration_sec=300.0, seed=21, session_id="api-test", phases=phases)
    path = tmp_path / "api_fixture.ndjson"
    write_ndjson(events, path)
    return path


@pytest.fixture
def engine(trained_artifact):
    model_path, manifest_path = trained_artifact
    return RuntimeEngine(model_path=model_path, manifest_path=manifest_path)


@pytest.fixture
def client(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealth:
    def test_reports_model_loaded_and_no_session(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["baseline_loaded"] is False
        assert body["source_connected"] is False
        assert body["session_id"] is None

    def test_reports_missing_model_when_artifact_absent(self):
        bad_engine = RuntimeEngine(model_path=Path("/nonexistent/m1.joblib"), manifest_path=Path("/nonexistent/m.json"))
        app.dependency_overrides[get_engine] = lambda: bad_engine
        try:
            with TestClient(app) as c:
                resp = c.get("/api/v1/health")
                assert resp.json()["model_loaded"] is False
                assert resp.json()["model_error"] is not None
        finally:
            app.dependency_overrides.clear()


class TestSystem:
    def test_reports_versions_and_threshold_policy(self, client):
        resp = client.get("/api/v1/system")
        body = resp.json()
        assert body["schema_version"] == "neuroshield.hw.v1"
        assert body["feature_version"] == "features-v2"
        assert body["model_version"] == "m3_multihead_personalized_v1"
        assert "green_max" in body["threshold_policy"]
        assert body["uptime_s"] >= 0


class TestSessionStart:
    def test_replay_session_starts_successfully(self, client, replay_fixture_path):
        resp = client.post(
            "/api/v1/session/start",
            json={"source_mode": "replay", "replay_path": str(replay_fixture_path), "session_id": "sess-1"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"session_id": "sess-1", "source_mode": "replay"}

    def test_synthetic_session_starts_successfully(self, client):
        resp = client.post(
            "/api/v1/session/start", json={"source_mode": "synthetic", "duration_sec": 30.0, "seed": 1}
        )
        assert resp.status_code == 200
        assert resp.json()["source_mode"] == "synthetic"

    def test_serial_source_mode_is_a_typed_unsupported_error(self, client):
        resp = client.post("/api/v1/session/start", json={"source_mode": "serial"})
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "unsupported_source_mode"

    def test_unknown_source_mode_is_rejected(self, client):
        resp = client.post("/api/v1/session/start", json={"source_mode": "bluetooth"})
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "unsupported_source_mode"

    def test_replay_without_path_errors(self, client):
        resp = client.post("/api/v1/session/start", json={"source_mode": "replay"})
        assert resp.status_code == 400

    def test_replay_file_with_zero_valid_events_is_schema_mismatch(self, client, tmp_path):
        bad_path = tmp_path / "garbage.ndjson"
        bad_path.write_text("not json at all\nalso not json\n")
        resp = client.post(
            "/api/v1/session/start", json={"source_mode": "replay", "replay_path": str(bad_path)}
        )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "schema_mismatch"


class TestStatusBeforeCalibration:
    def test_waiting_before_any_session(self, client):
        resp = client.get("/api/v1/status/latest")
        assert resp.json()["state"] == WAITING

    def test_calibrating_after_session_before_calibration(self, client, replay_fixture_path):
        client.post(
            "/api/v1/session/start", json={"source_mode": "replay", "replay_path": str(replay_fixture_path)}
        )
        resp = client.get("/api/v1/status/latest")
        assert resp.json()["state"] == CALIBRATING


class TestCalibrationAndFullFlow:
    def test_full_replay_flow_produces_expected_states(self, client, replay_fixture_path):
        client.post(
            "/api/v1/session/start",
            json={"source_mode": "replay", "replay_path": str(replay_fixture_path), "session_id": "full-flow"},
        )
        calib = client.post("/api/v1/calibration/start", json={"quiet_seconds": 85.0})
        assert calib.status_code == 200
        assert calib.json()["n_accepted_windows"] > 0

        latest = client.get("/api/v1/status/latest")
        assert latest.json()["state"] not in (WAITING, CALIBRATING)

        history = client.get("/api/v1/history")
        records = history.json()["records"]
        assert len(records) > 0
        states = {r["state"] for r in records}
        assert MOTION_PAUSED in states
        assert POOR_SIGNAL in states
        # every record carries model/feature version and a timestamp
        for r in records:
            assert r["feature_version"] == "features-v2"
            assert r["timestamp"]

    def test_enriched_fields_present_on_scored_windows(self, client, replay_fixture_path):
        client.post(
            "/api/v1/session/start", json={"source_mode": "replay", "replay_path": str(replay_fixture_path)}
        )
        client.post("/api/v1/calibration/start", json={"quiet_seconds": 85.0})
        records = client.get("/api/v1/history").json()["records"]
        scored = [r for r in records if r["stress_index"] is not None]
        assert scored, "expected at least one scored window with a stress index"
        for r in scored:
            assert 0 <= r["stress_index"] <= 100
            assert r["level"] in ("calm", "elevated", "high")
            assert set(r["axes"].keys()) == {"cardiac", "electrodermal", "thermal", "movement"}

    def test_session_summary_endpoint(self, client, replay_fixture_path):
        client.post(
            "/api/v1/session/start", json={"source_mode": "replay", "replay_path": str(replay_fixture_path)}
        )
        client.post("/api/v1/calibration/start", json={"quiet_seconds": 85.0})
        summary = client.get("/api/v1/session/summary").json()
        for key in ("time_in_state", "recovery_trend", "episodes", "index_summary"):
            assert key in summary

    def test_insights_endpoint_returns_artifacts_shape(self, client):
        body = client.get("/api/v1/insights").json()
        assert "validation_scoreboard" in body
        assert "nurse_context_insights_markdown" in body
        assert "descriptive" in body["note"].lower()

    def test_calibration_without_session_is_session_not_started(self, client):
        resp = client.post("/api/v1/calibration/start", json={"quiet_seconds": 60.0})
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "session_not_started"

    def test_calibration_with_insufficient_quiet_data_is_missing_baseline(self, client):
        client.post("/api/v1/session/start", json={"source_mode": "synthetic", "duration_sec": 5.0, "seed": 2})
        resp = client.post("/api/v1/calibration/start", json={"quiet_seconds": 150.0})
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "missing_baseline"


class TestHistoryEndpoint:
    def test_empty_before_calibration(self, client, replay_fixture_path):
        client.post(
            "/api/v1/session/start", json={"source_mode": "replay", "replay_path": str(replay_fixture_path)}
        )
        resp = client.get("/api/v1/history")
        assert resp.json()["records"] == []

    def test_limit_returns_only_last_n(self, client, replay_fixture_path):
        client.post(
            "/api/v1/session/start", json={"source_mode": "replay", "replay_path": str(replay_fixture_path)}
        )
        client.post("/api/v1/calibration/start", json={"quiet_seconds": 85.0})
        full = client.get("/api/v1/history").json()["records"]
        limited = client.get("/api/v1/history", params={"limit": 3}).json()["records"]
        assert len(limited) == 3
        assert limited == full[-3:]


class TestWebSocket:
    def test_streams_full_history_after_calibration(self, client, replay_fixture_path):
        client.post(
            "/api/v1/session/start", json={"source_mode": "replay", "replay_path": str(replay_fixture_path)}
        )
        client.post("/api/v1/calibration/start", json={"quiet_seconds": 85.0})
        expected = client.get("/api/v1/history").json()["records"]

        received = []
        with client.websocket_connect("/ws/v1/live") as ws:
            for _ in range(len(expected)):
                msg = ws.receive_json()
                assert msg["type"] == "status"
                received.append(msg["data"])

        assert len(received) == len(expected)
        assert [r["state"] for r in received] == [r["state"] for r in expected]

    def test_sends_single_waiting_message_with_no_session(self, client):
        with client.websocket_connect("/ws/v1/live") as ws:
            msg = ws.receive_json()
            assert msg["data"]["state"] == WAITING


class TestErrorResponseShape:
    def test_engine_errors_have_error_code_and_message(self, client):
        resp = client.post("/api/v1/session/start", json={"source_mode": "serial"})
        body = resp.json()
        assert set(body.keys()) == {"error_code", "message"}
        assert isinstance(body["error_code"], str)
        assert isinstance(body["message"], str)


class TestMissingModelErrors:
    @pytest.fixture
    def broken_client(self):
        bad_engine = RuntimeEngine(model_path=Path("/nonexistent/m1.joblib"), manifest_path=Path("/nonexistent/m.json"))
        app.dependency_overrides[get_engine] = lambda: bad_engine
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()

    def test_status_latest_returns_typed_missing_model_error(self, broken_client):
        resp = broken_client.get("/api/v1/status/latest")
        assert resp.status_code == 503
        assert resp.json()["error_code"] == "missing_model"

    def test_calibration_prioritizes_missing_model_over_missing_session(self, broken_client):
        resp = broken_client.post("/api/v1/calibration/start", json={"quiet_seconds": 60.0})
        assert resp.status_code == 503
        assert resp.json()["error_code"] == "missing_model"


def test_run_binds_to_localhost_by_default():
    sig = inspect.signature(run)
    assert sig.parameters["host"].default == "127.0.0.1"
