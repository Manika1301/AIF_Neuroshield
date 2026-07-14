import threading
import time
from unittest.mock import Mock

import pytest
import requests
import uvicorn

from neuroshield.client import (
    BackendClient,
    BackendUnreachableError,
    BackendValidationError,
    EXPECTED_FEATURE_VERSION,
    EXPECTED_SCHEMA_VERSION,
)


def _mock_response(status_code: int, body) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


@pytest.fixture
def client():
    c = BackendClient(base_url="http://fake")
    c.session.request = Mock()
    return c


HEALTH_OK = {
    "status": "ok",
    "model_loaded": True,
    "model_error": None,
    "baseline_loaded": False,
    "source_connected": False,
    "session_id": None,
}

SYSTEM_OK = {
    "schema_version": EXPECTED_SCHEMA_VERSION,
    "feature_version": EXPECTED_FEATURE_VERSION,
    "model_version": "m1_wesad_features_v1",
    "threshold_policy": {"green_max": 0.45, "amber_max": 0.70},
    "source_mode": None,
    "session_id": None,
    "uptime_s": 1.0,
}

STATUS_OK = {
    "timestamp": "2026-01-01T00:00:00+00:00",
    "state": "green",
    "probability": 0.1,
    "model_version": "m1_wesad_features_v1",
    "feature_version": EXPECTED_FEATURE_VERSION,
    "quality": {},
    "values": {},
    "reasons": [],
}


class TestUnreachable:
    def test_connection_error_raises_backend_unreachable(self, client):
        client.session.request.side_effect = requests.ConnectionError("refused")
        with pytest.raises(BackendUnreachableError):
            client.health()

    def test_timeout_raises_backend_unreachable(self, client):
        client.session.request.side_effect = requests.Timeout("timed out")
        with pytest.raises(BackendUnreachableError):
            client.health()


class TestHealth:
    def test_success_returns_body(self, client):
        client.session.request.return_value = _mock_response(200, HEALTH_OK)
        assert client.health() == HEALTH_OK

    def test_missing_field_raises_validation_error(self, client):
        incomplete = dict(HEALTH_OK)
        del incomplete["session_id"]
        client.session.request.return_value = _mock_response(200, incomplete)
        with pytest.raises(BackendValidationError, match="session_id"):
            client.health()


class TestSystem:
    def test_success_returns_body(self, client):
        client.session.request.return_value = _mock_response(200, SYSTEM_OK)
        assert client.system() == SYSTEM_OK

    def test_schema_version_mismatch_raises(self, client):
        bad = dict(SYSTEM_OK, schema_version="neuroshield.hw.v2")
        client.session.request.return_value = _mock_response(200, bad)
        with pytest.raises(BackendValidationError, match="schema_version"):
            client.system()

    def test_feature_version_mismatch_raises(self, client):
        bad = dict(SYSTEM_OK, feature_version="features-v0")
        client.session.request.return_value = _mock_response(200, bad)
        with pytest.raises(BackendValidationError, match="feature_version"):
            client.system()

    def test_null_feature_version_is_tolerated(self, client):
        # No model loaded yet -> backend may legitimately report feature_version=None.
        ok_null = dict(SYSTEM_OK, feature_version=None)
        client.session.request.return_value = _mock_response(200, ok_null)
        assert client.system()["feature_version"] is None


class TestErrorResponses:
    def test_4xx_with_error_code_becomes_validation_error(self, client):
        client.session.request.return_value = _mock_response(
            409, {"error_code": "missing_baseline", "message": "not enough quiet data"}
        )
        with pytest.raises(BackendValidationError, match=r"\[missing_baseline\]"):
            client.status_latest()

    def test_5xx_without_recognizable_body_still_raises(self, client):
        client.session.request.return_value = _mock_response(500, {"unexpected": "shape"})
        with pytest.raises(BackendValidationError, match="unknown_error"):
            client.status_latest()

    def test_non_json_response_raises_validation_error(self, client):
        resp = Mock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        client.session.request.return_value = resp
        with pytest.raises(BackendValidationError, match="non-JSON"):
            client.health()


class TestStatusLatest:
    def test_success(self, client):
        client.session.request.return_value = _mock_response(200, STATUS_OK)
        assert client.status_latest() == STATUS_OK

    def test_feature_version_mismatch_raises(self, client):
        bad = dict(STATUS_OK, feature_version="features-v0")
        client.session.request.return_value = _mock_response(200, bad)
        with pytest.raises(BackendValidationError, match="feature_version"):
            client.status_latest()


class TestHistory:
    def test_success_returns_records(self, client):
        client.session.request.return_value = _mock_response(200, {"records": [STATUS_OK]})
        assert client.history() == [STATUS_OK]

    def test_missing_records_key_raises(self, client):
        client.session.request.return_value = _mock_response(200, {"oops": []})
        with pytest.raises(BackendValidationError, match="records"):
            client.history()


class TestRequestShapes:
    def test_start_session_posts_expected_body(self, client):
        client.session.request.return_value = _mock_response(200, {"session_id": "s", "source_mode": "replay"})
        client.start_session("replay", session_id="s", replay_path="x.ndjson")
        _, kwargs = client.session.request.call_args
        assert kwargs["json"]["source_mode"] == "replay"
        assert kwargs["json"]["replay_path"] == "x.ndjson"

    def test_history_passes_limit_as_query_param(self, client):
        client.session.request.return_value = _mock_response(200, {"records": []})
        client.history(limit=5)
        _, kwargs = client.session.request.call_args
        assert kwargs["params"] == {"limit": 5}


class TestLiveBackendSmoke:
    """A real end-to-end check against an actual running uvicorn server (not mocked), since the
    mocked tests above only prove the client's own validation logic, not real wire compatibility."""

    @staticmethod
    @pytest.fixture(scope="class")
    def live_server():
        from neuroshield.api.main import app

        config = uvicorn.Config(app, host="127.0.0.1", port=8731, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        yield "http://127.0.0.1:8731"
        server.should_exit = True
        thread.join(timeout=5)

    def test_health_and_system_against_real_server(self, live_server):
        client = BackendClient(base_url=live_server)
        health = client.health()
        assert "model_loaded" in health
        system = client.system()
        assert system["schema_version"] == EXPECTED_SCHEMA_VERSION
