"""HTTP client for the NeuroShield FastAPI backend, with response validation.

Every response is checked before the dashboard trusts it: missing fields, or a
``schema_version``/``feature_version`` the dashboard doesn't recognize, raise
``BackendValidationError`` instead of being silently rendered. The expected versions are
declared as literal constants here -- deliberately *not* imported from the backend package --
so this check catches real version skew between a running backend and a dashboard that wasn't
restarted, rather than trivially comparing a constant to itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 5.0

EXPECTED_SCHEMA_VERSION = "neuroshield.hw.v1"
EXPECTED_FEATURE_VERSION = "features-v2"


class BackendUnreachableError(Exception):
    """The backend could not be reached at all (connection refused, timeout, DNS, ...)."""


class BackendValidationError(Exception):
    """The backend responded, but the response is malformed, an error, or a version mismatch."""


@dataclass
class BackendClient:
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = DEFAULT_TIMEOUT_S
    session: requests.Session = field(default_factory=requests.Session)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            resp = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout_s, **kwargs)
        except requests.RequestException as exc:
            raise BackendUnreachableError(f"cannot reach backend at {self.base_url}: {exc}") from exc

        try:
            body = resp.json()
        except ValueError as exc:
            raise BackendValidationError(
                f"backend returned non-JSON response (status {resp.status_code})"
            ) from exc

        if resp.status_code >= 400:
            error_code = body.get("error_code", "unknown_error") if isinstance(body, dict) else "unknown_error"
            message = body.get("message", str(body)) if isinstance(body, dict) else str(body)
            raise BackendValidationError(f"[{error_code}] {message}")

        if not isinstance(body, dict):
            raise BackendValidationError(f"expected a JSON object from {path}, got {type(body).__name__}")

        return body

    def _require_fields(self, body: dict, fields: tuple[str, ...], context: str) -> None:
        missing = [f for f in fields if f not in body]
        if missing:
            raise BackendValidationError(f"{context} response missing field(s): {missing}")

    def health(self) -> dict:
        body = self._request("GET", "/api/v1/health")
        self._require_fields(
            body, ("status", "model_loaded", "baseline_loaded", "source_connected", "session_id"), "health"
        )
        return body

    def system(self) -> dict:
        body = self._request("GET", "/api/v1/system")
        self._require_fields(body, ("schema_version", "feature_version", "model_version", "threshold_policy"), "system")
        if body["schema_version"] != EXPECTED_SCHEMA_VERSION:
            raise BackendValidationError(
                f"backend schema_version {body['schema_version']!r} does not match the dashboard's "
                f"expected {EXPECTED_SCHEMA_VERSION!r} -- refusing to trust this backend"
            )
        if body["feature_version"] not in (None, EXPECTED_FEATURE_VERSION):
            raise BackendValidationError(
                f"backend feature_version {body['feature_version']!r} does not match the dashboard's "
                f"expected {EXPECTED_FEATURE_VERSION!r}"
            )
        return body

    def start_session(
        self,
        source_mode: str,
        session_id: str = "demo-001",
        replay_path: str | None = None,
        duration_sec: float = 600.0,
        seed: int = 0,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/session/start",
            json={
                "source_mode": source_mode,
                "session_id": session_id,
                "replay_path": replay_path,
                "duration_sec": duration_sec,
                "seed": seed,
            },
        )

    def start_calibration(self, quiet_seconds: float = 150.0) -> dict:
        return self._request("POST", "/api/v1/calibration/start", json={"quiet_seconds": quiet_seconds})

    def status_latest(self) -> dict:
        body = self._request("GET", "/api/v1/status/latest")
        self._require_fields(
            body,
            ("timestamp", "state", "probability", "model_version", "feature_version", "quality", "reasons"),
            "status",
        )
        if body["feature_version"] not in (None, EXPECTED_FEATURE_VERSION):
            raise BackendValidationError(f"status feature_version mismatch: {body['feature_version']!r}")
        return body

    def history(self, limit: int | None = None) -> list[dict]:
        params = {"limit": limit} if limit is not None else None
        body = self._request("GET", "/api/v1/history", params=params)
        self._require_fields(body, ("records",), "history")
        return body["records"]

    def session_summary(self) -> dict:
        body = self._request("GET", "/api/v1/session/summary")
        self._require_fields(body, ("time_in_state", "recovery_trend", "episodes", "index_summary"), "summary")
        return body

    def insights(self) -> dict:
        body = self._request("GET", "/api/v1/insights")
        self._require_fields(body, ("validation_scoreboard", "nurse_context_insights_markdown"), "insights")
        return body
