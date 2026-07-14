"""FastAPI backend serving live NeuroShield status to the dashboard.

Binds to 127.0.0.1 by default (see ``run()``) -- this is a local-only MVP server, never intended
to be exposed on a network interface. A single global session/engine is served via FastAPI's
dependency injection (``get_engine``); this is a local single-user dashboard, not a multi-tenant
service, and tests override ``get_engine`` to inject a test-configured engine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from neuroshield.api.engine import EngineError, MissingModelError, RuntimeEngine
from neuroshield.features.extract import FEATURE_VERSION
from neuroshield.runtime.replay_source import SCHEMA_VERSION

SCOREBOARD_PATH = Path("artifacts/metrics/validation_scoreboard.json")
NURSE_INSIGHTS_PATH = Path("artifacts/metrics/nurse_context_insights.md")

app = FastAPI(title="NeuroShield API")
_START_TIME = time.monotonic()
_default_engine: RuntimeEngine | None = None


def get_engine() -> RuntimeEngine:
    """Default engine provider. Overridden in tests via app.dependency_overrides[get_engine]."""
    global _default_engine
    if _default_engine is None:
        _default_engine = RuntimeEngine()
    return _default_engine


@app.exception_handler(EngineError)
async def engine_error_handler(request, exc: EngineError):  # noqa: ANN001, ARG001
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": str(exc)},
    )


class SessionStartRequest(BaseModel):
    source_mode: str
    session_id: str = "demo-001"
    replay_path: str | None = None
    duration_sec: float = 600.0
    seed: int = 0


class CalibrationStartRequest(BaseModel):
    quiet_seconds: float = 150.0


@app.get("/api/v1/health")
def get_health(engine: RuntimeEngine = Depends(get_engine)):
    return {
        "status": "ok",
        "model_loaded": engine.model_loaded,
        "model_error": engine.model_error,
        "baseline_loaded": engine.baseline_loaded,
        "source_connected": engine.source_connected,
        "session_id": engine.session_id,
    }


@app.get("/api/v1/system")
def get_system(engine: RuntimeEngine = Depends(get_engine)):
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": engine.manifest["model_version"] if engine.manifest else None,
        "threshold_policy": engine.manifest["threshold_policy"] if engine.manifest else None,
        "source_mode": engine.source_mode,
        "session_id": engine.session_id,
        "uptime_s": time.monotonic() - _START_TIME,
    }


@app.post("/api/v1/session/start")
def post_session_start(body: SessionStartRequest, engine: RuntimeEngine = Depends(get_engine)):
    engine.start_session(
        source_mode=body.source_mode,
        session_id=body.session_id,
        replay_path=Path(body.replay_path) if body.replay_path else None,
        duration_sec=body.duration_sec,
        seed=body.seed,
    )
    return {"session_id": engine.session_id, "source_mode": engine.source_mode}


@app.post("/api/v1/calibration/start")
def post_calibration_start(body: CalibrationStartRequest, engine: RuntimeEngine = Depends(get_engine)):
    baseline = engine.run_calibration(body.quiet_seconds)
    return {
        "n_accepted_windows": baseline["n_accepted_windows"],
        "accepted_seconds": baseline["accepted_seconds"],
        "feature_version": baseline["feature_version"],
    }


@app.get("/api/v1/status/latest")
def get_status_latest(engine: RuntimeEngine = Depends(get_engine)):
    if not engine.model_loaded:
        raise MissingModelError(engine.model_error or "no multi-head model artifact is loaded")
    return engine.latest_status().to_dict()


@app.get("/api/v1/history")
def get_history(limit: int | None = None, engine: RuntimeEngine = Depends(get_engine)):
    return {"records": [r.to_dict() for r in engine.history(limit=limit)]}


@app.get("/api/v1/session/summary")
def get_session_summary(engine: RuntimeEngine = Depends(get_engine)):
    """Tier-3 dynamics over the current session (time-in-state, recovery trend, episodes)."""
    return engine.session_summary()


@app.get("/api/v1/insights")
def get_insights():
    """Tier-4 offline artifacts: the 3-dataset validation scoreboard and nurse context analytics.

    These are descriptive research artifacts, not live predictions (see docs/no_clinical_claims.md).
    Returns whatever artifacts are present on disk; absent ones are reported as null.
    """
    scoreboard = json.loads(SCOREBOARD_PATH.read_text()) if SCOREBOARD_PATH.exists() else None
    nurse_insights = NURSE_INSIGHTS_PATH.read_text() if NURSE_INSIGHTS_PATH.exists() else None
    return {
        "validation_scoreboard": scoreboard,
        "nurse_context_insights_markdown": nurse_insights,
        "note": "Descriptive research artifacts only; not a live cause predictor.",
    }


@app.websocket("/ws/v1/live")
async def ws_live(websocket: WebSocket, engine: RuntimeEngine = Depends(get_engine)):
    await websocket.accept()
    try:
        if engine.status_history:
            for record in engine.status_history:
                await websocket.send_json({"type": "status", "data": record.to_dict()})
        else:
            await websocket.send_json({"type": "status", "data": engine.latest_status().to_dict()})
        await websocket.close()
    except WebSocketDisconnect:
        pass


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
