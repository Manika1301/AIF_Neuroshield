"""FastAPI backend serving live NeuroShield status to the dashboards.

Binds to 127.0.0.1 by default (see ``run()``) -- this is a local-only research server, never
intended to be exposed on a network interface, and it carries no authentication for exactly that
reason. A single global session/engine is served via FastAPI's dependency injection
(``get_engine``); this is a local single-user dashboard, not a multi-tenant service, and tests
override ``get_engine`` to inject a test-configured engine.

Two consumers, one session: the browser dashboard subscribes to ``/ws/v1/live`` and is *pushed* one
record per window by the ``SessionPlayer``; the Streamlit dashboard polls REST. Both read the same
``status_history``, which the player fills progressively, so they cannot disagree.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from neuroshield.api.engine import EngineError, MissingModelError, RuntimeEngine
from neuroshield.api.streaming import DEFAULT_SPEED, MSG_SESSION_COMPLETE, SessionPlayer
from neuroshield.features.extract import FEATURE_VERSION
from neuroshield.runtime.replay_source import SCHEMA_VERSION

# Resolve artifacts relative to the package root, not the process CWD. These were CWD-relative, so
# launching uvicorn from anywhere but the repo root made /insights silently return nulls.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = Path(os.environ.get("NEUROSHIELD_ARTIFACTS_DIR", _PACKAGE_ROOT / "artifacts"))
SCOREBOARD_PATH = ARTIFACTS_DIR / "metrics" / "validation_scoreboard.json"
NURSE_INSIGHTS_PATH = ARTIFACTS_DIR / "metrics" / "nurse_context_insights.md"

# The browser dashboard is served from a different origin (localhost:3000) than this API
# (127.0.0.1:8000), so without CORS every fetch from it is blocked before it ever reaches us.
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ORIGINS = [o.strip() for o in os.environ.get("NEUROSHIELD_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if o.strip()]

app = FastAPI(title="NeuroShield API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

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
    # Playback rate for the live feed. 10x -> one 30s-step window every 3 wall-clock seconds.
    # 0 disables pacing (windows are emitted as fast as they compute).
    speed: float = DEFAULT_SPEED


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
async def post_session_start(body: SessionStartRequest, engine: RuntimeEngine = Depends(get_engine)):
    if engine.player is not None:
        await engine.player.stop()  # a new session supersedes whatever was playing
        engine.player = None

    engine.start_session(
        source_mode=body.source_mode,
        session_id=body.session_id,
        replay_path=Path(body.replay_path) if body.replay_path else None,
        duration_sec=body.duration_sec,
        seed=body.seed,
    )
    engine.player = SessionPlayer(engine, speed=body.speed)
    return {"session_id": engine.session_id, "source_mode": engine.source_mode, "speed": body.speed}


@app.post("/api/v1/calibration/start")
async def post_calibration_start(body: CalibrationStartRequest, engine: RuntimeEngine = Depends(get_engine)):
    """Compute the personal baseline, then start streaming windows in the background.

    Returns as soon as the baseline exists -- it no longer blocks on processing the whole session.
    Windows arrive afterwards via ``/ws/v1/live`` (pushed) or ``/history`` (polled) as the player
    advances.

    Must be ``async``: ``SessionPlayer.start()`` calls ``asyncio.create_task``, which needs a running
    event loop. A sync endpoint would be run in FastAPI's threadpool, where there isn't one.
    """
    baseline = engine.run_calibration(body.quiet_seconds)
    if engine.player is not None:
        engine.player.start()
    return {
        "n_accepted_windows": baseline["n_accepted_windows"],
        "accepted_seconds": baseline["accepted_seconds"],
        "feature_version": baseline["feature_version"],
        "streaming": engine.player is not None,
    }


@app.get("/api/v1/session/progress")
def get_session_progress(engine: RuntimeEngine = Depends(get_engine)):
    """How far the live session has got. Lets a client tell 'still streaming' from 'finished'."""
    progress = engine.progress()
    progress["streaming"] = engine.player is not None and engine.player.running
    return progress


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
    """Live feed: one message per window as it is processed, until the session ends.

    This used to dump ``status_history`` and close immediately. It now subscribes to the running
    SessionPlayer, so the client receives the backlog (whatever has already streamed), then each new
    window as it lands, then ``session_complete`` -- and the socket stays open, so a client that
    connects before or during a session sees the whole thing.
    """
    await websocket.accept()
    player = engine.player

    # Always give a client something to render immediately. Without this, a dashboard that connects
    # after /session/start but before /calibration/start would subscribe to an empty queue and hang
    # on a blank screen until the first window finally streamed.
    if not engine.status_history:
        await websocket.send_json({"type": "status", "data": engine.latest_status().to_dict()})

    if player is None:
        # No session armed: hold the socket open so the client can wait rather than reconnect-poll.
        try:
            await websocket.receive_text()  # blocks until the client goes away
        except WebSocketDisconnect:
            pass
        return

    queue = player.subscribe()
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
            if message["type"] == MSG_SESSION_COMPLETE:
                break
        # Session finished, but hold the socket open: the client decides when it's done reading.
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            pass
    except WebSocketDisconnect:
        pass
    finally:
        player.unsubscribe(queue)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
