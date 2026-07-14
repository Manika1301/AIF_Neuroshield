"""SessionPlayer: drives the engine one window at a time and fans each record out to subscribers.

This is what makes the feed live. The engine produces windows lazily (``RuntimeEngine.advance``);
the player pulls them on a timer and publishes each to every subscribed WebSocket, then appends to
``status_history`` so REST pollers (the Streamlit dashboard) observe the same session filling up
progressively. One producer, many consumers, one shared history -- so the two frontends can never
disagree about what happened.

Pacing: a 60s window stepped every 30s means real hardware would emit one record every 30 seconds.
Waiting that long to demo is absurd, so ``speed`` compresses it -- ``speed=10`` emits one window
every 3 seconds. ``speed=0`` disables pacing entirely (windows are produced as fast as the CPU
allows), which is what the tests use so they don't sleep.

Subscribing mid-session is safe: a new subscriber is handed the backlog already in
``status_history`` before it starts receiving live records, so a browser refresh does not lose the
earlier part of the session.
"""

from __future__ import annotations

import asyncio
import contextlib

from neuroshield.api.engine import RuntimeEngine
from neuroshield.features.extract import DEFAULT_STEP_SEC

MSG_STATUS = "status"
MSG_SESSION_COMPLETE = "session_complete"
MSG_ERROR = "error"

DEFAULT_SPEED = 10.0  # 10x real time: one 30s-step window every 3 seconds

# A subscriber that stops draining its queue must not be able to stall the producer or balloon
# memory; past this depth we drop the connection instead.
MAX_QUEUE_DEPTH = 256


class SessionPlayer:
    """Advances one engine session and broadcasts each window to subscribed clients."""

    def __init__(self, engine: RuntimeEngine, speed: float = DEFAULT_SPEED, step_sec: float = DEFAULT_STEP_SEC):
        self.engine = engine
        self.speed = speed
        self.step_sec = step_sec
        self._subscribers: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None
        self._done = asyncio.Event()

    @property
    def interval_s(self) -> float:
        """Wall-clock seconds between windows. 0 when speed<=0 (unbounded: no sleeping)."""
        if self.speed <= 0:
            return 0.0
        return self.step_sec / self.speed

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def subscribe(self) -> asyncio.Queue:
        """Register a consumer. It receives the backlog first, then every subsequent record."""
        queue: asyncio.Queue = asyncio.Queue()
        for record in self.engine.status_history:
            queue.put_nowait({"type": MSG_STATUS, "data": record.to_dict()})
        if self.engine.is_complete:
            queue.put_nowait({"type": MSG_SESSION_COMPLETE, "data": self.engine.progress()})
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _publish(self, message: dict) -> None:
        for queue in list(self._subscribers):
            if queue.qsize() >= MAX_QUEUE_DEPTH:
                self.unsubscribe(queue)  # a consumer this far behind is gone, not slow
                continue
            queue.put_nowait(message)

    def start(self) -> None:
        """Kick off the background player. Idempotent: a second call while running is a no-op."""
        if self.running:
            return
        self._done.clear()
        self._task = asyncio.create_task(self.run())

    async def run(self) -> None:
        try:
            while True:
                # advance() is CPU work (feature window + model predict), measured in milliseconds,
                # so running it on the event loop is fine and keeps the engine single-threaded.
                record = self.engine.advance()
                if record is None:
                    self._publish({"type": MSG_SESSION_COMPLETE, "data": self.engine.progress()})
                    return
                self._publish({"type": MSG_STATUS, "data": record.to_dict()})
                # sleep(0) still yields to the event loop, so an unpaced player (speed=0) cannot
                # starve the socket writers or the REST handlers while it drains.
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a player crash must reach the client, not vanish
            self._publish({"type": MSG_ERROR, "data": {"message": str(exc)}})
        finally:
            self._done.set()

    async def wait_until_complete(self, timeout: float | None = None) -> None:
        if self._task is None:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._done.wait(), timeout=timeout)

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._subscribers.clear()
