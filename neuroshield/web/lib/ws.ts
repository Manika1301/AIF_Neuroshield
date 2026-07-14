"use client";

// Live feed. The backend pushes one message per 60s window as it is processed
// (see src/neuroshield/api/streaming.py), so the dashboard never polls for status.

import { useCallback, useEffect, useRef, useState } from "react";
import { StatusRecord, WS_URL } from "./api";

export type FeedMessage =
  | { type: "status"; data: StatusRecord }
  | { type: "session_complete"; data: { n_windows: number; complete: boolean } }
  | { type: "error"; data: { message: string } };

export type ConnectionState = "connecting" | "open" | "closed";

const RECONNECT_DELAY_MS = 2000;

export interface LiveFeed {
  records: StatusRecord[];
  latest: StatusRecord | null;
  connection: ConnectionState;
  complete: boolean;
  error: string | null;
  reset: () => void;
}

export function useLiveFeed(): LiveFeed {
  const [records, setRecords] = useState<StatusRecord[]>([]);
  const [snapshot, setSnapshot] = useState<StatusRecord | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [complete, setComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Survives re-renders and reconnects; set when the component unmounts so a scheduled
  // reconnect cannot resurrect a socket for a dead component.
  const disposedRef = useRef(false);

  /** Clear the feed when a new session starts, so windows from the old one don't linger. */
  const reset = useCallback(() => {
    setRecords([]);
    setSnapshot(null);
    setComplete(false);
    setError(null);
  }, []);

  useEffect(() => {
    disposedRef.current = false;

    const connect = () => {
      if (disposedRef.current) return;
      setConnection("connecting");

      let socket: WebSocket;
      try {
        socket = new WebSocket(WS_URL);
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = socket;

      socket.onopen = () => {
        if (!disposedRef.current) setConnection("open");
      };

      socket.onmessage = (event) => {
        if (disposedRef.current) return;
        let message: FeedMessage;
        try {
          message = JSON.parse(event.data);
        } catch {
          return; // a malformed frame is not worth tearing the feed down for
        }

        if (message.type === "status") {
          const record = message.data;
          if (record.window_start_s == null) {
            // The pre-session snapshot (waiting / calibrating). A live state, not a history row --
            // it has no window, so it must never enter the timeline or the charts.
            setSnapshot(record);
          } else {
            setRecords((prev) => {
              // The socket replays its backlog on (re)connect, so a window can arrive twice.
              // Window start time is the natural identity.
              if (prev.some((r) => r.window_start_s === record.window_start_s)) return prev;
              return [...prev, record];
            });
          }
        } else if (message.type === "session_complete") {
          setComplete(true);
        } else if (message.type === "error") {
          setError(message.data.message);
        }
      };

      socket.onerror = () => {
        if (!disposedRef.current) setConnection("closed");
      };

      socket.onclose = () => {
        if (disposedRef.current) return;
        setConnection("closed");
        scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (disposedRef.current || reconnectRef.current) return;
      reconnectRef.current = setTimeout(() => {
        reconnectRef.current = null;
        connect();
      }, RECONNECT_DELAY_MS);
    };

    connect();

    return () => {
      disposedRef.current = true;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      socketRef.current?.close();
    };
  }, []);

  return {
    records,
    // Once real windows exist they are the truth; before that, the snapshot is all we have.
    latest: records.length > 0 ? records[records.length - 1] : snapshot,
    connection,
    complete,
    error,
    reset,
  };
}
