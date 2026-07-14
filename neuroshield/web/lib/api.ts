// Typed API client for the NeuroShield FastAPI backend.
// Mirrors app/backend_client.py: validates the schema version and surfaces
// disconnected / backend-error states instead of rendering stale data.

export const EXPECTED_SCHEMA_VERSION = "neuroshield.hw.v1";
export const EXPECTED_FEATURE_VERSION = "features-v2";

const BASE_URL =
  process.env.NEXT_PUBLIC_NEUROSHIELD_API_URL ?? "http://127.0.0.1:8000";

export const WS_URL = `${BASE_URL.replace(/^http/, "ws")}/ws/v1/live`;

export class BackendUnreachableError extends Error {}
export class BackendValidationError extends Error {}

export interface AxisScore {
  score: number | null;
  level: string;
  n_features: number;
}

export interface StatusRecord {
  timestamp: string;
  state: string;
  probability: number | null;
  model_version: string | null;
  feature_version: string | null;
  quality: Record<string, number | null>;
  values: Record<string, number | null>;
  reasons: string[];
  window_start_s: number | null;
  window_end_s: number | null;
  stress_index: number | null;
  level: string | null;
  affect_state: string | null;
  affect_confidence: number | null;
  axes: Record<string, AxisScore>;
}

export interface SystemInfo {
  schema_version: string;
  feature_version: string | null;
  model_version: string | null;
  threshold_policy: Record<string, number> | null;
  source_mode: string | null;
  session_id: string | null;
  uptime_s: number;
  /** Set when the backend's feature version differs from ours. See getSystem(). */
  versionWarning?: string;
}

export interface SessionProgress {
  session_id: string | null;
  n_windows: number;
  complete: boolean;
  calibrated: boolean;
  streaming: boolean;
}

export interface SessionSummary {
  n_windows: number;
  n_scored_windows: number;
  time_in_state: Record<string, number>;
  recovery_trend: string;
  hrv_proxy_recovery: number | null;
  episodes: Array<{
    start_s: number;
    end_s: number;
    n_windows: number;
    peak_index: number | null;
    peak_state: string;
  }>;
  index_summary: { mean: number | null; max: number | null; latest: number | null };
}

export interface Insights {
  validation_scoreboard: {
    per_dataset?: Record<
      string,
      { evaluation: string; balanced_accuracy: number; macro_f1: number; n_windows: number }
    >;
    note?: string;
    STALE?: string;
  } | null;
  nurse_context_insights_markdown: string | null;
  note: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch (err) {
    throw new BackendUnreachableError(`cannot reach backend at ${BASE_URL}: ${err}`);
  }
  let body: any;
  try {
    body = await resp.json();
  } catch {
    throw new BackendValidationError(`non-JSON response (status ${resp.status})`);
  }
  if (!resp.ok) {
    const code = body?.error_code ?? "unknown_error";
    const message = body?.message ?? JSON.stringify(body);
    throw new BackendValidationError(`[${code}] ${message}`);
  }
  return body as T;
}

export async function getHealth() {
  return request<{
    status: string;
    model_loaded: boolean;
    model_error: string | null;
    baseline_loaded: boolean;
    source_connected: boolean;
    session_id: string | null;
  }>("/api/v1/health");
}

export async function getSystem(): Promise<SystemInfo> {
  const body = await request<SystemInfo>("/api/v1/system");

  // A schema mismatch is fatal: the raw event contract itself differs, so nothing can be trusted.
  if (body.schema_version !== EXPECTED_SCHEMA_VERSION) {
    throw new BackendValidationError(
      `backend schema_version ${body.schema_version} != expected ${EXPECTED_SCHEMA_VERSION}`
    );
  }

  // A feature-version mismatch is NOT fatal. It used to throw, which meant one stale constant here
  // blanked the entire dashboard behind an error banner and hid a working backend. Degrade loudly
  // instead: render everything, and show the user exactly what drifted.
  if (body.feature_version && body.feature_version !== EXPECTED_FEATURE_VERSION) {
    body.versionWarning =
      `Backend feature_version is "${body.feature_version}" but this UI was built for ` +
      `"${EXPECTED_FEATURE_VERSION}". Values may be misaligned — rebuild the frontend.`;
  }
  return body;
}

export async function startSession(body: {
  source_mode: string;
  session_id?: string;
  replay_path?: string | null;
  duration_sec?: number;
  seed?: number;
  speed?: number;
}) {
  return request<{ session_id: string; source_mode: string; speed: number }>("/api/v1/session/start", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function startCalibration(quiet_seconds = 150) {
  return request<{
    n_accepted_windows: number;
    accepted_seconds: number;
    feature_version: string;
    streaming: boolean;
  }>("/api/v1/calibration/start", {
    method: "POST",
    body: JSON.stringify({ quiet_seconds }),
  });
}

export async function getStatusLatest() {
  return request<StatusRecord>("/api/v1/status/latest");
}

export async function getHistory(limit?: number) {
  const q = limit != null ? `?limit=${limit}` : "";
  const body = await request<{ records: StatusRecord[] }>(`/api/v1/history${q}`);
  return body.records;
}

export async function getSessionProgress() {
  return request<SessionProgress>("/api/v1/session/progress");
}

export async function getSessionSummary() {
  return request<SessionSummary>("/api/v1/session/summary");
}

export async function getInsights() {
  return request<Insights>("/api/v1/insights");
}
