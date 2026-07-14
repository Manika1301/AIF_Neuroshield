// Typed API client for the NeuroShield FastAPI backend (D8).
// Mirrors app/backend_client.py: validates the schema version and surfaces
// disconnected / backend-error states instead of rendering stale data.

export const EXPECTED_SCHEMA_VERSION = "neuroshield.hw.v1";
export const EXPECTED_FEATURE_VERSION = "features-v1";

const BASE_URL =
  process.env.NEXT_PUBLIC_NEUROSHIELD_API_URL ?? "http://127.0.0.1:8000";

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
  axes: Record<string, AxisScore>;
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

export async function getSystem() {
  const body = await request<{
    schema_version: string;
    feature_version: string | null;
    model_version: string | null;
    threshold_policy: Record<string, number> | null;
  }>("/api/v1/system");
  if (body.schema_version !== EXPECTED_SCHEMA_VERSION) {
    throw new BackendValidationError(
      `backend schema_version ${body.schema_version} != expected ${EXPECTED_SCHEMA_VERSION}`
    );
  }
  if (body.feature_version && body.feature_version !== EXPECTED_FEATURE_VERSION) {
    throw new BackendValidationError(
      `backend feature_version ${body.feature_version} != expected ${EXPECTED_FEATURE_VERSION}`
    );
  }
  return body;
}

export async function startSession(body: {
  source_mode: string;
  session_id?: string;
  replay_path?: string | null;
  duration_sec?: number;
  seed?: number;
}) {
  return request("/api/v1/session/start", { method: "POST", body: JSON.stringify(body) });
}

export async function startCalibration(quiet_seconds = 150) {
  return request("/api/v1/calibration/start", {
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

export async function getSessionSummary() {
  return request<{
    time_in_state: Record<string, number>;
    recovery_trend: string;
    hrv_proxy_recovery: number | null;
    episodes: Array<Record<string, unknown>>;
    index_summary: { mean: number | null; max: number | null; latest: number | null };
  }>("/api/v1/session/summary");
}

export async function getInsights() {
  return request<{
    validation_scoreboard: any;
    nurse_context_insights_markdown: string | null;
    note: string;
  }>("/api/v1/insights");
}
