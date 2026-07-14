// Display helpers, mirroring app/view_state.py.

/** state -> [label, css class]. The class drives colour; see globals.css. */
export const STATE_DISPLAY: Record<string, [string, string]> = {
  waiting: ["Waiting for data", "s-neutral"],
  calibrating: ["Calibrating your baseline", "s-info"],
  green: ["Calm", "s-calm"],
  amber: ["Elevated", "s-elevated"],
  red: ["High", "s-high"],
  motion_paused: ["Paused — hand motion", "s-neutral"],
  poor_signal: ["Paused — poor signal", "s-neutral"],
  stale: ["Stale — no recent data", "s-elevated"],
  error: ["Backend reported an error", "s-high"],
  disconnected: ["Disconnected", "s-high"],
  backend_error: ["Backend error", "s-high"],
};

export function labelAndClass(state: string): [string, string] {
  return STATE_DISPLAY[state] ?? [state, "s-neutral"];
}

export function isColorState(state: string): boolean {
  return state === "green" || state === "amber" || state === "red";
}

/** True when the model deliberately declined to score this window (motion / signal quality). */
export function isAbstained(state: string): boolean {
  return state === "motion_paused" || state === "poor_signal";
}

export function levelClass(level: string | null): string {
  if (level === "calm") return "calm";
  if (level === "elevated") return "elevated";
  if (level === "high") return "high";
  return "";
}

export function fmt(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function fmtSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export const SERIES_COLORS = {
  index: "#2563eb",
  hr: "#dc2626",
  eda: "#16a34a",
} as const;
