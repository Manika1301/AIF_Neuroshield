// Display helpers, mirroring app/view_state.py.

export const STATE_DISPLAY: Record<string, [string, string]> = {
  waiting: ["Waiting for data", "#6b7280"],
  calibrating: ["Calibrating baseline", "#2563eb"],
  green: ["Green - calm", "#16a34a"],
  amber: ["Amber - elevated", "#d97706"],
  red: ["Red - high", "#dc2626"],
  motion_paused: ["Motion paused", "#6b7280"],
  poor_signal: ["Poor signal", "#6b7280"],
  stale: ["Stale - no recent data", "#d97706"],
  error: ["Backend reported an error", "#dc2626"],
  disconnected: ["Disconnected", "#dc2626"],
  backend_error: ["Backend error", "#dc2626"],
};

export function labelAndColor(state: string): [string, string] {
  return STATE_DISPLAY[state] ?? [state, "#6b7280"];
}

export function isColorState(state: string): boolean {
  return state === "green" || state === "amber" || state === "red";
}
