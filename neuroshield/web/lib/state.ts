// The plain-language layer.
//
// The backend speaks in the vocabulary of the research: `electrodermal`, `poor_signal`,
// `hrv_proxy_recovery`, `stress_prob`. That vocabulary is correct and it stays in the API -- but a
// person looking at a dashboard should never have to decode it. Every user-facing string in the app
// is translated here, once, so the UI cannot drift into jargon by accident.

import type { StatusRecord } from "./api";

export type Tone = "calm" | "elevated" | "high" | "neutral" | "info";

export interface StateCopy {
  /** What the app is doing/seeing, in a few words. */
  title: string;
  /** One sentence a non-expert can act on. */
  detail: string;
  tone: Tone;
}

export const STATE_COPY: Record<string, StateCopy> = {
  waiting: {
    title: "Waiting for data",
    detail: "No session is running yet. Start one below to begin reading the wearable.",
    tone: "neutral",
  },
  calibrating: {
    title: "Learning your baseline",
    detail:
      "Measuring what quiet looks like for you. Everything afterwards is judged against this, not against a population average.",
    tone: "info",
  },
  green: {
    title: "Calm",
    detail: "Your body's arousal is at or near your own resting level.",
    tone: "calm",
  },
  amber: {
    title: "Elevated",
    detail: "Arousal is above your resting level. This is a signal, not a diagnosis.",
    tone: "elevated",
  },
  red: {
    title: "High",
    detail: "Arousal is well above your resting level and has stayed there.",
    tone: "high",
  },
  motion_paused: {
    title: "Paused — you're moving",
    detail:
      "A wrist sensor can't read pulse reliably through hand movement, so the model has stopped scoring rather than guess.",
    tone: "neutral",
  },
  poor_signal: {
    title: "Paused — weak signal",
    detail:
      "The sensor isn't making good contact, so the reading would be unreliable. The model has stopped scoring rather than guess.",
    tone: "neutral",
  },
  stale: {
    title: "No recent data",
    detail: "The wearable hasn't sent anything for a while. The last reading is too old to trust.",
    tone: "elevated",
  },
  error: {
    title: "Something went wrong",
    detail: "The backend reported an error. Check the server logs.",
    tone: "high",
  },
};

export function stateCopy(state: string): StateCopy {
  return (
    STATE_COPY[state] ?? {
      title: state,
      detail: "Unrecognised state.",
      tone: "neutral",
    }
  );
}

/** True when the model deliberately declined to score this window. */
export function isPaused(state: string): boolean {
  return state === "motion_paused" || state === "poor_signal";
}

/** The four measured systems, in words rather than physiology terms. */
export const AXIS_COPY: Record<string, { label: string; help: string }> = {
  cardiac: {
    label: "Heart",
    help: "Heart rate and how much it varies beat to beat. Under stress the rate rises and the variation shrinks.",
  },
  electrodermal: {
    label: "Sweat response",
    help: "Tiny changes in skin conductance from sweat glands. The most direct read on arousal — and it rises whether the arousal is good or bad.",
  },
  thermal: {
    label: "Skin temperature",
    help: "Under stress, blood moves away from the extremities and skin gets cooler. So cooler means more aroused here, not less.",
  },
  movement: {
    label: "Movement",
    help: "How much the wrist is moving. Shown because motion is also what makes a reading untrustworthy.",
  },
};

export const AXIS_ORDER = ["cardiac", "electrodermal", "thermal", "movement"];

/** What each affect class actually means to a user. */
export const AFFECT_COPY: Record<string, { label: string; help: string }> = {
  baseline: { label: "Baseline", help: "Resting — nothing much going on." },
  stress: { label: "Stress", help: "Arousal that looks like strain rather than enjoyment." },
  amusement: {
    label: "Amusement",
    help: "Aroused, but in a positive way. This is why a raised heart rate alone isn't enough to call something stress.",
  },
  meditation: { label: "Calm focus", help: "Deliberately relaxed — lower arousal than plain rest." },
};

export const TREND_COPY: Record<string, string> = {
  rising: "Getting more stressed",
  falling: "Recovering",
  steady: "Holding steady",
};

/** What a 0-100 index actually means, in a sentence. */
export function indexMeaning(index: number | null): string {
  if (index == null) return "Not scored for this window.";
  if (index < 45) return "In your normal resting range.";
  if (index < 70) return "Above your resting range.";
  return "Well above your resting range.";
}

export function toneClasses(tone: Tone): { bg: string; text: string; border: string; fill: string } {
  switch (tone) {
    case "calm":
      return {
        bg: "bg-calm-muted",
        text: "text-calm-foreground",
        border: "border-calm/40",
        fill: "bg-calm",
      };
    case "elevated":
      return {
        bg: "bg-elevated-muted",
        text: "text-elevated-foreground",
        border: "border-elevated/40",
        fill: "bg-elevated",
      };
    case "high":
      return {
        bg: "bg-high-muted",
        text: "text-high-foreground",
        border: "border-high/40",
        fill: "bg-high",
      };
    case "info":
      return {
        bg: "bg-secondary",
        text: "text-secondary-foreground",
        border: "border-border",
        fill: "bg-primary",
      };
    default:
      return {
        bg: "bg-muted",
        text: "text-muted-foreground",
        border: "border-border",
        fill: "bg-muted-foreground",
      };
  }
}

export function levelTone(level: string | null): Tone {
  if (level === "calm") return "calm";
  if (level === "elevated") return "elevated";
  if (level === "high") return "high";
  return "neutral";
}

export function fmt(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/** The axis score is clipped to this in axes.py, so anything at the cap is really "at least". */
const AXIS_CLIP = 5;

/**
 * Describe a signed axis score.
 *
 * The score is an *arousal-direction* aggregate, NOT a raw deviation: for skin temperature the
 * sign is flipped, because cooler skin means more arousal. So "+5" on the thermal axis means
 * "strongly toward arousal" — i.e. the skin got *colder*. Wording it as "above your baseline"
 * would tell the user their skin is warm while the reasons panel tells them it's cool. Everything
 * here is therefore phrased in terms of arousal, which is what the number actually measures.
 */
export function describeAxis(score: number | null): string {
  if (score == null) return "Not measurable this window";
  const magnitude = Math.abs(score);
  if (magnitude < 1) return "Normal for you";

  const capped = magnitude >= AXIS_CLIP;
  const amount = capped ? `${AXIS_CLIP}+` : fmt(magnitude, 1);
  const direction = score >= 0 ? "toward arousal" : "toward calm";
  return `${amount} SD ${direction}`;
}

export function scoredRecords(records: StatusRecord[]): StatusRecord[] {
  return records.filter((r) => r.stress_index != null);
}
