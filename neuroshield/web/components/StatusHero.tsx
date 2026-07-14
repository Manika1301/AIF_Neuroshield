"use client";

// The single most important thing on the page: what is happening, right now, in words.
//
// The 0-100 number is meaningless on its own -- 62 out of what? measured how? -- so it never
// appears without (a) a plain-English verdict, (b) what it's relative to (your own baseline), and
// (c) the reasons behind it. If we can only show one of those, show the words.

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Activity, HeartPulse, PauseCircle } from "lucide-react";
import type { StatusRecord } from "@/lib/api";
import { AFFECT_COPY, fmt, indexMeaning, isPaused, levelTone, stateCopy, toneClasses } from "@/lib/state";
import { cn } from "@/lib/utils";

export function StatusHero({ latest }: { latest: StatusRecord | null }) {
  if (!latest) {
    return (
      <Card>
        <CardContent className="flex items-center gap-3 py-10 text-muted-foreground">
          <Activity className="h-5 w-5" />
          <span>No readings yet. Start a session below.</span>
        </CardContent>
      </Card>
    );
  }

  const copy = stateCopy(latest.state);
  const tone = toneClasses(copy.tone);
  const paused = isPaused(latest.state);
  const index = latest.stress_index;

  return (
    <Card className={cn("overflow-hidden border-l-4", tone.border)}>
      <CardContent className="p-0">
        <div className={cn("px-6 py-5", tone.bg)}>
          <div className="flex flex-wrap items-center gap-3">
            {paused ? (
              <PauseCircle className={cn("h-6 w-6", tone.text)} />
            ) : (
              <HeartPulse className={cn("h-6 w-6", tone.text)} />
            )}
            <h2 className={cn("text-2xl font-semibold tracking-tight", tone.text)}>{copy.title}</h2>
            {latest.level && !paused && (
              <Badge variant="outline" className={cn("border-current", tone.text)}>
                {latest.level}
              </Badge>
            )}
          </div>
          <p className={cn("mt-1.5 max-w-2xl text-sm", tone.text, "opacity-90")}>{copy.detail}</p>
        </div>

        <div className="grid gap-6 px-6 py-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          {/* The number, with everything it needs to mean something. */}
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Stress index</p>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="tnum text-6xl font-bold leading-none tracking-tight">
                {index ?? "--"}
              </span>
              <span className="text-lg text-muted-foreground">/ 100</span>
            </div>

            <Progress
              value={index ?? 0}
              className="mt-4 h-2"
              indicatorClassName={paused ? "bg-muted-foreground" : toneClasses(levelTone(latest.level)).fill}
            />

            <p className="mt-2 text-sm text-muted-foreground">
              {paused
                ? "Not scored — the model is paused (see above)."
                : `${indexMeaning(index)} Calibrated probability of stress: ${fmt(latest.probability, 2)}.`}
            </p>
          </div>

          {/* Why the number is what it is. This is the part people actually need. */}
          <div className="md:border-l md:pl-6">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Why</p>
            {latest.reasons.length > 0 ? (
              <ul className="mt-2 space-y-1.5">
                {latest.reasons.map((reason) => (
                  <li key={reason} className="flex gap-2 text-sm">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground" />
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-sm text-muted-foreground">
                Nothing to explain yet — the model hasn&apos;t scored a window.
              </p>
            )}

            {latest.affect_state && !paused && (
              <div className="mt-4 rounded-md border bg-muted/40 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Type of arousal
                  </span>
                  <Badge variant="secondary">
                    {AFFECT_COPY[latest.affect_state]?.label ?? latest.affect_state}
                  </Badge>
                  {latest.affect_confidence != null && (
                    <span className="tnum text-xs text-muted-foreground">
                      {Math.round(latest.affect_confidence * 100)}% confident
                    </span>
                  )}
                </div>
                <p className="mt-1.5 text-sm text-muted-foreground">
                  {AFFECT_COPY[latest.affect_state]?.help ??
                    "A second model separates stress from other kinds of arousal."}
                </p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
