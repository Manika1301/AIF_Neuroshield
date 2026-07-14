"use client";

// The four measured systems. Not model outputs -- these are the raw signals expressed against the
// user's own baseline, which is why they can be stated as fact rather than prediction.
//
// The bar is anchored at the CENTRE, because the score is signed: left of centre is below this
// person's baseline, right is above. A left-anchored bar (the obvious default) would render "your
// skin is cooler than usual" and "your skin is warmer than usual" identically.

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import type { StatusRecord } from "@/lib/api";
import { AXIS_COPY, AXIS_ORDER, describeAxis, toneClasses } from "@/lib/state";
import { cn } from "@/lib/utils";

const CLIP = 5; // axes.py clips scores to +/-5

function toneFor(level: string): "calm" | "elevated" | "high" | "neutral" {
  if (level === "high") return "high";
  if (level === "elevated") return "elevated";
  return "neutral";
}

export function AxisBars({ axes }: { axes: StatusRecord["axes"] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Which system is driving this?</CardTitle>
        <CardDescription>
          Each bar is one measured signal compared with <em>your</em> quiet baseline. Centre = normal
          for you; right = more aroused; left = less.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <TooltipProvider delayDuration={150}>
          {AXIS_ORDER.map((name) => {
            const axis = axes?.[name];
            const score = axis?.score ?? null;
            const level = axis?.level ?? "normal";
            const copy = AXIS_COPY[name];
            const tone = toneClasses(toneFor(level));

            const magnitude = score == null ? 0 : Math.min(Math.abs(score), CLIP) / CLIP;
            const widthPct = magnitude * 50;
            const positive = score == null || score >= 0;

            return (
              <div key={name}>
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm font-medium">{copy.label}</span>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          aria-label={`What ${copy.label} means`}
                          className="text-muted-foreground transition-colors hover:text-foreground"
                        >
                          <Info className="h-3.5 w-3.5" />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p>{copy.help}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <span className="tnum text-xs text-muted-foreground">{describeAxis(score)}</span>
                </div>

                <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-muted">
                  <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border" />
                  <span
                    className={cn("absolute inset-y-0 rounded-full transition-all", tone.fill)}
                    style={
                      positive
                        ? { left: "50%", width: `${widthPct}%` }
                        : { right: "50%", width: `${widthPct}%` }
                    }
                  />
                </div>
              </div>
            );
          })}
        </TooltipProvider>

        <p className="border-t pt-4 text-xs text-muted-foreground">
          These are measured, not predicted. Note the direction: falling heart-rate variability and{" "}
          <em>cooler</em> skin both mean <em>more</em> arousal, so they push the bar right.
        </p>
      </CardContent>
    </Card>
  );
}
