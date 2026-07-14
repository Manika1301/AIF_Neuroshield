"use client";

// Model transparency, in the product rather than buried in a docs folder.
//
// The accuracy number is worthless without its error bar, and worse than worthless without the
// per-subject spread: this model is good on average and can be near-useless on an individual. A
// user of a health dashboard is entitled to know that before they trust a reading about themselves.

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { AlertTriangle, Stethoscope } from "lucide-react";
import type { Insights, SystemInfo } from "@/lib/api";
import { fmt } from "@/lib/state";

const INPUTS = [
  { group: "Pulse", detail: "Heart rate, beat-to-beat variability, frequency-domain HRV, signal quality" },
  { group: "Sweat response", detail: "Level, slope, response count and size, plus a tonic/phasic split (cvxEDA)" },
  { group: "Skin temperature", detail: "Mean and trend" },
  { group: "Movement", detail: "Motion energy and peaks" },
];

const OUTPUTS = [
  { name: "Stress index", detail: "0–100, from a calibrated probability. The headline number." },
  { name: "Level", detail: "calm / elevated / high — bands on that same probability." },
  { name: "Type of arousal", detail: "baseline / stress / amusement / calm focus, from a second model." },
  { name: "Four systems", detail: "Measured signals vs your baseline. Not model output — direct measurement." },
  { name: "Paused", detail: "The model declining to score, when motion or signal quality make it unreliable." },
];

export function AboutModel({ system, insights }: { system: SystemInfo | null; insights: Insights | null }) {
  const scoreboard = insights?.validation_scoreboard;
  const perDataset = scoreboard?.per_dataset;
  const stale = scoreboard?.STALE;

  return (
    <div className="space-y-4">
      <Alert>
        <Stethoscope className="h-4 w-4" />
        <AlertTitle>This is not a medical device</AlertTitle>
        <AlertDescription>
          It measures <strong>physiological arousal</strong> relative to your own baseline. It does not
          detect, diagnose, or predict panic attacks, anxiety disorders, burnout, or any medical
          condition — and it is built so that it cannot claim to.
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">How accurate is it, honestly?</CardTitle>
          <CardDescription>
            Measured by leave-one-subject-out: the model is only ever scored on people it never trained
            on. That&apos;s the number that answers &quot;will this work on someone new&quot;.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="tnum text-4xl font-bold tracking-tight">0.919</span>
            <span className="text-sm text-muted-foreground">balanced accuracy (±0.036)</span>
            <Badge variant="secondary">vs 0.500 chance</Badge>
          </div>

          <Alert variant="destructive" className="bg-elevated-muted text-elevated-foreground [&>svg]:text-elevated-foreground border-elevated/40">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Good on average. Not uniformly good.</AlertTitle>
            <AlertDescription>
              Across the 15 test subjects, per-person accuracy ranged from <strong>0.50 (no better than a
              coin flip)</strong> to <strong>1.00</strong>. Treat the index as a trend for one person over
              time, not a precise measurement of how stressed they are right now.
            </AlertDescription>
          </Alert>

          <Separator />

          <div>
            <p className="text-sm font-medium">Cross-dataset validation</p>
            {stale ? (
              <Alert className="mt-2">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>These numbers are out of date</AlertTitle>
                <AlertDescription>{stale}</AlertDescription>
              </Alert>
            ) : null}

            {perDataset ? (
              <div className="mt-3 overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Dataset</TableHead>
                      <TableHead>How it was scored</TableHead>
                      <TableHead className="text-right">Bal. accuracy</TableHead>
                      <TableHead className="text-right">Windows</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(perDataset).map(([name, m]) => (
                      <TableRow key={name}>
                        <TableCell className="font-mono text-xs">{name}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">{m.evaluation}</TableCell>
                        <TableCell className="tnum text-right">{fmt(m.balanced_accuracy, 3)}</TableCell>
                        <TableCell className="tnum text-right">{m.n_windows}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <p className="mt-2 text-sm text-muted-foreground">
                No scoreboard available on the backend.
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">What goes in</CardTitle>
            <CardDescription>
              36 numbers per 60-second window: each signal in absolute units, and again as a deviation
              from your own baseline.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {INPUTS.map((i) => (
              <div key={i.group}>
                <p className="text-sm font-medium">{i.group}</p>
                <p className="text-xs text-muted-foreground">{i.detail}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">What comes out</CardTitle>
            <CardDescription>Everything the app shows you, and where it comes from.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {OUTPUTS.map((o) => (
              <div key={o.name}>
                <p className="text-sm font-medium">{o.name}</p>
                <p className="text-xs text-muted-foreground">{o.detail}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {system && (
        <p className="text-xs text-muted-foreground">
          Model <span className="font-mono">{system.model_version}</span> · features{" "}
          <span className="font-mono">{system.feature_version}</span>
        </p>
      )}
    </div>
  );
}
