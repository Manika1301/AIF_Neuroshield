"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { SessionSummary } from "@/lib/api";
import { TREND_COPY, fmt, fmtDuration, stateCopy } from "@/lib/state";

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <Card>
      <CardContent className="py-5">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
        <p className="tnum mt-1 text-2xl font-semibold">{value}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

export function SummaryView({ summary }: { summary: SessionSummary | null }) {
  if (!summary || summary.n_windows === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-muted-foreground">
          No session to summarise yet.
        </CardContent>
      </Card>
    );
  }

  const states = Object.entries(summary.time_in_state ?? {}).filter(([, s]) => s > 0);
  const trend = TREND_COPY[summary.recovery_trend] ?? summary.recovery_trend;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Right now" value={trend} hint="Direction over the last 5 readings" />
        <Stat
          label="Peak"
          value={summary.index_summary?.max != null ? String(summary.index_summary.max) : "—"}
          hint="Highest index this session"
        />
        <Stat label="Average" value={fmt(summary.index_summary?.mean, 0)} hint="Mean index this session" />
        <Stat
          label="Recovery signal"
          value={fmt(summary.hrv_proxy_recovery, 0)}
          hint="Heart-rate variability, recent (higher = more recovered)"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Stress episodes</CardTitle>
          <CardDescription>
            Stretches of at least two consecutive elevated or high readings. A single spike is not an
            episode — bodies are noisy.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {summary.episodes?.length ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Started</TableHead>
                    <TableHead>Ended</TableHead>
                    <TableHead className="text-right">Lasted</TableHead>
                    <TableHead className="text-right">Peak index</TableHead>
                    <TableHead>Peak level</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {summary.episodes.map((ep, i) => (
                    <TableRow key={i}>
                      <TableCell className="tnum">{fmtDuration(ep.start_s)}</TableCell>
                      <TableCell className="tnum">{fmtDuration(ep.end_s)}</TableCell>
                      <TableCell className="tnum text-right">{fmtDuration(ep.end_s - ep.start_s)}</TableCell>
                      <TableCell className="tnum text-right">{ep.peak_index ?? "—"}</TableCell>
                      <TableCell>
                        <Badge variant={ep.peak_state === "red" ? "destructive" : "secondary"}>
                          {ep.peak_state === "red" ? "high" : "elevated"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No sustained episodes this session.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Where the time went</CardTitle>
          <CardDescription>
            Including time the model spent paused — that&apos;s a measure of signal quality, not of you.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>State</TableHead>
                  <TableHead className="text-right">Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {states.map(([state, seconds]) => (
                  <TableRow key={state}>
                    <TableCell>{stateCopy(state).title}</TableCell>
                    <TableCell className="tnum text-right">{fmtDuration(seconds)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
