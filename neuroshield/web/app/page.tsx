"use client";

// Layout rule: the data leads, the explanation follows.
//
// This page used to open with four explainer cards and a settings panel, with the only actionable
// button buried among them -- so a first-time visitor read a wall of text, couldn't tell what to do,
// clicked start, saw the top of the page not change, and concluded it was broken. Now: before a
// session there is exactly one thing on screen to do; during a session the reading is the first
// thing you see, with visible progress; the explanation lives in its own tab.

import { useCallback, useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle } from "lucide-react";

import { AboutModel } from "@/components/AboutModel";
import { AxisBars } from "@/components/AxisBars";
import { GetStarted } from "@/components/GetStarted";
import { HowItWorks } from "@/components/HowItWorks";
import { SessionBar } from "@/components/SessionBar";
import { StatusHero } from "@/components/StatusHero";
import { SummaryView } from "@/components/SummaryView";
import { TrendsView } from "@/components/TrendsView";

import {
  Insights,
  SessionSummary,
  SystemInfo,
  getInsights,
  getSessionSummary,
  getSystem,
  startCalibration,
  startSession,
} from "@/lib/api";
import { fmt } from "@/lib/state";
import { useLiveFeed } from "@/lib/ws";
import { cn } from "@/lib/utils";

const QUIET_SECONDS = 150;
const DURATION_SEC = 600;

export default function Page() {
  const feed = useLiveFeed();

  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [speed, setSpeed] = useState(10);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    getSystem()
      .then(setSystem)
      .catch((e) => setError(String(e?.message ?? e)));
    getInsights()
      .then(setInsights)
      .catch(() => undefined);
  }, []);

  const nRecords = feed.records.length;
  useEffect(() => {
    if (nRecords === 0) return;
    getSessionSummary()
      .then(setSummary)
      .catch(() => undefined);
  }, [nRecords]);

  const runSession = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      feed.reset();
      await startSession({
        source_mode: "synthetic",
        session_id: "web-demo",
        duration_sec: DURATION_SEC,
        seed: 0,
        speed,
      });
      await startCalibration(QUIET_SECONDS);
      setStarted(true);
      setSystem(await getSystem());
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }, [feed, speed]);

  const latest = feed.latest;
  const hasSession = started || nRecords > 0;
  const backendDown = feed.connection === "closed" || Boolean(error);

  return (
    <main className="mx-auto max-w-6xl px-5 py-8">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">NeuroShield</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Reads a wrist wearable and tells you how aroused your body is{" "}
            <strong className="font-medium text-foreground">compared with your own calm baseline</strong>{" "}
            — in plain words, including when it can&apos;t tell.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              feed.connection === "open"
                ? "bg-calm"
                : feed.connection === "connecting"
                  ? "bg-elevated"
                  : "bg-destructive"
            )}
          />
          {feed.connection === "open" ? "Connected" : feed.connection === "connecting" ? "Connecting…" : "Offline"}
        </div>
      </header>

      <div className="space-y-4">
        {backendDown && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Can&apos;t reach the backend</AlertTitle>
            <AlertDescription>
              {error ?? "The live feed dropped."} Start it with{" "}
              <code className="rounded bg-black/10 px-1 py-0.5 font-mono text-xs">
                uv run uvicorn neuroshield.api.main:app --port 8000
              </code>
            </AlertDescription>
          </Alert>
        )}

        {system?.versionWarning && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Version mismatch</AlertTitle>
            <AlertDescription>{system.versionWarning}</AlertDescription>
          </Alert>
        )}

        {!hasSession ? (
          <>
            <GetStarted onRun={runSession} busy={busy} />
            <HowItWorks />
          </>
        ) : (
          <>
            <SessionBar
              nWindows={nRecords}
              complete={feed.complete}
              busy={busy}
              speed={speed}
              onSpeedChange={setSpeed}
              onRestart={runSession}
            />

            <Tabs defaultValue="now">
              <TabsList>
                <TabsTrigger value="now">Right now</TabsTrigger>
                <TabsTrigger value="trends">Over time</TabsTrigger>
                <TabsTrigger value="summary">Session</TabsTrigger>
                <TabsTrigger value="about">How it works</TabsTrigger>
              </TabsList>

              <TabsContent value="now" className="space-y-4">
                <StatusHero latest={latest} />
                {latest && (
                  <div className="grid gap-4 lg:grid-cols-2">
                    <AxisBars axes={latest.axes} />
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">The raw signals</CardTitle>
                        <CardDescription>
                          What the sensors measured in the last 60 seconds, before any modelling.
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="grid grid-cols-2 gap-5">
                        <Reading label="Heart rate" value={fmt(latest.values.hr_mean_bpm, 0)} unit="bpm" />
                        <Reading label="Sweat response" value={fmt(latest.values.eda_level, 2)} unit="µS" />
                        <Reading label="Skin temperature" value={fmt(latest.values.temp_mean_c, 1)} unit="°C" />
                        <Reading
                          label="Beat-to-beat variation"
                          value={fmt(latest.values.ibi_rmssd_ms, 0)}
                          unit="ms"
                        />
                        <div className="col-span-2 border-t pt-4">
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                            Can we trust this reading?
                          </p>
                          <dl className="mt-2 grid grid-cols-2 gap-x-5 gap-y-1.5 text-sm">
                            <Quality
                              label="Sensor coverage"
                              value={fmt(latest.quality.valid_fraction, 2)}
                              ok={(latest.quality.valid_fraction ?? 0) >= 0.9}
                              rule="needs ≥ 0.90"
                            />
                            <Quality
                              label="Pulse quality"
                              value={fmt(latest.quality.ppg_quality, 2)}
                              ok={(latest.quality.ppg_quality ?? 0) >= 0.7}
                              rule="needs ≥ 0.70"
                            />
                            <Quality
                              label="Hand movement"
                              value={fmt(latest.quality.motion_dynamic_rms, 2)}
                              ok={(latest.quality.motion_dynamic_rms ?? 0) <= 1.0}
                              rule="needs ≤ 1.0"
                            />
                          </dl>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="trends">
                <TrendsView records={feed.records} />
              </TabsContent>

              <TabsContent value="summary">
                <SummaryView summary={summary} />
              </TabsContent>

              <TabsContent value="about" className="space-y-4">
                <HowItWorks />
                <AboutModel system={system} insights={insights} />
              </TabsContent>
            </Tabs>
          </>
        )}
      </div>
    </main>
  );
}

function Reading({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="tnum mt-0.5 text-xl font-semibold">
        {value} <span className="text-sm font-normal text-muted-foreground">{unit}</span>
      </p>
    </div>
  );
}

function Quality({ label, value, ok, rule }: { label: string; value: string; ok: boolean; rule: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="tnum flex items-center justify-end gap-2">
        <span className={cn(!ok && "text-destructive")}>{value}</span>
        <span className="text-xs text-muted-foreground">{rule}</span>
      </dd>
    </>
  );
}
