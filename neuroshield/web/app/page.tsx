"use client";

// NeuroShield dashboard. Status arrives over the WebSocket (one push per 60s window); the slower
// derived views (session summary, research insights, system info) are fetched over REST.

import { CSSProperties, useCallback, useEffect, useState } from "react";
import {
  Insights,
  SessionSummary,
  StatusRecord,
  SystemInfo,
  getInsights,
  getSessionSummary,
  getSystem,
  startCalibration,
  startSession,
} from "@/lib/api";
import { SERIES_COLORS, fmt, fmtSeconds, isAbstained, labelAndClass, levelClass } from "@/lib/state";
import { useLiveFeed } from "@/lib/ws";
import { TimeSeries } from "./components/TimeSeries";

type Tab = "live" | "trends" | "summary" | "insights";
const TABS: Array<[Tab, string]> = [
  ["live", "Live"],
  ["trends", "Trends"],
  ["summary", "Session summary"],
  ["insights", "Research insights"],
];

export default function Page() {
  const feed = useLiveFeed();
  const [tab, setTab] = useState<Tab>("live");

  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [sourceMode, setSourceMode] = useState("synthetic");
  const [replayPath, setReplayPath] = useState("data/fixtures/calm_motion_stress.ndjson");
  const [durationSec, setDurationSec] = useState(600);
  const [seed, setSeed] = useState(0);
  const [speed, setSpeed] = useState(10);
  const [quietSeconds, setQuietSeconds] = useState(150);

  useEffect(() => {
    getSystem()
      .then(setSystem)
      .catch((e) => setError(String(e?.message ?? e)));
    getInsights()
      .then(setInsights)
      .catch(() => undefined);
  }, []);

  // The summary is derived from the whole history, so refresh it as the feed advances rather than
  // on a timer -- no new windows, no change to compute.
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
        source_mode: sourceMode,
        session_id: "web-demo",
        replay_path: sourceMode === "replay" ? replayPath : null,
        duration_sec: durationSec,
        seed,
        speed,
      });
      await startCalibration(quietSeconds);
      setSystem(await getSystem());
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }, [feed, sourceMode, replayPath, durationSec, seed, speed, quietSeconds]);

  const latest = feed.latest;
  const connLabel =
    feed.connection === "open" ? "Live" : feed.connection === "connecting" ? "Connecting…" : "Disconnected";

  return (
    <main className="page">
      <header className="masthead">
        <h1>NeuroShield</h1>
        <span className="conn">
          <span className={`dot ${feed.connection}`} />
          {connLabel}
          {feed.complete && " · session complete"}
        </span>
      </header>
      <p className="subtitle">
        Wearable stress signals, relative to your own baseline.{" "}
        {system?.model_version && <span className="mono">{system.model_version}</span>} · Not a medical device.
      </p>

      {system?.versionWarning && <div className="banner warn">{system.versionWarning}</div>}
      {error && <div className="banner err">{error}</div>}
      {feed.error && <div className="banner err">Stream error: {feed.error}</div>}
      {feed.connection === "closed" && !error && (
        <div className="banner err">
          Lost the live feed — retrying. Is the backend running on <span className="mono">127.0.0.1:8000</span>?
        </div>
      )}

      <SessionControls
        sourceMode={sourceMode}
        setSourceMode={setSourceMode}
        replayPath={replayPath}
        setReplayPath={setReplayPath}
        durationSec={durationSec}
        setDurationSec={setDurationSec}
        seed={seed}
        setSeed={setSeed}
        speed={speed}
        setSpeed={setSpeed}
        quietSeconds={quietSeconds}
        setQuietSeconds={setQuietSeconds}
        busy={busy}
        onRun={runSession}
      />

      <nav className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={`tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>

      {tab === "live" && <LiveView latest={latest} nWindows={nRecords} />}
      {tab === "trends" && <TrendsView records={feed.records} />}
      {tab === "summary" && <SummaryView summary={summary} />}
      {tab === "insights" && <InsightsView insights={insights} />}
    </main>
  );
}

interface ControlProps {
  sourceMode: string;
  setSourceMode: (v: string) => void;
  replayPath: string;
  setReplayPath: (v: string) => void;
  durationSec: number;
  setDurationSec: (v: number) => void;
  seed: number;
  setSeed: (v: number) => void;
  speed: number;
  setSpeed: (v: number) => void;
  quietSeconds: number;
  setQuietSeconds: (v: number) => void;
  busy: boolean;
  onRun: () => void;
}

function SessionControls(props: ControlProps) {
  return (
    <section className="card" style={{ marginBottom: 16 }}>
      <h2>Session</h2>
      <div className="controls">
        <div className="field">
          <label htmlFor="src">Source</label>
          <select id="src" value={props.sourceMode} onChange={(e) => props.setSourceMode(e.target.value)}>
            <option value="synthetic">Synthetic</option>
            <option value="replay">Replay file</option>
          </select>
        </div>

        {props.sourceMode === "replay" ? (
          <div className="field" style={{ flex: 1, minWidth: 240 }}>
            <label htmlFor="path">Replay path</label>
            <input
              id="path"
              style={{ width: "100%" }}
              value={props.replayPath}
              onChange={(e) => props.setReplayPath(e.target.value)}
            />
          </div>
        ) : (
          <>
            <div className="field">
              <label htmlFor="dur">Duration (s)</label>
              <input
                id="dur"
                type="number"
                value={props.durationSec}
                onChange={(e) => props.setDurationSec(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label htmlFor="seed">Seed</label>
              <input id="seed" type="number" value={props.seed} onChange={(e) => props.setSeed(Number(e.target.value))} />
            </div>
          </>
        )}

        <div className="field">
          <label htmlFor="quiet">Calibration (s)</label>
          <input
            id="quiet"
            type="number"
            value={props.quietSeconds}
            onChange={(e) => props.setQuietSeconds(Number(e.target.value))}
          />
        </div>

        <div className="field">
          <label htmlFor="speed">Playback</label>
          <select id="speed" value={props.speed} onChange={(e) => props.setSpeed(Number(e.target.value))}>
            <option value={1}>1× (real time)</option>
            <option value={10}>10×</option>
            <option value={30}>30×</option>
            <option value={0}>As fast as possible</option>
          </select>
        </div>

        <button className="primary" onClick={props.onRun} disabled={props.busy}>
          {props.busy ? "Starting…" : "Start session"}
        </button>
      </div>
      <p className="muted" style={{ marginTop: 10, marginBottom: 0 }}>
        Calibration measures your quiet baseline. Every reading below is expressed relative to it.
      </p>
    </section>
  );
}

function Metric({ k, v, sub }: { k: string; v: string; sub?: string }) {
  return (
    <div className="metric">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
      {sub && <span className="sub">{sub}</span>}
    </div>
  );
}

function LiveView({ latest, nWindows }: { latest: StatusRecord | null; nWindows: number }) {
  if (!latest) {
    return <p className="empty">No data yet. Start a session to begin streaming.</p>;
  }

  const [label, cls] = labelAndClass(latest.state);

  return (
    <div className="stack">
      <div className={`status-badge ${cls}`}>
        <span>{label}</span>
        {latest.reasons.length > 0 && <span className="reasons">{latest.reasons.join(" ")}</span>}
      </div>

      {isAbstained(latest.state) && (
        <div className="banner info">
          The model is <strong>not guessing</strong> right now. A wrist signal during hand motion or poor
          skin contact isn&apos;t a reliable stress reading, so this window is deliberately left unscored.
        </div>
      )}

      <div className="grid cols-2">
        <section className="card">
          <h2>Stress index</h2>
          <div className="index-hero">
            <span className="index-value">{latest.stress_index ?? "—"}</span>
            <span className="index-max">/ 100</span>
            {latest.level && <span className={`chip ${levelClass(latest.level)}`}>{latest.level}</span>}
          </div>
          <p className="muted" style={{ marginTop: 10, marginBottom: 0 }}>
            {latest.probability != null
              ? `Calibrated P(stress) = ${fmt(latest.probability, 2)}`
              : "Not scored for this window."}
          </p>
        </section>

        <section className="card">
          <h2>Affect state</h2>
          <div className="grid cols-2">
            <Metric
              k="State"
              v={latest.affect_state ?? "—"}
              sub="Separates stress from positive arousal"
            />
            <Metric
              k="Confidence"
              v={latest.affect_confidence != null ? `${Math.round(latest.affect_confidence * 100)}%` : "—"}
              sub="One lab protocol — suggestive only"
            />
          </div>
        </section>
      </div>

      <section className="card">
        <h2>Which system is driving this?</h2>
        <Axes axes={latest.axes} />
      </section>

      <div className="grid cols-2">
        <section className="card">
          <h2>Latest values</h2>
          <div className="grid cols-2">
            <Metric k="Heart rate" v={fmt(latest.values.hr_mean_bpm, 0)} sub="bpm" />
            <Metric k="Skin conductance" v={fmt(latest.values.eda_level, 2)} sub="µS" />
            <Metric k="Skin temp" v={fmt(latest.values.temp_mean_c, 1)} sub="°C" />
            <Metric k="Pulse variability" v={fmt(latest.values.ibi_rmssd_ms, 0)} sub="RMSSD ms" />
          </div>
        </section>

        <section className="card">
          <h2>Signal quality</h2>
          <div className="grid cols-2">
            <Metric k="Coverage" v={fmt(latest.quality.valid_fraction, 2)} sub="≥ 0.90 to score" />
            <Metric k="Pulse quality" v={fmt(latest.quality.ppg_quality, 2)} sub="≥ 0.70 to score" />
            <Metric k="Motion (rms)" v={fmt(latest.quality.motion_dynamic_rms, 2)} sub="≤ 1.0 to score" />
            <Metric k="Windows" v={String(nWindows)} sub="streamed this session" />
          </div>
        </section>
      </div>
    </div>
  );
}

const AXIS_NAMES = ["cardiac", "electrodermal", "thermal", "movement"];
const AXIS_CLIP = 5; // axes.py clips scores to +/-5

function Axes({ axes }: { axes: StatusRecord["axes"] }) {
  return (
    <div>
      {AXIS_NAMES.map((name) => {
        const axis = axes?.[name];
        const score = axis?.score ?? null;
        const level = axis?.level ?? "normal";

        // The bar is anchored at the midpoint because the score is signed: left of centre is BELOW
        // this person's own baseline, right is above. A left-anchored bar would misread the sign.
        const magnitude = score == null ? 0 : Math.min(Math.abs(score), AXIS_CLIP) / AXIS_CLIP;
        const widthPct = magnitude * 50;
        const style: CSSProperties =
          score == null || score >= 0
            ? { left: "50%", width: `${widthPct}%` }
            : { right: "50%", width: `${widthPct}%` };

        return (
          <div className="axis" key={name}>
            <div className="axis-head">
              <span className="axis-name">{name}</span>
              <span className="axis-score">
                {score == null ? "no data" : `${score > 0 ? "+" : ""}${fmt(score, 2)} SD`}
              </span>
            </div>
            <div className="axis-track">
              <span className="axis-mid" />
              <span className={`axis-fill ${level}`} style={style} />
            </div>
          </div>
        );
      })}
      <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
        Standard deviations from <em>your</em> quiet baseline, in the arousal direction. These are
        measured, not predicted — falling pulse variability and cooling skin both mean more arousal.
      </p>
    </div>
  );
}

function TrendsView({ records }: { records: StatusRecord[] }) {
  if (records.length === 0) {
    return <p className="empty">No windows yet. Start a session to see trends.</p>;
  }

  const x = records.map((r) => r.window_start_s ?? 0);

  return (
    <div className="stack">
      <section className="card">
        <h2>Stress index over time</h2>
        <TimeSeries
          x={x}
          yMin={0}
          yMax={100}
          series={[{ label: "Stress index", color: SERIES_COLORS.index, points: records.map((r) => r.stress_index) }]}
        />
        <p className="muted" style={{ marginBottom: 0 }}>
          Breaks in the line are windows the model declined to score (motion or poor signal).
        </p>
      </section>

      <section className="card">
        <h2>Heart rate</h2>
        <TimeSeries
          x={x}
          series={[
            {
              label: "Heart rate (bpm)",
              color: SERIES_COLORS.hr,
              points: records.map((r) => r.values.hr_mean_bpm ?? null),
            },
          ]}
        />
      </section>

      <section className="card">
        <h2>Skin conductance</h2>
        <TimeSeries
          x={x}
          series={[
            {
              label: "Skin conductance (µS)",
              color: SERIES_COLORS.eda,
              points: records.map((r) => r.values.eda_level ?? null),
            },
          ]}
        />
      </section>
    </div>
  );
}

function SummaryView({ summary }: { summary: SessionSummary | null }) {
  if (!summary) {
    return <p className="empty">No session summary yet.</p>;
  }

  const states = Object.entries(summary.time_in_state ?? {}).filter(([, seconds]) => seconds > 0);

  return (
    <div className="stack">
      <div className="grid cols-4">
        <section className="card">
          <Metric k="Recovery trend" v={summary.recovery_trend ?? "—"} sub="last 5 windows" />
        </section>
        <section className="card">
          <Metric
            k="Peak index"
            v={summary.index_summary?.max != null ? String(summary.index_summary.max) : "—"}
            sub="highest this session"
          />
        </section>
        <section className="card">
          <Metric k="Mean index" v={fmt(summary.index_summary?.mean, 0)} sub="session average" />
        </section>
        <section className="card">
          <Metric k="HRV proxy" v={fmt(summary.hrv_proxy_recovery, 0)} sub="RMSSD ms, recent" />
        </section>
      </div>

      <section className="card">
        <h2>Stress episodes</h2>
        {summary.episodes?.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Start</th>
                  <th>End</th>
                  <th className="num">Windows</th>
                  <th className="num">Peak index</th>
                  <th>Peak state</th>
                </tr>
              </thead>
              <tbody>
                {summary.episodes.map((ep, i) => (
                  <tr key={i}>
                    <td>{fmtSeconds(ep.start_s)}</td>
                    <td>{fmtSeconds(ep.end_s)}</td>
                    <td className="num">{ep.n_windows}</td>
                    <td className="num">{ep.peak_index ?? "—"}</td>
                    <td>
                      <span className={`chip ${ep.peak_state === "red" ? "high" : "elevated"}`}>{ep.peak_state}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted" style={{ margin: 0 }}>
            No sustained episodes (a run of at least two consecutive elevated or high windows).
          </p>
        )}
      </section>

      <section className="card">
        <h2>Time in each state</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>State</th>
                <th className="num">Time</th>
              </tr>
            </thead>
            <tbody>
              {states.map(([state, seconds]) => (
                <tr key={state}>
                  <td>{labelAndClass(state)[0]}</td>
                  <td className="num">{fmtSeconds(seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function InsightsView({ insights }: { insights: Insights | null }) {
  if (!insights) {
    return <p className="empty">No research artifacts available.</p>;
  }

  const scoreboard = insights.validation_scoreboard;
  const perDataset = scoreboard?.per_dataset;
  const stale = scoreboard?.STALE;

  return (
    <div className="stack">
      <section className="card">
        <h2>Cross-dataset validation</h2>

        {/* Never present superseded numbers as current: the scoreboard on disk predates the
            shipped model, and the backend flags that explicitly. */}
        {stale && (
          <div className="banner warn">
            <strong>These numbers are out of date.</strong> {stale}
          </div>
        )}

        {perDataset ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Evaluation</th>
                  <th className="num">Balanced acc.</th>
                  <th className="num">Macro F1</th>
                  <th className="num">Windows</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(perDataset).map(([name, m]) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td className="muted">{m.evaluation}</td>
                    <td className="num">{fmt(m.balanced_accuracy, 3)}</td>
                    <td className="num">{fmt(m.macro_f1, 3)}</td>
                    <td className="num">{m.n_windows}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted" style={{ margin: 0 }}>
            No scoreboard on disk. Regenerate with <span className="mono">scripts/build_scoreboard.py</span>.
          </p>
        )}
      </section>

      {insights.nurse_context_insights_markdown && (
        <section className="card">
          <h2>Nurse shift context</h2>
          <pre className="raw">{insights.nurse_context_insights_markdown}</pre>
        </section>
      )}

      <div className="banner info">{insights.note}</div>
    </div>
  );
}
