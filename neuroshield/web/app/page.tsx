"use client";

import { useCallback, useEffect, useState } from "react";
import {
  BackendUnreachableError,
  BackendValidationError,
  getHealth,
  getHistory,
  getInsights,
  getSessionSummary,
  getStatusLatest,
  getSystem,
  startCalibration,
  startSession,
  type StatusRecord,
} from "@/lib/api";
import { isColorState, labelAndColor } from "@/lib/state";

type Tab = "live" | "summary" | "trends" | "insights";

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("live");
  const [status, setStatus] = useState<StatusRecord | null>(null);
  const [history, setHistory] = useState<StatusRecord[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [insights, setInsights] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [replayPath, setReplayPath] = useState(
    "data/fixtures/calm_motion_stress.ndjson"
  );

  const refresh = useCallback(async () => {
    try {
      await getSystem(); // validates schema/feature version before trusting anything
      setHealth(await getHealth());
      setStatus(await getStatusLatest());
      setHistory(await getHistory(40));
      try {
        setSummary(await getSessionSummary());
      } catch {
        setSummary(null);
      }
      try {
        setInsights(await getInsights());
      } catch {
        setInsights(null);
      }
      setError(null);
    } catch (err) {
      if (err instanceof BackendUnreachableError) setError(`Disconnected: ${err.message}`);
      else if (err instanceof BackendValidationError) setError(`Backend error: ${err.message}`);
      else setError(String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  async function onStart() {
    try {
      await startSession({ source_mode: "replay", replay_path: replayPath, session_id: "web-demo" });
      await startCalibration(150);
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  if (error) {
    return (
      <main style={{ maxWidth: 960, margin: "40px auto", padding: 24 }}>
        <h1>NeuroShield</h1>
        <div style={{ padding: 16, background: "#fee2e2", borderRadius: 8, color: "#991b1b" }}>
          {error}
        </div>
      </main>
    );
  }

  const [label, color] = status ? labelAndColor(status.state) : ["Loading...", "#6b7280"];

  return (
    <main style={{ maxWidth: 960, margin: "24px auto", padding: 24 }}>
      <h1 style={{ marginBottom: 4 }}>NeuroShield</h1>
      <p style={{ color: "#64748b", marginTop: 0 }}>
        Personalized, multi-dimensional stress &amp; recovery
      </p>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={replayPath}
          onChange={(e) => setReplayPath(e.target.value)}
          style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #cbd5e1" }}
        />
        <button onClick={onStart} style={btnStyle}>
          Start replay + calibrate
        </button>
      </div>

      <nav style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {(["live", "summary", "trends", "insights"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{ ...tabStyle, ...(tab === t ? activeTab : {}) }}
          >
            {t}
          </button>
        ))}
      </nav>

      {tab === "live" && <LiveView status={status} health={health} color={color} label={label} />}
      {tab === "summary" && <SummaryView summary={summary} />}
      {tab === "trends" && <TrendsView history={history} />}
      {tab === "insights" && <InsightsView insights={insights} />}
    </main>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ background: "white", borderRadius: 10, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.08)", marginBottom: 12 }}>
      {children}
    </div>
  );
}

function LiveView({ status, health, color, label }: any) {
  if (!status) return <Card>Loading status...</Card>;
  return (
    <>
      <Card>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ width: 20, height: 20, borderRadius: "50%", background: color }} />
          <h2 style={{ margin: 0 }}>{label}</h2>
        </div>
        <div style={{ display: "flex", gap: 32, marginTop: 16 }}>
          <Metric label="Stress index" value={status.stress_index ?? "n/a"} />
          <Metric label="Level" value={status.level ?? "n/a"} />
          <Metric label="Affect" value={status.affect_state ?? "n/a"} />
          <Metric label="Baseline" value={String(health?.baseline_loaded ?? false)} />
        </div>
      </Card>

      <Card>
        <h3>Four physiological axes</h3>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr><th style={th}>Axis</th><th style={th}>Score</th><th style={th}>Level</th></tr>
          </thead>
          <tbody>
            {Object.entries(status.axes ?? {}).map(([name, a]: any) => (
              <tr key={name}>
                <td style={td}>{name}</td>
                <td style={td}>{a.score == null ? "n/a" : a.score.toFixed(2)}</td>
                <td style={td}>{a.level}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card>
        <h3>Reasons</h3>
        {status.reasons?.length ? (
          <ul>{status.reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul>
        ) : (
          <p style={{ color: "#64748b" }}>
            {isColorState(status.state) ? "No reasons for this window." : "Waiting for a scored window."}
          </p>
        )}
      </Card>
    </>
  );
}

function SummaryView({ summary }: any) {
  if (!summary) return <Card>No session summary yet — start a session.</Card>;
  return (
    <Card>
      <h3>Session summary</h3>
      <div style={{ display: "flex", gap: 32 }}>
        <Metric label="Recovery trend" value={summary.recovery_trend} />
        <Metric label="Peak index" value={summary.index_summary?.max ?? "n/a"} />
        <Metric label="HRV-proxy" value={summary.hrv_proxy_recovery ?? "n/a"} />
        <Metric label="Episodes" value={(summary.episodes ?? []).length} />
      </div>
      <h4>Time in state (s)</h4>
      <pre style={preStyle}>{JSON.stringify(summary.time_in_state, null, 2)}</pre>
    </Card>
  );
}

function TrendsView({ history }: { history: StatusRecord[] }) {
  if (!history.length) return <Card>No history yet.</Card>;
  const max = 100;
  return (
    <Card>
      <h3>Stress index over the session</h3>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 160 }}>
        {history.map((r, i) => (
          <div
            key={i}
            title={`${r.state} (${r.stress_index ?? "-"})`}
            style={{
              flex: 1,
              height: `${((r.stress_index ?? 0) / max) * 100}%`,
              background: labelAndColor(r.state)[1],
              minHeight: 2,
            }}
          />
        ))}
      </div>
    </Card>
  );
}

function InsightsView({ insights }: any) {
  if (!insights) return <Card>No insights available.</Card>;
  return (
    <Card>
      <h3>Research insights (descriptive only)</h3>
      <p style={{ color: "#64748b" }}>{insights.note}</p>
      {insights.validation_scoreboard && (
        <>
          <h4>3-dataset validation scoreboard</h4>
          <pre style={preStyle}>
            {JSON.stringify(insights.validation_scoreboard.per_dataset, null, 2)}
          </pre>
        </>
      )}
      {insights.nurse_context_insights_markdown && (
        <>
          <h4>Nurse context co-occurrence</h4>
          <pre style={preStyle}>{insights.nurse_context_insights_markdown}</pre>
        </>
      )}
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: "#64748b" }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 600 }}>{value}</div>
    </div>
  );
}

const btnStyle: React.CSSProperties = { padding: "8px 16px", borderRadius: 6, border: "none", background: "#1f6fb2", color: "white", cursor: "pointer" };
const tabStyle: React.CSSProperties = { padding: "6px 14px", borderRadius: 6, border: "1px solid #cbd5e1", background: "white", cursor: "pointer", textTransform: "capitalize" };
const activeTab: React.CSSProperties = { background: "#1f6fb2", color: "white", border: "1px solid #1f6fb2" };
const th: React.CSSProperties = { textAlign: "left", borderBottom: "1px solid #e2e8f0", padding: 6, fontSize: 13, color: "#64748b" };
const td: React.CSSProperties = { padding: 6, borderBottom: "1px solid #f1f5f9" };
const preStyle: React.CSSProperties = { background: "#f8fafc", padding: 12, borderRadius: 6, overflow: "auto", fontSize: 12 };
