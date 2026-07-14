"use client";

// A minimal inline-SVG line chart. Deliberately dependency-free: adding a charting library for a
// handful of series would be the only heavy dependency in the app.
//
// Gaps matter here. A window where the model abstained (motion_paused / poor_signal) has no stress
// index, and drawing a straight line across it would invent data the model explicitly refused to
// produce. Null runs are therefore breaks in the line, not interpolations.

export interface Series {
  label: string;
  color: string;
  points: Array<number | null>;
}

interface Props {
  x: number[]; // window start seconds
  series: Series[];
  height?: number;
  yLabel?: string;
  yMin?: number;
  yMax?: number;
}

const PAD = { top: 10, right: 12, bottom: 24, left: 40 };
const WIDTH = 720;

export function TimeSeries({ x, series, height = 220, yLabel, yMin, yMax }: Props) {
  const values = series.flatMap((s) => s.points).filter((v): v is number => v != null);
  if (x.length === 0 || values.length === 0) {
    return <p className="empty">No data to plot yet.</p>;
  }

  const lo = yMin ?? Math.min(...values);
  const hi = yMax ?? Math.max(...values);
  const span = hi - lo || 1;

  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  const xAt = (i: number) => PAD.left + (x.length === 1 ? plotW / 2 : (i / (x.length - 1)) * plotW);
  const yAt = (v: number) => PAD.top + plotH - ((v - lo) / span) * plotH;

  /** Split into contiguous runs of non-null points, so abstained windows leave a visible gap. */
  const segments = (points: Array<number | null>): string[] => {
    const paths: string[] = [];
    let current: string[] = [];
    points.forEach((v, i) => {
      if (v == null) {
        if (current.length > 1) paths.push(current.join(" "));
        current = [];
        return;
      }
      current.push(`${current.length === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`);
    });
    if (current.length > 1) paths.push(current.join(" "));
    return paths;
  };

  const ticks = [lo, lo + span / 2, hi];

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={yLabel ? `${yLabel} over time` : "time series"}
      >
        {ticks.map((t, i) => (
          <g key={i}>
            <line className="grid-line" x1={PAD.left} x2={WIDTH - PAD.right} y1={yAt(t)} y2={yAt(t)} />
            <text className="axis-label" x={PAD.left - 6} y={yAt(t) + 3} textAnchor="end">
              {Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(1)}
            </text>
          </g>
        ))}

        {series.map((s) =>
          segments(s.points).map((d, i) => (
            <path key={`${s.label}-${i}`} d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
          ))
        )}

        <text className="axis-label" x={PAD.left} y={height - 6}>
          {(x[0] / 60).toFixed(0)} min
        </text>
        <text className="axis-label" x={WIDTH - PAD.right} y={height - 6} textAnchor="end">
          {(x[x.length - 1] / 60).toFixed(0)} min
        </text>
      </svg>

      <div className="legend">
        {series.map((s) => (
          <span className="key" key={s.label}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
