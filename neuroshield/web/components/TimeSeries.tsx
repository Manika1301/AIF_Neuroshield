"use client";

// A minimal inline-SVG line chart. Deliberately dependency-free: a charting library would be the
// heaviest thing in the app, for four series.
//
// Gaps are the point. A window the model refused to score (motion / poor signal) has no value, and
// drawing a line straight through it would invent data the model explicitly declined to produce.
// Null runs are therefore breaks in the line, never interpolations.

export interface Series {
  label: string;
  /** A CSS color, usually hsl(var(--chart-N)). */
  color: string;
  points: Array<number | null>;
}

interface Props {
  x: number[]; // window start, in seconds
  series: Series[];
  height?: number;
  yMin?: number;
  yMax?: number;
  /** Drawn as a dashed horizontal rule, e.g. the "elevated" threshold. */
  marker?: { value: number; label: string };
}

const PAD = { top: 12, right: 14, bottom: 26, left: 42 };
const WIDTH = 760;

export function TimeSeries({ x, series, height = 200, yMin, yMax, marker }: Props) {
  const values = series.flatMap((s) => s.points).filter((v): v is number => v != null);
  if (x.length === 0 || values.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">Nothing to plot yet.</p>;
  }

  const lo = yMin ?? Math.min(...values);
  const hi = yMax ?? Math.max(...values);
  const span = hi - lo || 1;

  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  const xAt = (i: number) => PAD.left + (x.length === 1 ? plotW / 2 : (i / (x.length - 1)) * plotW);
  const yAt = (v: number) => PAD.top + plotH - ((v - lo) / span) * plotH;

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
        viewBox={`0 0 ${WIDTH} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full"
        role="img"
        aria-label={series.map((s) => s.label).join(", ") + " over time"}
      >
        {ticks.map((t, i) => (
          <g key={i}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={yAt(t)}
              y2={yAt(t)}
              className="stroke-border"
              strokeWidth={1}
            />
            <text x={PAD.left - 8} y={yAt(t) + 3} textAnchor="end" className="fill-muted-foreground text-[10px]">
              {Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(1)}
            </text>
          </g>
        ))}

        {marker && marker.value > lo && marker.value < hi && (
          <g>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={yAt(marker.value)}
              y2={yAt(marker.value)}
              className="stroke-elevated"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
            <text x={WIDTH - PAD.right} y={yAt(marker.value) - 4} textAnchor="end" className="fill-elevated text-[10px]">
              {marker.label}
            </text>
          </g>
        )}

        {series.map((s) =>
          segments(s.points).map((d, i) => (
            <path
              key={`${s.label}-${i}`}
              d={d}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))
        )}

        <text x={PAD.left} y={height - 6} className="fill-muted-foreground text-[10px]">
          {(x[0] / 60).toFixed(0)} min
        </text>
        <text x={WIDTH - PAD.right} y={height - 6} textAnchor="end" className="fill-muted-foreground text-[10px]">
          {(x[x.length - 1] / 60).toFixed(0)} min
        </text>
      </svg>

      <div className="mt-2 flex flex-wrap gap-4 text-xs text-muted-foreground">
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
