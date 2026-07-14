"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { StatusRecord } from "@/lib/api";
import { TimeSeries } from "./TimeSeries";

export function TrendsView({ records }: { records: StatusRecord[] }) {
  if (records.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-muted-foreground">
          Nothing recorded yet. Start a session to see how the reading moves over time.
        </CardContent>
      </Card>
    );
  }

  const x = records.map((r) => r.window_start_s ?? 0);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Stress index over the session</CardTitle>
          <CardDescription>
            Gaps in the line are windows the model refused to score. Drawing through them would invent
            data it deliberately declined to produce.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <TimeSeries
            x={x}
            yMin={0}
            yMax={100}
            marker={{ value: 45, label: "elevated" }}
            series={[
              {
                label: "Stress index",
                color: "hsl(var(--chart-1))",
                points: records.map((r) => r.stress_index),
              },
            ]}
          />
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Heart rate</CardTitle>
            <CardDescription>Beats per minute, as measured at the wrist.</CardDescription>
          </CardHeader>
          <CardContent>
            <TimeSeries
              x={x}
              height={170}
              series={[
                {
                  label: "bpm",
                  color: "hsl(var(--chart-2))",
                  points: records.map((r) => r.values.hr_mean_bpm ?? null),
                },
              ]}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sweat response</CardTitle>
            <CardDescription>Skin conductance (µS). Rises with arousal of any kind.</CardDescription>
          </CardHeader>
          <CardContent>
            <TimeSeries
              x={x}
              height={170}
              series={[
                {
                  label: "µS",
                  color: "hsl(var(--chart-3))",
                  points: records.map((r) => r.values.eda_level ?? null),
                },
              ]}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
