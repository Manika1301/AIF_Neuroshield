"use client";

// One button, not three. The old flow exposed the backend's two-step lifecycle (start a session,
// then calibrate) and left the user to guess the order -- and to guess what "quiet seconds" meant.
// Both steps run from one click, and the advanced knobs are collapsed behind a toggle.

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2, Play, Settings2 } from "lucide-react";

export interface SessionConfig {
  sourceMode: string;
  replayPath: string;
  durationSec: number;
  seed: number;
  speed: number;
  quietSeconds: number;
}

export const DEFAULT_CONFIG: SessionConfig = {
  sourceMode: "synthetic",
  replayPath: "data/fixtures/calm_motion_stress.ndjson",
  durationSec: 600,
  seed: 0,
  speed: 10,
  quietSeconds: 150,
};

interface Props {
  config: SessionConfig;
  onChange: (config: SessionConfig) => void;
  onRun: () => void;
  busy: boolean;
  running: boolean;
}

export function SessionSetup({ config, onChange, onRun, busy, running }: Props) {
  const [advanced, setAdvanced] = useState(false);
  const set = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) =>
    onChange({ ...config, [key]: value });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Run a session</CardTitle>
        <CardDescription>
          There is no real wristband attached, so pick a signal source. The app first sits quiet to
          learn your baseline, then streams the rest of the recording one minute at a time.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="source">Signal source</Label>
            <Select value={config.sourceMode} onValueChange={(v) => set("sourceMode", v)}>
              <SelectTrigger id="source" className="w-[220px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="synthetic">Simulated wearer</SelectItem>
                <SelectItem value="replay">Recorded file</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="speed">Playback speed</Label>
            <Select value={String(config.speed)} onValueChange={(v) => set("speed", Number(v))}>
              <SelectTrigger id="speed" className="w-[190px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">Real time (1 min/window)</SelectItem>
                <SelectItem value="10">10× faster</SelectItem>
                <SelectItem value="30">30× faster</SelectItem>
                <SelectItem value="0">Instant</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button onClick={onRun} disabled={busy} className="gap-2">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {busy ? "Starting…" : running ? "Restart session" : "Start session"}
          </Button>

          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-muted-foreground"
            onClick={() => setAdvanced((v) => !v)}
          >
            <Settings2 className="h-3.5 w-3.5" />
            {advanced ? "Hide" : "Advanced"}
          </Button>
        </div>

        {advanced && (
          <div className="grid gap-4 rounded-md border bg-muted/30 p-4 sm:grid-cols-2 lg:grid-cols-3">
            {config.sourceMode === "replay" ? (
              <div className="space-y-1.5 sm:col-span-2 lg:col-span-3">
                <Label htmlFor="path">Recording path</Label>
                <Input
                  id="path"
                  value={config.replayPath}
                  onChange={(e) => set("replayPath", e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Path on the backend, relative to the repo root.
                </p>
              </div>
            ) : (
              <>
                <div className="space-y-1.5">
                  <Label htmlFor="duration">Recording length (seconds)</Label>
                  <Input
                    id="duration"
                    type="number"
                    value={config.durationSec}
                    onChange={(e) => set("durationSec", Number(e.target.value))}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="seed">Random seed</Label>
                  <Input
                    id="seed"
                    type="number"
                    value={config.seed}
                    onChange={(e) => set("seed", Number(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">Same seed, same simulated wearer.</p>
                </div>
              </>
            )}

            <div className="space-y-1.5">
              <Label htmlFor="quiet">Baseline calibration (seconds)</Label>
              <Input
                id="quiet"
                type="number"
                value={config.quietSeconds}
                onChange={(e) => set("quietSeconds", Number(e.target.value))}
              />
              <p className="text-xs text-muted-foreground">
                How much quiet data defines &quot;normal for you&quot;. Everything is judged against it.
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
