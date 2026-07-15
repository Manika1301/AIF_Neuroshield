"use client";

// Once a session is running, the controls shrink to a single strip and the screen belongs to the
// data. Crucially it shows PROGRESS: windows arrive every few seconds, and without a visible
// counter a user who clicked "start" has no way to tell the difference between "it's working" and
// "it's broken" -- which is exactly how a working app gets reported as dead.

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CheckCircle2, Loader2, RotateCcw } from "lucide-react";

interface Props {
  nWindows: number;
  complete: boolean;
  busy: boolean;
  speed: number;
  onSpeedChange: (speed: number) => void;
  onRestart: () => void;
}

// The synthetic demo is 600s at a 30s step, so ~19 windows. Used only to draw a progress bar; the
// real end of the session is the backend's session_complete message, never this estimate.
const EXPECTED_WINDOWS = 19;

export function SessionBar({ nWindows, complete, busy, speed, onSpeedChange, onRestart }: Props) {
  const pct = complete ? 100 : Math.min(100, (nWindows / EXPECTED_WINDOWS) * 100);

  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-x-6 gap-y-3 py-4">
        <div className="flex min-w-[220px] flex-1 flex-col gap-1.5">
          <div className="flex items-center gap-2 text-sm">
            {complete ? (
              <CheckCircle2 className="h-4 w-4 text-calm" />
            ) : (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
            <span className="font-medium">
              {complete ? "Session complete" : `Reading… ${nWindows} ${nWindows === 1 ? "minute" : "minutes"} in`}
            </span>
            <span className="text-muted-foreground">
              {complete ? `${nWindows} readings` : "one reading per minute"}
            </span>
          </div>
          <Progress value={pct} className="h-1.5" indicatorClassName={complete ? "bg-calm" : undefined} />
        </div>

        <div className="flex items-center gap-2">
          <Select value={String(speed)} onValueChange={(v) => onSpeedChange(Number(v))}>
            <SelectTrigger className="h-9 w-[150px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1">Real time</SelectItem>
              <SelectItem value="10">10× faster</SelectItem>
              <SelectItem value="30">30× faster</SelectItem>
              <SelectItem value="0">Instant</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" size="sm" onClick={onRestart} disabled={busy} className="gap-1.5">
            <RotateCcw className="h-3.5 w-3.5" />
            Restart
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
