"use client";

// The empty state IS the onboarding. Before this, the page opened with four explainer cards and a
// settings panel, and the only actionable control was buried in the middle of them -- so a first-time
// visitor had to read a paragraph to discover there was a button, and scroll to find out it had done
// anything. One screen, one sentence, one button.

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Play } from "lucide-react";

export function GetStarted({ onRun, busy }: { onRun: () => void; busy: boolean }) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center gap-5 px-6 py-14 text-center">
        <div className="max-w-lg space-y-2">
          <h2 className="text-xl font-semibold tracking-tight">Watch a stress reading, live</h2>
          <p className="text-sm text-muted-foreground">
            No wristband needed. We&apos;ll simulate someone wearing one for ten minutes — resting, then
            under stress, then moving around — and read their body one minute at a time, exactly as the
            real device would.
          </p>
        </div>

        <Button size="lg" onClick={onRun} disabled={busy} className="gap-2">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {busy ? "Starting…" : "Start the demo"}
        </Button>

        <p className="text-xs text-muted-foreground">
          Takes about a minute. It begins by sitting quiet to learn their baseline.
        </p>
      </CardContent>
    </Card>
  );
}
