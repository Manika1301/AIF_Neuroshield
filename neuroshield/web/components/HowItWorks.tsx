"use client";

// A dashboard that shows a number without saying where it came from is asking to be misread.
// This is the one-screen explanation of the whole product, kept above the data.

import { Card, CardContent } from "@/components/ui/card";
import { Fingerprint, Radio, ShieldQuestion, Gauge } from "lucide-react";

const STEPS = [
  {
    icon: Radio,
    title: "1. Read the wrist",
    body: "Four sensors: pulse, sweat response, skin temperature, and movement. One reading every 60 seconds.",
  },
  {
    icon: Fingerprint,
    title: "2. Compare to you",
    body: "A few quiet minutes set your personal baseline. Two people can both be calm at 58 and 82 bpm, so absolute numbers alone mean little.",
  },
  {
    icon: Gauge,
    title: "3. Score the window",
    body: "A model trained on 15 people under a lab stress protocol turns those signals into a 0–100 index and a calm / elevated / high level.",
  },
  {
    icon: ShieldQuestion,
    title: "4. Refuse when unsure",
    body: "If you're moving or the sensor has poor contact, the reading isn't trustworthy — so it reports nothing instead of guessing.",
  },
];

export function HowItWorks() {
  return (
    <Card className="bg-muted/30">
      <CardContent className="grid gap-5 py-5 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map(({ icon: Icon, title, body }) => (
          <div key={title} className="flex gap-3">
            <Icon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div>
              <p className="text-sm font-medium">{title}</p>
              <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{body}</p>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
