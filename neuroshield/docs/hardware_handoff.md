# Hardware handoff contract

This is the one document a firmware builder needs to read before writing a line of embedded
code. Hardware acceptance begins the moment a real device emits `neuroshield.hw.v1` events that
the existing replay/serial adapter accepts -- nothing else in this codebase (feature pipeline,
model, backend, dashboard) needs to change for that to happen. If you find yourself wanting to
change anything outside this contract to make hardware work, stop and re-read this document
first.

## The one job

Emit newline-delimited JSON events, one per sample or health update, over a serial connection,
matching the schema below exactly. That's it. You do not need to understand model training,
feature extraction, the dashboard, or the backend to do this job.

## Raw event schema (`neuroshield.hw.v1`)

Full definition, with worked examples for every event type, lives in **`docs/contracts.md`** --
read that first. Summary of the required common fields on every line:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Must be exactly `"neuroshield.hw.v1"`. |
| `type` | string | One of `ppg`, `eda`, `temp`, `imu`, `health`. |
| `source` | string | Use `"serial"` for a real device. |
| `session_id` | string | Stable for the lifetime of one connection. |
| `seq` | integer | Monotonic counter from 0, shared across all event types on this connection. |
| `t_us` | integer | Microseconds since session start (device clock), monotonic non-decreasing. |
| `ok` | boolean | `false` if this sample is a known fault; include an `error` code and omit the payload. |

## Sample-rate targets and units

| Type | Target rate | Payload | Units |
|---|---|---|---|
| `ppg` | 64 Hz | `ppg_raw` (int) | Raw ADC counts (sensor-specific scale; relative use only). |
| `eda` | 4 Hz | `eda_level` (float), `eda_unit` (string) | `"uS"` if calibrated, else `"relative"` -- declare it on every event. |
| `temp` | 4 Hz | `temp_c` (float) | Degrees Celsius. |
| `imu` | 32 Hz | `acc_x`, `acc_y`, `acc_z` (float) | m/s². Gyro fields optional. |
| `health` | ~1 Hz | see below | -- |

These rates match WESAD's wrist Empatica E4 device, which the entire feature pipeline
(`features-v1`, T6) and the frozen M1 model (T9) were built and windowed against. Deviating from
these rates does not break anything outright (the feature extractor windows by elapsed time, not
sample count), but large deviations should be flagged and discussed, since motion/quality
thresholds (T13) were calibrated assuming roughly this sampling density.

## `health` event requirements

The device must emit a `health` event roughly once per second (or on any state change), with:

```json
{"schema_version":"neuroshield.hw.v1","type":"health","source":"serial","session_id":"...",
 "seq":N,"t_us":T,"battery_pct":91.0,
 "channels":{"ppg":true,"eda":true,"temp":true,"imu":true},
 "link_quality":-71,"uptime_s":128.4,"fault":null,"ok":true}
```

- `battery_pct`: 0-100, or `null` if not measurable.
- `channels`: per-channel boolean, whether that channel is currently producing valid samples.
- `link_quality`: signal strength / link quality indicator (e.g. an RSSI-like value), or `null`.
- `fault`: a short machine-readable code when something is wrong (e.g. `"low_battery"`,
  `"sensor_disconnected"`), else `null`.

## Error handling

When a channel fails for one or more samples, emit those samples with `ok:false`, no payload
fields, and an `error` code (e.g. `"sensor_disconnected"`, `"adc_saturation"`,
`"checksum_fail"`). **Do not simply stop sending events for that channel** -- a gap with no
events looks identical to "nothing is wrong, there's just no new data," whereas an explicit
`ok:false` event lets the software-side quality gate (T13) mark that window `poor_signal`
immediately instead of guessing. `seq` must keep incrementing through faulted samples; do not
reuse or skip sequence numbers.

## Serial link settings

The MVP replay/serial adapter (`src/neuroshield/runtime/replay_source.py`) expects each physical
line of the connection to be exactly one JSON object terminated by `\n`, UTF-8 encoded. Suggested
defaults for a UART/USB-serial link, to be confirmed against the actual hardware once selected:

- Baud rate: 115200 (8 data bits, no parity, 1 stop bit -- "8N1").
- Framing: one event per line, `\n`-terminated, no other framing/checksum wrapper needed (the
  event's own `ok`/`error` fields carry fault information; a transport-level checksum is optional
  and, if used, must not change the JSON payload itself).
- Flow control: none required at these rates; hardware flow control (RTS/CTS) is acceptable if
  the platform provides it.
- No binary framing, no length-prefixing -- keep it newline-delimited JSON so the exact same
  parser (`ReplaySource` / the Pydantic event models in `replay_source.py`) can read a live serial
  stream or a recorded file identically.

A `serial_source.py` module (not yet built, since no hardware is connected) will wrap a
`pyserial.Serial` object's line-oriented read loop and feed each line through the exact same
`_parse_and_validate` function `ReplaySource` already uses -- same Pydantic models, same
`ReplayCounters`, same raw error log. That is the whole integration surface.

## Sample replay file and expected parser counters

The committed fixture at `data/fixtures/calm_motion_stress.ndjson` (generated by T10;
`schema_version=neuroshield.hw.v1`, `session_id=demo-001`, 600 seconds, seed 7) is the reference
example. Running it through the parser:

```bash
uv run python3 -c "
from neuroshield.runtime.replay_source import ReplaySource
source = ReplaySource('data/fixtures/calm_motion_stress.ndjson', speed=None)
list(source)
print(source.counters.as_dict())
"
```

produces:

```json
{"valid_events": 63000, "invalid_events": 0, "missing_fields": 0, "unknown_schema_versions": 0, "stale_periods": 0}
```

A real device's first live connection should be checked against these same counters: valid
events should track total samples sent 1:1, and `invalid_events` /
`missing_fields` / `unknown_schema_versions` should be zero. Any non-zero `stale_periods` means a
gap larger than the configured threshold (default 2 seconds) occurred between valid events --
worth investigating on a real link, since it may indicate a dropped connection rather than a
quiet period.

## When hardware acceptance begins

Hardware acceptance begins the moment a device, connected over serial, produces a live event
stream that:

1. Passes through the existing replay/serial adapter with `invalid_events == 0` and
   `unknown_schema_versions == 0` over a sustained session.
2. Flows through the unchanged software pipeline -- `events_to_bundle` → `extract_features` →
   baseline z-scoring → quality gate → M1 → status state machine → explanations → API → dashboard
   -- and produces a sensible live status.

At that point hardware has done its one job. Everything downstream was already built and tested
against the software replay gate (T19) and does not need to change.
