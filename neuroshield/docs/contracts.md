# Raw event contract: `neuroshield.hw.v1`

This document defines the raw event stream format shared by the synthetic generator, the replay
player, and any future hardware firmware. It is written first, before any streaming code, so that
nothing downstream has to guess field names or units. If a piece of code (generator, parser,
firmware) produces events that satisfy this document, it is a valid NeuroShield raw source.

## Wire format

- The stream is **newline-delimited JSON (NDJSON)**: exactly one JSON object per line, one line
  per sample or per health update.
- Each line is a complete, self-contained JSON object. No multi-line objects, no trailing commas,
  no comments.
- Encoding is UTF-8. Lines are terminated with `\n`.
- Consumers must not assume a fixed interleaving of event types — `ppg`, `eda`, `temp`, `imu`, and
  `health` events arrive on the same stream, interleaved, each at its own rate.
- Unknown extra fields on an otherwise-valid event must be **ignored, not rejected** (forward
  compatibility). Missing required fields, wrong field types, or a line that does not parse as
  JSON are all invalid and must be preserved in a raw error log rather than silently dropped
  (see T11).

## Common fields (present on every event)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema_version` | string | yes | Contract version, currently `"neuroshield.hw.v1"`. A parser must reject (not guess-fix) any other value it does not explicitly support. |
| `type` | string | yes | One of `"ppg"`, `"eda"`, `"temp"`, `"imu"`, `"health"`. |
| `source` | string | yes | Origin of the stream: `"synthetic"`, `"replay"`, `"serial"`, or a dataset name when replaying a converted dataset fixture (e.g. `"wesad"`). |
| `session_id` | string | yes | Identifier for one continuous recording/streaming session. Stable for the lifetime of the connection. |
| `seq` | integer | yes | Monotonically increasing counter, starting at `0`, shared across **all** event types from this source in this session. Used to detect dropped or reordered events — a gap in `seq` means lost events. |
| `t_us` | integer | yes | Microseconds elapsed since the start of the session (`t_us = 0` at the first event). This is a logical session clock, not a wall-clock timestamp. It must be monotonic non-decreasing across all events from the same source. Mapping session start to wall-clock time, if needed, is the responsibility of the backend/replay layer, not the event itself. |
| `ok` | boolean | yes | `true` if this event's payload is a valid reading. `false` if the sensor/source detected a fault for this sample; in that case type-specific payload fields may be `null` and an `error` field (string, short machine-readable code, e.g. `"adc_saturation"`, `"sensor_disconnected"`, `"checksum_fail"`) should be present. |

## Event types and units

| Type | Target rate | Payload fields | Units |
|---|---|---|---|
| `ppg` | 64 Hz | `ppg_raw` (int) | Raw ADC counts. Sensor/ADC-specific scale — consumers must not assume an absolute physical scale, only use it for relative/derived features (pulse detection, HRV). |
| `eda` | 4 Hz | `eda_level` (float), `eda_unit` (string) | `eda_unit` is `"uS"` (microsiemens) when the source supports calibrated skin-conductance output, or `"relative"` for an uncalibrated 0–1 relative level. Every `eda` event must declare its own `eda_unit`; consumers must not assume a fixed unit for a whole session. |
| `temp` | 4 Hz | `temp_c` (float) | Degrees Celsius. |
| `imu` | 32 Hz | `acc_x`, `acc_y`, `acc_z` (float); optionally `gyro_x`, `gyro_y`, `gyro_z` (float) | Acceleration in m/s². Gyroscope fields are optional and may be omitted by sources (including the MVP synthetic/replay sources) that only provide acceleration. |
| `health` | ~1 Hz (or on change) | `battery_pct` (float 0-100, nullable), `channels` (object mapping `"ppg"`/`"eda"`/`"temp"`/`"imu"` to bool), `link_quality` (int, nullable, e.g. RSSI or serial link score), `uptime_s` (float), `fault` (string, nullable) | `health` is a status update about the source itself, not a physiological sample. `channels` reports whether each channel is currently producing valid data. `fault`, when non-null, is a short machine-readable code, e.g. `"low_battery"`, `"sensor_disconnected"`, `"clock_drift"`. |

Sample-rate targets above match the wrist Empatica E4 rates used for WESAD (T4/T6): BVP/PPG 64 Hz,
EDA/TEMP 4 Hz, ACC 32 Hz. The synthetic generator, replay player, and future firmware should all
target these rates so the same feature-extraction window logic works unchanged across sources.

## Versioning and compatibility

- `schema_version` follows `neuroshield.hw.vN`. The MVP only accepts exact matches to
  `"neuroshield.hw.v1"`.
- Any consumer (replay parser, backend ingestion, feature pipeline) that receives an event with an
  unrecognized `schema_version` must reject it with a typed, visible error rather than attempting
  to interpret it. This applies to the whole session: a stream should declare one schema version
  and not change it mid-session.
- Adding new optional fields to an event type is backward compatible under `v1` (see "unknown
  extra fields" above). Changing the meaning or unit of an existing field, or adding a new
  required field, requires a new `schema_version` (`v2`).

## Examples

`ppg`
```json
{"schema_version":"neuroshield.hw.v1","type":"ppg","source":"synthetic","session_id":"demo-001","seq":100,"t_us":4484000,"ppg_raw":2148,"ok":true}
```

`eda` (relative unit, uncalibrated source)
```json
{"schema_version":"neuroshield.hw.v1","type":"eda","source":"synthetic","session_id":"demo-001","seq":144,"t_us":4500000,"eda_level":0.42,"eda_unit":"relative","ok":true}
```

`eda` (calibrated microsiemens source)
```json
{"schema_version":"neuroshield.hw.v1","type":"eda","source":"replay","session_id":"wesad-s2","seq":9821,"t_us":612000000,"eda_level":3.87,"eda_unit":"uS","ok":true}
```

`temp`
```json
{"schema_version":"neuroshield.hw.v1","type":"temp","source":"synthetic","session_id":"demo-001","seq":145,"t_us":4500000,"temp_c":33.12,"ok":true}
```

`imu`
```json
{"schema_version":"neuroshield.hw.v1","type":"imu","source":"synthetic","session_id":"demo-001","seq":260,"t_us":4531250,"acc_x":0.02,"acc_y":9.79,"acc_z":-0.15,"ok":true}
```

`health` (nominal)
```json
{"schema_version":"neuroshield.hw.v1","type":"health","source":"synthetic","session_id":"demo-001","seq":300,"t_us":5000000,"battery_pct":91.0,"channels":{"ppg":true,"eda":true,"temp":true,"imu":true},"link_quality":null,"uptime_s":5.0,"fault":null,"ok":true}
```

`health` (fault) and a faulted `ppg` sample immediately after, showing `seq` continuing without a
gap even though `ok` is `false`:
```json
{"schema_version":"neuroshield.hw.v1","type":"health","source":"serial","session_id":"bench-003","seq":511,"t_us":9120000,"battery_pct":12.5,"channels":{"ppg":true,"eda":false,"temp":true,"imu":true},"link_quality":-71,"uptime_s":128.4,"fault":"low_battery","ok":true}
{"schema_version":"neuroshield.hw.v1","type":"eda","source":"serial","session_id":"bench-003","seq":512,"t_us":9120250,"eda_level":null,"eda_unit":"uS","ok":false,"error":"sensor_disconnected"}
```

## What implementations must not do

- Must not invent additional required common fields beyond the table above.
- Must not silently reinterpret `eda_level` units — always read `eda_unit` per event.
- Must not treat a missing/invalid line as "no event"; it must be counted and logged (see T11
  parser counters: valid events, invalid events, missing fields, unknown schema versions, stale
  periods).
- Must not fabricate `t_us` gaps to smooth over dropped samples; gaps are meaningful signal for
  the quality/abstention layer (T13).
