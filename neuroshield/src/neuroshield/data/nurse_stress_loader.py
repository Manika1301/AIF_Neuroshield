"""Loader for the Nurse Stress Dataset (naturalistic real-world validation).

Same Empatica E4 raw export format as Stress-Predict (BVP 64Hz, EDA/TEMP 4Hz, ACC 32Hz, ACC in
1/64 g), but packaged very differently:

  Stress_dataset.zip
    <ID>/<ID>_<unix_start>.zip   one nested zip per recording session
      ACC.csv BVP.csv EDA.csv TEMP.csv IBI.csv HR.csv info.txt tags.csv

Each nurse has ~40 disjoint sessions (separate recording bouts, not one continuous stream), so a
*session* is the natural unit here, not a *subject*. Sessions are read directly out of the nested
zips in memory (``zipfile`` inside ``zipfile``) -- the ~620 sessions are never extracted to disk.

Ground truth is a sparse self-report event log (``SurveyResults.xlsx``), not a continuous label:
a nurse logged specific stress episodes with a start/end time and a 0/1/2 level. Only samples that
fall inside a reported event get a label; everything else is ``UNLABELED`` (-1) and must NOT be
assumed calm -- an unlogged period simply means nothing was reported, not that nothing happened.

Survey times are local America/New_York (verified by cross-referencing session UTC start times
against reported event times); they are converted to UTC before alignment. Level 2 -> stress (1),
level 0 -> baseline (0); levels 1 ("moderate", ambiguous) and "na" (unrated) are excluded.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("data/doi_10_5061_dryad_5hqbzkh6f__v20210917")
STRESS_ZIP_NAME = "Stress_dataset.zip"
SURVEY_XLSX_NAME = "SurveyResults.xlsx"
SURVEY_TIMEZONE = "America/New_York"

UNLABELED = -1
BASELINE_LABEL = 0
STRESS_LABEL = 1

# survey "Stress level" -> binary training/eval label; 1 and "na" deliberately absent (excluded)
SURVEY_LEVEL_TO_BINARY = {0: BASELINE_LABEL, 2: STRESS_LABEL}

_SESSION_NAME_RE = re.compile(r"^(?P<pid>[0-9A-Za-z]+)/(?P=pid)_(?P<ts>\d+)\.zip$")

CONTEXT_COLUMNS = [
    "COVID related",
    "Treating a covid patient",
    "Patient in Crisis",
    "Patient or patient's family",
    "Doctors or colleagues",
    "Increased Workload",
    "Technology related stress",
    "Lack of supplies",
    "Documentation",
    "Competency related stress",
    "Saftey (physical or physiological threats)",
    "Work Environment - Physical or others: work processes or procedures",
]


@dataclass
class NurseStressSessionRaw:
    participant_id: str
    session_id: str  # "<ID>_<unix_start>"
    bvp: np.ndarray
    eda: np.ndarray
    temp: np.ndarray
    acc: np.ndarray
    labels: dict[str, np.ndarray]
    sample_rates_hz: dict[str, float]


def _read_e4_single_column(fileobj) -> tuple[float, float, np.ndarray]:
    """Parse an E4 single-signal CSV from a binary file object (line1=start, line2=rate, then data)."""
    text = io.TextIOWrapper(fileobj, encoding="utf-8")
    start_time = float(text.readline().strip())
    rate = float(text.readline().strip())
    values = np.array([float(line) for line in text if line.strip()], dtype=np.float64)
    return start_time, rate, values


def _read_e4_acc(fileobj) -> tuple[float, float, np.ndarray]:
    text = io.TextIOWrapper(fileobj, encoding="utf-8")
    start_time = float(text.readline().split(",")[0].strip())
    rate = float(text.readline().split(",")[0].strip())
    rows = [[float(v) for v in line.split(",")] for line in text if line.strip()]
    return start_time, rate, np.array(rows, dtype=np.float64)


def list_participants(root: Path = DEFAULT_ROOT) -> list[str]:
    with zipfile.ZipFile(root / STRESS_ZIP_NAME) as zf:
        pids = set()
        for name in zf.namelist():
            m = _SESSION_NAME_RE.match(name)
            if m:
                pids.add(m.group("pid"))
    return sorted(pids)


def list_participant_sessions(participant_id: str, root: Path = DEFAULT_ROOT) -> list[str]:
    """Return session ids (``<ID>_<unix_start>``) for one participant, sorted by start time."""
    with zipfile.ZipFile(root / STRESS_ZIP_NAME) as zf:
        sessions = []
        for name in zf.namelist():
            m = _SESSION_NAME_RE.match(name)
            if m and m.group("pid") == participant_id:
                sessions.append((int(m.group("ts")), f"{participant_id}_{m.group('ts')}"))
    return [sid for _, sid in sorted(sessions)]


def sessions_covering_labeled_events(events: pd.DataFrame, root: Path = DEFAULT_ROOT) -> list[str]:
    """Session ids that were recording when a labelled event started.

    Maps each labelled survey event to the participant's session whose start time is the latest one
    at or before the event start -- i.e. the recording bout active at that moment. Lets callers
    extract only the ~150 sessions that can contain labelled windows instead of all ~620.
    """
    sessions_by_pid = {
        pid: sorted((int(sid.split("_")[1]), sid) for sid in list_participant_sessions(pid, root))
        for pid in list_participants(root)
    }
    result = set()
    labeled = events[events["binary_label"].notna()]
    for row in labeled.itertuples():
        pid = row.participant_id
        if pid not in sessions_by_pid or pd.isna(row.start_utc):
            continue
        t = row.start_utc.timestamp()
        candidates = [sid for (st, sid) in sessions_by_pid[pid] if st <= t]
        if candidates:
            result.add(candidates[-1])
    return sorted(result)


def load_survey_events(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Load the self-report log with real UTC start/end timestamps and a binary label column.

    Returns one row per reported event with columns: participant_id, start_utc, end_utc,
    stress_level (raw 0/1/2/'na'), binary_label (0/1 or NaN when excluded), plus the raw context
    trigger columns. Rows keep their original level so Tier-4 analytics can use all of them.
    """
    df = pd.read_excel(root / SURVEY_XLSX_NAME)
    df["participant_id"] = df["ID"].astype(str)
    date = pd.to_datetime(df["date"]).dt.normalize()

    def _combine(col: str) -> pd.Series:
        times = pd.to_timedelta(df[col].astype(str))
        naive = date + times
        return naive.dt.tz_localize(SURVEY_TIMEZONE, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")

    df["start_utc"] = _combine("Start time")
    df["end_utc"] = _combine("End time")
    # end before start would mean the event crossed midnight; nudge end forward a day.
    crossed = df["end_utc"].notna() & df["start_utc"].notna() & (df["end_utc"] < df["start_utc"])
    df.loc[crossed, "end_utc"] = df.loc[crossed, "end_utc"] + pd.Timedelta(days=1)

    df["stress_level"] = df["Stress level"]
    df["binary_label"] = df["stress_level"].map(
        lambda v: SURVEY_LEVEL_TO_BINARY.get(v, np.nan) if not isinstance(v, str) else np.nan
    )

    keep = ["participant_id", "start_utc", "end_utc", "stress_level", "binary_label", *[
        c for c in CONTEXT_COLUMNS if c in df.columns
    ]]
    return df[keep].copy()


def label_session_samples(
    sample_unix_times: np.ndarray, participant_id: str, events: pd.DataFrame
) -> np.ndarray:
    """Label each sample by whether its unix time falls inside a reported binary event interval."""
    labels = np.full(len(sample_unix_times), UNLABELED, dtype=np.int64)
    subset = events[(events["participant_id"] == participant_id) & events["binary_label"].notna()]
    if subset.empty:
        return labels
    for row in subset.itertuples():
        if pd.isna(row.start_utc) or pd.isna(row.end_utc):
            continue
        start = row.start_utc.timestamp()
        end = row.end_utc.timestamp()
        inside = (sample_unix_times >= start) & (sample_unix_times <= end)
        labels[inside] = int(row.binary_label)
    return labels


def load_nurse_stress_session(
    session_id: str, events: pd.DataFrame | None = None, root: Path = DEFAULT_ROOT
) -> NurseStressSessionRaw:
    """Load one session's wrist signals and per-sample labels.

    ``session_id`` is ``"<ID>_<unix_start>"``. ``events`` may be passed in to avoid re-reading the
    survey xlsx for every session (recommended when looping); if None, it is loaded once here.
    """
    participant_id = session_id.split("_")[0]
    inner_name = f"{participant_id}/{session_id}.zip"

    with zipfile.ZipFile(root / STRESS_ZIP_NAME) as outer:
        if inner_name not in outer.namelist():
            raise FileNotFoundError(f"session {inner_name!r} not found in {root / STRESS_ZIP_NAME}")
        inner_bytes = outer.read(inner_name)

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as session_zf:
        with session_zf.open("BVP.csv") as f:
            bvp_start, bvp_rate, bvp = _read_e4_single_column(f)
        with session_zf.open("EDA.csv") as f:
            eda_start, eda_rate, eda = _read_e4_single_column(f)
        with session_zf.open("TEMP.csv") as f:
            temp_start, temp_rate, temp = _read_e4_single_column(f)
        with session_zf.open("ACC.csv") as f:
            acc_start, acc_rate, acc = _read_e4_acc(f)

    if events is None:
        events = load_survey_events(root)

    labels = {
        "BVP": label_session_samples(bvp_start + np.arange(len(bvp)) / bvp_rate, participant_id, events),
        "EDA": label_session_samples(eda_start + np.arange(len(eda)) / eda_rate, participant_id, events),
        "TEMP": label_session_samples(temp_start + np.arange(len(temp)) / temp_rate, participant_id, events),
        "ACC": label_session_samples(acc_start + np.arange(len(acc)) / acc_rate, participant_id, events),
    }

    return NurseStressSessionRaw(
        participant_id=participant_id,
        session_id=session_id,
        bvp=bvp,
        eda=eda,
        temp=temp,
        acc=acc,
        labels=labels,
        sample_rates_hz={"BVP": bvp_rate, "EDA": eda_rate, "TEMP": temp_rate, "ACC": acc_rate},
    )
