# NeuroShield — Project Report

**A personalized, multi-signal stress-detection system**

---

## Abstract

I built NeuroShield, a system that reads a person's physiological signals and reports how
stressed their body is — not against a population average, but against *their own* calm baseline.
The central observation driving the whole project is that absolute physiological numbers are
misleading: two people can both be perfectly at rest, one at 58 beats per minute and one at 82, so
a stress model fed raw values wastes its capacity learning *who someone is* instead of *how aroused
they are*. My solution calibrates to each individual during a short quiet period and then expresses
every subsequent reading relative to that baseline.

Every 60 seconds the system computes 36 features from four physiological channels — pulse, skin
conductance, skin temperature, and motion — and feeds them to a two-headed machine-learning model.
The first head produces a calibrated 0–100 stress index; the second separates *stress* arousal from
*positive* arousal such as amusement. Under the strictest evaluation available — training on some
people and testing only on people the model has never seen (leave-one-subject-out) — I reached a
**balanced accuracy of 0.919**, which exceeds the best comparable published result for this class of
sensor. Crucially, I also built in restraint: when motion or poor signal quality make a reading
untrustworthy, the model refuses to score rather than guess, and the system is structurally
prevented from making any clinical claim. This report documents the research, problem, objectives,
plan, methodology, datasets, and results behind that system.

---

## Background research

Before writing any code I studied how physiological stress detection actually works and where
existing approaches fail. Three findings shaped everything that followed.

**Stress has a measurable physiological signature.** Psychological stress activates the sympathetic
branch of the autonomic nervous system — the "fight-or-flight" response. This produces a consistent
peripheral pattern: heart rate rises, heart-rate variability falls, sweat-gland activity increases,
and blood is drawn away from the extremities so the skin cools. Each of my four sensors targets one
part of that pattern. I learned that skin conductance is special: eccrine sweat glands are
innervated *only* by the sympathetic nervous system, with no opposing parasympathetic input, which
makes it the single cleanest measure of arousal available.

**The published accuracy numbers are mostly inflated by weak evaluation.** Many papers report
93–99% accuracy, but I found that these almost always use k-fold or intra-subject cross-validation,
which allows the *same person* to appear in both the training and test sets. That inflates the score
because the model can memorise an individual rather than generalise. The honest evaluation is
leave-one-subject-out (LOSO), where the model is tested only on people it never trained on. Under
that discipline, the strongest comparable wrist-only result I found in the literature (Siirtola,
2019) was **0.874**. I adopted LOSO as my only evaluation metric from the start.

**Personalization is the largest lever, and naïve data-pooling backfires.** The wearable-stress
literature (Schmidt et al., WESAD 2018; Gil-Martín et al., 2022) repeatedly identifies
subject-relative normalization as the biggest source of cross-subject gain. I also found that simply
merging multiple datasets to get "more data" *reduces* accuracy without domain adaptation, because
it folds cross-protocol differences into the model. Both findings later proved true in my own
experiments.

---

## The problem

Stress is one of the most common and damaging chronic health burdens, yet the tools for it are
poor. Self-report is unreliable — people are bad at noticing their own stress, and by the time they
consciously register it, they have often been under load for hours. The body, however, reacts
first, and it reacts measurably.

The problem I set out to solve was: **can I turn a stream of low-cost physiological sensor data into
an honest, understandable, real-time estimate of a person's stress — one that respects individual
differences and knows the limits of what it can claim?**

Three sub-problems sat inside that:

1. **Individual variation.** A fixed threshold ("heart rate above 90 = stressed") is meaningless
   across different people. The system has to adapt to each individual.
2. **Trust.** A wearable sensor on a moving wrist produces garbage half the time. A model that
   confidently outputs a number during hand movement is worse than useless — it is misleading.
3. **Over-claiming.** The temptation in a health product is to claim to detect "anxiety" or
   "burnout." The sensors physically cannot support those claims, and making them would be
   dishonest and potentially harmful.

---

## Objectives

I defined the project against these concrete objectives:

1. **Build a complete software-first pipeline** — datasets → features → model → live runtime →
   dashboard — that works end to end from replayed/streamed data, with no hardware dependency, so
   that the hardware could later be swapped in as a data source without changing anything downstream.
2. **Detect stress in a personalized way**, calibrating to each user's own baseline rather than a
   population average.
3. **Achieve accuracy competitive with the literature under honest (LOSO) evaluation** — my target
   was to match or beat the ~0.874 benchmark.
4. **Produce a graded, interpretable output** — a 0–100 index with a plain-language explanation of
   *which* physiological system is driving it — not a bare red/amber/green light.
5. **Build in abstention** — the model must refuse to predict when the signal is untrustworthy.
6. **Enforce honesty structurally** — make it impossible for the system to emit a clinical or
   diagnostic claim.
7. **Deliver an understandable interface** so a non-expert can see what is happening and what it
   does not claim.

---

## Planning the problem

I decomposed the problem into six sequential concerns, each depending on the previous:

- **Data & contracts.** Define a fixed raw-event schema up front so that a synthetic generator,
  replayed dataset, and future firmware all speak the same protocol. Choose datasets.
- **Feature engineering.** Convert raw signals into a fixed, version-pinned feature schema, with a
  quality/coverage measure attached to every window.
- **Modelling.** Choose a model class appropriate to the data scale, train it under LOSO, and
  calibrate its output into a real probability.
- **Runtime.** Personal-baseline calibration, an abstention gate, a state machine with hysteresis,
  and plain-language explanations.
- **Serving.** A backend that streams results window-by-window, and a dashboard that renders them
  live.
- **Validation & honesty.** LOSO evaluation, held-out datasets, a written prediction spec, and a
  hard guard against clinical language.

A key planning decision was to be **software-first**: I would prove the entire pipeline on public
datasets and a synthetic sensor stream *before* touching hardware, so that model, backend, and UI
were all tested against the exact protocol the real device would later use.

---

## Timeline (Gantt chart)

The project ran across roughly ten working weeks. The chart below shows the phases and their
overlap; the bars indicate active weeks.

```
Phase                              W1  W2  W3  W4  W5  W6  W7  W8  W9  W10
--------------------------------------------------------------------------
1. Background research & scoping   ██  ██
2. Data contracts & loaders        ██  ██  ██
3. Feature pipeline (v1)               ██  ██
4. Baseline model (M1) + LOSO              ██  ██
5. Runtime: calibration, gate,                 ██  ██  ██
   state machine, explanations
6. Backend + dashboard (v1)                        ██  ██  ██
7. Multi-dataset harmonization                         ██  ██
   & held-out validation
8. Accuracy improvement                                    ██  ██
   (features-v2, personalization)
9. Real streaming + UI rebuild                                 ██  ██
   (shadcn), comprehension pass
10. Documentation, report,                                        ██  ██
    prediction spec, hardening
--------------------------------------------------------------------------
Milestones:   ▲ M1 (0.831)     ▲ features-v2 (0.880)    ▲ M3 (0.919)
              end of W4          W8                       W8-9
```

The two heaviest phases were 5 (the runtime, where abstention and personalization live) and 8 (the
accuracy work that took the model from 0.831 to 0.919). Phase 7 produced an important negative
result: naïve pooling of datasets, which I abandoned.

---

## Methodology

**Windowing.** I slide a 60-second window over the signals, stepping every 30 seconds. Every window
becomes one fixed-schema feature row. A window that falls during a sensor dropout gets a low
coverage score and is handled by the abstention layer rather than silently dropped.

**Feature extraction.** From the four channels I compute 19 features per window (detailed under
*Inputs* below), using the NeuroKit2 library for pulse-peak detection and electrodermal
decomposition. This includes frequency-domain heart-rate-variability features and a convex-optimisation
decomposition of skin conductance into slow (tonic) and fast (phasic) components.

**Personalization.** This is the core of the method. During the first ~300 seconds of quiet data I
compute each feature's mean and standard deviation for that individual. Every physiological feature
is then expressed *twice*: once in absolute units, and once as a deviation from the person's own
baseline (`(x − personal_mean) / personal_std`). This doubles the feature count to 36 and is what
allows the model to generalise to a person — and a dataset — it has never seen. The reference is
defined in seconds of signal, not a count of windows, so that the calibration the live app performs
exactly matches the reference the model was trained against.

**Model.** I use gradient-boosted decision trees (`HistGradientBoostingClassifier`). I chose this
over deep learning deliberately: with only 15 training subjects and tabular features, tree ensembles
are the correct and best-performing tool at this data scale, they handle missing values natively (a
dropped sensor simply becomes a NaN), and they remain interpretable. The model has two heads:
- **Head A** — binary stress, wrapped in an isotonic `CalibratedClassifierCV` so that its output is
  a genuine probability. This calibration is what makes the 0–100 index meaningful rather than an
  arbitrary score.
- **Head B** — a 4-class affect model (baseline / stress / amusement / meditation) that separates
  stress arousal from positive arousal.

**Evaluation.** Every number I report comes from grouped leave-one-subject-out cross-validation: for
each fold, one subject is held out entirely, the model trains on the rest, and is scored only on
that unseen subject. I use balanced accuracy (which is robust to class imbalance) against a
majority-class dummy baseline of 0.5.

**Abstention.** Before the model runs, a quality gate checks motion and signal coverage. If wrist
motion is too high or coverage/pulse quality too low, the model is not called at all and the window
is reported as `motion_paused` or `poor_signal`. This is a feature, not a limitation.

**Honesty enforcement.** A regression test fails the build if any user-facing string contains
clinical terms (panic, anxiety, burnout, diagnosis, clinical, medical). Over-claiming is made
structurally impossible.

**Serving.** The backend (FastAPI) streams one result per window over a WebSocket as it is computed,
so the dashboard (Next.js + shadcn/ui) updates live, exactly as a real device would deliver data.

---

## Datasets used

| Dataset | Subjects | Role | Notes |
|---|---|---|---|
| **WESAD** | 15 | **Training** | Lab protocol (Trier Social Stress Test). I used only the wearable channels, discarding the cleaner chest sensors on purpose, because the product does not have a chest strap. ~919 labelled windows after quality filtering. |
| **Stress-Predict** | 35 | Held out | Different people, a different lab stress protocol. Used to test generalization; never trained on. |
| **Nurse Stress** | 15 | Held out | Real hospital shifts with sparse self-report labels — the messiest, most realistic data. Never trained on. |

I trained **only on WESAD**. My own experiments confirmed the literature: naïvely pooling WESAD with
Stress-Predict *lowered* LOSO accuracy from 0.83 to 0.62, so I dropped pooling and kept the other
two datasets strictly as external validation. WESAD's labels come from a validated protocol, which
is why it is trustworthy ground truth; there is no equivalent labelled dataset for conditions like
burnout, which is one reason I do not attempt to predict them.

---

## Inputs

Each 60-second window produces **36 model inputs**: 19 absolute features plus 17 personalized (`_p`)
versions of them (the two quality/coverage features are not personalized, as they describe the
recording rather than the person).

**Pulse / PPG (7):** mean heart rate, two measures of heart-rate variability (SDNN-like and RMSSD),
signal quality, and three frequency-domain HRV features (LF power, HF power, and the LF/HF ratio —
the classic sympathovagal-balance index).

**Skin conductance / EDA (7):** tonic level, slope, count and mean amplitude of skin-conductance
responses, and a tonic/phasic decomposition (sustained arousal vs. momentary spikes). *This is the
purest arousal signal in the set.*

**Skin temperature (2):** mean and trend. Under stress the skin gets *cooler* (peripheral
vasoconstriction), so this feature carries a negative sign in the arousal direction.

**Motion / accelerometer (2):** dynamic motion energy (RMS) and its 95th percentile. Doubles as the
abstention trigger.

**Coverage (1):** the fraction of the window with usable samples.

---

## Results

### Accuracy

The final model (`m3_multihead_personalized_v1`), under grouped leave-one-subject-out on WESAD:

| Head | Task | Balanced accuracy | Chance baseline |
|---|---|---|---|
| **A** | Graded stress (binary) | **0.919** | 0.500 |
| **B** | Affect (4-class) | **0.616** | 0.250 |

Head A's macro-F1 is 0.918. For context, the best comparable wrist-only result under the same LOSO
discipline in the literature is 0.874, so my system exceeds it.

### How I got there

Every step below is measured the same way (WESAD LOSO), so they are directly comparable:

| Step | Balanced accuracy |
|---|---|
| Baseline model (13 features, logistic regression) | 0.831 |
| Naïve pooling of WESAD + Stress-Predict *(rejected)* | 0.617 |
| Drop pooling — train on WESAD only | 0.831 |
| + frequency-domain HRV and cvxEDA features | 0.880 |
| + **per-subject baseline personalization** | **0.919** |

The two gains that survived were the richer feature set (+0.049) and personalization (+0.039).

### The honest limits (which I report inside the product, not just here)

- **The error bar is real.** With only 15 subjects, the standard error on the accuracy is
  **±0.036**.
- **It is good on average, not uniformly good.** Per-subject accuracy ranged from **0.50 (a coin
  flip) to 1.00**. I therefore present the index as a personal trend over time, never as a precise
  measurement of a specific person at a specific moment.
- **0.919 is the ceiling for this data.** After reaching it I ran a systematic tuning sweep (window
  size, prediction smoothing, seed ensembling, calibration length, hyperparameters). Nothing beat
  0.919 with statistical significance. In the process I caught two mistakes in my own work — a bug
  where a calibration parameter silently changed with window size, and a case where I had selected
  the best result off the same test data — and I documented both as a negative result. The
  conclusion is that the limit is the dataset (15 people), not the model.

### System results

- The full automated test suite passes (300+ tests); the software acceptance gate passes end to end.
- The backend genuinely streams: I verified 18 windows arriving one at a time over a live WebSocket,
  correctly showing the full arc from calm → high stress → paused-for-motion → poor-signal.
- The dashboard was verified in a real browser: a first-time user sees one clear action, watches the
  reading build live, and the model's refusal states and error bar are shown honestly.

---

## Hardware status (honest position)

The intended device uses a **MAX30102** (optical pulse), a **BNO** IMU (motion), and an **MLX90614**
(infrared skin temperature). This covers three of my four channels. It is currently **missing a
skin-conductance (EDA) sensor**, which accounts for 7 of the 19 features and is — as noted above —
the single cleanest arousal signal. My next hardware step is to add that sensor, and I am quantifying
exactly what it is worth by retraining the model without EDA on the same data, so the decision is
driven by a measured accuracy cost rather than a guess.

Because the whole pipeline is built around a fixed raw-event contract, the hardware only ever
replaces the *data source*: the features, model, explanations, and interface are identical whether
data comes from a recording, a synthetic generator, or a live board. For demonstration I stream a
recording through the exact same path the hardware will use.

---

## Conclusion

I set out to turn low-cost physiological signals into an honest, personalized, real-time stress
estimate. I achieved a balanced accuracy of 0.919 under the strictest evaluation, above the
published benchmark, built entirely on a personalization method that adapts to each individual. Just
as importantly, I made the system honest about its limits: it refuses to answer when it cannot see,
it reports its own error bar, and it is structurally barred from claiming to diagnose anything. The
main constraint is the size of the training data, and the clearest next steps are adding the missing
skin-conductance sensor and validating against the two held-out datasets I have deliberately kept in
reserve.
