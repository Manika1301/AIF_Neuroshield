# External validation: Stress-Predict Dataset

Generated: 2026-07-11T01:42:57.604414+00:00
Frozen model: `m1_wesad_features_v1` (feature_version `features-v1`)

## Channel compatibility

Stress-Predict was recorded on the same Empatica E4 wrist hardware family as WESAD, so every channel M1 needs (BVP, EDA, TEMP, ACC) is present and every `features-v1` column is computable -- this is a same-channel-set external validation, not a partial one. ACC is in the same raw unit (1/64 g) on both datasets. Ground truth differs: Stress-Predict labels come from a per-second binary table (0/1) derived from a different stressor protocol (Stroop colour-word test, interview, hyperventilation) than WESAD's stress task, so this is a test of generalization across stress *inducers*, not just across people.

One subject (S01) exists in the raw data but has no entry in the processed label table (likely an excluded pilot participant) and is skipped for lack of ground truth.

## Result: ran through the frozen pipeline

- Subjects evaluated: 34
- Windows evaluated: 3710
- Class balance: baseline=2482, stress=1228
- Balanced accuracy: 0.541
- Macro F1: 0.506
- Confusion matrix [[TN,FP],[FN,TP]]: [[1115, 1367], [451, 777]]

## Skipped subjects

- `S01`: no processed ground-truth labels available for this subject

## Per-subject balanced accuracy

| Subject | Windows | Balanced accuracy |
|---|---|---|
| S02 | 117 | 0.481 |
| S03 | 109 | 0.587 |
| S04 | 116 | 0.533 |
| S05 | 108 | 0.507 |
| S06 | 109 | 0.573 |
| S07 | 111 | 0.503 |
| S08 | 103 | 0.549 |
| S09 | 105 | 0.467 |
| S10 | 100 | 0.678 |
| S11 | 108 | 0.514 |
| S12 | 110 | 0.667 |
| S13 | 110 | 0.516 |
| S14 | 118 | 0.629 |
| S15 | 114 | 0.596 |
| S16 | 113 | 0.437 |
| S17 | 109 | 0.676 |
| S18 | 113 | 0.437 |
| S19 | 105 | 0.616 |
| S20 | 112 | 0.542 |
| S21 | 106 | 0.600 |
| S22 | 101 | 0.456 |
| S23 | 101 | 0.541 |
| S24 | 108 | 0.500 |
| S25 | 106 | 0.554 |
| S26 | 107 | 0.462 |
| S27 | 111 | 0.553 |
| S28 | 109 | 0.609 |
| S29 | 108 | 0.514 |
| S30 | 110 | 0.548 |
| S31 | 109 | 0.468 |
| S32 | 105 | 0.498 |
| S33 | 106 | 0.566 |
| S34 | 117 | 0.483 |
| S35 | 116 | 0.575 |