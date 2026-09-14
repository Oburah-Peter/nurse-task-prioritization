# Nurse Task Prioritization — Technical Response

## Workflow Diagram

End-to-end pipeline from raw EHR events to a prioritized, ranked nurse task queue (left to right):

![Nurse task prioritization workflow diagram, left to right](./images/workflow_diagram.png)

*Blue = data engineering (Steps 1 to 4), Green = modeling (Steps 5 to 7), Yellow = nurse-facing output (Step 8), Pink = evaluation/feedback loop (Step 9).*

## 1. Organizing and Combining Events Over Time

All five event tables (chartevents, labevents, emar, emar_detail, procedureevents) share the same three core fields: patient ID, timestamp, and what happened. This makes it possible to normalize them into one common long-format table: `(patient_id, event_time, event_type, value, end_time)`.

From there, events are grouped into fixed time windows (e.g., hourly bins) to build a patient-hour trajectory table, one row per patient per hour, summarizing vitals, labs, medications, and procedures for that hour. Interval events like procedures or infusions are represented as "active/inactive" flags with elapsed duration rather than single timestamps, since they span a period rather than a moment. A parallel "time since last observed" column is kept for each feature, since measurement timing itself is informative.

This structure allows patients to be compared row by row, feeds cleanly into a model (which needs consistent structured input), and makes trends easy to spot over time.

## 2. Features Extracted From Each Table

**chartevents:** current vitals and deviation from normal range (e.g., HR > 120, MAP < 65); trend/slope over the last 30 to 60 minutes; composite early-warning scores (NEWS, SOFA); device-related signals (ventilator settings, new line placement).

**labevents:** abnormal-value flags and distance from normal range; rate of change (e.g., rising lactate, dropping hemoglobin); time since last result; derived organ-dysfunction indicators (e.g., AKI stage from creatinine).

**emar / emar_detail:** recent medication administrations and route; dose changes or titration of high-risk drugs (vasopressors, insulin); scheduled vs. PRN dosing, including overdue doses; presence of high-alert medications requiring monitoring.

**procedureevents:** recent or upcoming procedures and their post-procedure monitoring windows; procedure risk level; associated orders (e.g., frequent vitals after transfusion).

## 3. Assigning a Priority Score Across Multiple Patients

Tasks differ in type, so they can't be compared on raw values. The shared question across every task is: *"If this isn't done right away, how bad could it get, and how soon?"* This splits into two components: **Severity** (how dangerous if ignored) and **Time Pressure** (how soon it becomes a problem). These combine as:

```
Priority Score = Severity × Time Pressure
```

| Patient | Task | Severity | Time Pressure | Priority Score |
|---|---|---|---|---|
| A | Insulin due in 5 min | 6 | 9 | 54 |
| B | HR rising fast | 8 | 8 | **64** |
| C | Post-extubation monitoring | 7 | 6 | 42 |
| D | High lactate result | 9 | 7 | 63 |

**Ranked order:** Patient B (64), then Patient D (63), then Patient A (54), then Patient C (42).

Severity and time pressure can come from a rule-based checklist (simple, explainable, but rigid) or be learned from historical outcome data (more flexible, feeds into Section 4).

## 4. Machine Learning Approach, Target, and Evaluation

**Approach:** frame this as a risk-prediction model feeding into a ranking layer. Stage 1 trains a model (gradient-boosted trees, or a survival/time-to-event model) to estimate the probability of a significant decline within a short horizon (e.g., 1 to 4 hours). Stage 2 applies a learning-to-rank model (e.g., LambdaMART) to directly order tasks across all patients.

**Alternative approach:** a GRU/LSTM sequence model, specifically **GRU-D**, consumes the full patient trajectory directly rather than a single snapshot, learning temporal patterns without hand-engineered trend features. GRU-D is well-suited to irregular, frequently missing EHR data since it incorporates elapsed time since each variable was last observed directly into the network. This captures richer temporal dynamics but needs more labeled data and sacrifices some interpretability.

**Prediction target (`Y_t`):** a well-defined, objective adverse outcome (clinical deterioration, rapid response activation, or unplanned transfer) within a horizon `h`, not a reconstruction of historical nurse behavior, which would encode staffing bias rather than clinical urgency.

**Evaluation** compares predicted risk (`Ŷ_t`) against actual outcome (`Y_t`):

| Level | Purpose | Metrics |
|---|---|---|
| Predictive accuracy | Do risk scores match actual outcomes? | AUROC, AUPRC (weighted heavily given rare events), calibration |
| Ranking quality | Is the priority ordering correct across patients? | NDCG@k, Mean Average Precision |
| Clinical utility | Would this help in practice without alert fatigue? | False-alert rate, number needed to alert, retrospective "silent mode" or pilot comparison |

## 5. Major Data Challenges and Mitigations

| Challenge | Mitigation |
|---|---|
| **Irregular/informative sampling**: measurement frequency reflects perceived severity | Encode time-since-observation as a feature rather than imputing naively |
| **Non-random missingness**: a missing value often reflects a clinical decision, not randomness | Use masking plus time-decay (GRU-D) instead of mean-fill imputation |
| **Confounding by clinical response**: clinicians already act on perceived risk, so the data reflects treated, not natural, trajectories | Careful feature-time alignment and causal caution when interpreting associations |
| **Label scarcity / class imbalance**: no ground-truth urgency label exists and adverse events are rare | Use imbalance-aware metrics (AUPRC) and clinician-annotated validation samples |
| **Timestamp leakage**: delayed documentation can leak future information | Enforce strict point-in-time feature construction |
| **Cross-source standardization**: inconsistent naming/units across labs and medications | Build a standardization/mapping layer (e.g., to LOINC/RxNorm) early in the pipeline |
| **Generalizability**: case mix, staffing, and EHR systems differ across sites | Validate externally and monitor for performance drift |
| **Privacy/governance**: patient-level data requires strict controls | De-identification, access control, and IRB oversight throughout |

## Assumptions & Limitations

This is a conceptual and methodological response prepared without access to an actual dataset. No models have been trained or validated; all feature examples, scoring formulations, and evaluation plans are illustrative and intended to demonstrate approach and reasoning rather than empirical results.
