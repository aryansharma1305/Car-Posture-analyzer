# Car Posture Analyzer

[![tests](https://github.com/aryansharma1305/Car-Posture-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/aryansharma1305/Car-Posture-analyzer/actions/workflows/tests.yml)

Camera-based driver posture measurement for automotive seat ergonomics research.

A webcam watches a seated driver, MediaPipe Pose gives the joint landmarks, and this
project turns them into calibrated posture angles, a comfort score against published
automotive ergonomics guidance, and a per-session CSV you can attribute to a specific
driver, seat and camera position.

It also runs as a desk-posture monitor, because the two share everything except the
definition of good posture — and keeping that distinction explicit is the whole design.

```
camera ─► pose_core ─► landmark_confidence ─► posture_angles ─► smoothing
                            (is this                 (angles,          (EMA, NaN-safe)
                             landmark real?)          signed + magnitude)
                                                                   │
                        ┌──────────────────────────────────────────┤
                        ▼                                          ▼
                  driver_model                             session_schema
            (DRIVING | DESK: one set of                (one CSV contract +
             ranges, risks, thresholds)                 metadata sidecar)
                        │
       ┌────────────────┼────────────────┬──────────────────┐
       ▼                ▼                ▼                  ▼
 posture_rules   posture_fullbody   ergonomics_scores   driver_comfort_analyzer
   (labels)        (body state)      (RULA / REBA)        (comfort + risk)
```

---

## Why recline is the point

Most posture software is desk software. At a desk, upright is the goal and any departure
from vertical is a fault. In a car it is the opposite: the seat exists to *support* a
reclined torso, and automotive guidance puts a correctly seated driver's torso line
**5–30° behind vertical**.

That means a desk model, pointed at a driver, reports a correctly seated person as a
sloucher. It also means a direction-invariant angle — the kind you get from
`abs(angle_from_vertical)` — cannot tell a supported recline from a driver hunched over
the wheel. They produce the same number and they are opposite postures.

So this project carries two things most posture code does not:

- **Signed torso angles.** `trunk_signed` / `neck_signed`, where **negative means
  reclined**. The magnitudes (`trunk_deg`, `neck_deg`) are derived from them, so the two
  views can never disagree.
- **Two named posture models**, not one with tweaked constants. `driver_model.DRIVING`
  and `driver_model.DESK` each own their ideal ranges, risk rules, label thresholds, and
  which *view* of the torso angle each threshold reads.

The same frame, under both models:

```
driving  trunk_deg=18.4  label=Neutral  advice=['All Clear: Good posture.']
desk     trunk_deg=18.4  label=Slouch   advice=['Trunk: Mild slouching detected.']
```

Both are correct. That is the point.

### Sources for the driving ranges

| Angle | Range | Source |
|---|---|---|
| `trunk_signed` | −30 … −5 | Grandjean (1980) backrest 20–25° from vertical; SAE J1100 / J826 dimension A40 ("back angle") ≈ 22–25° |
| `neck_signed` | −10 … +15 | Head near vertical; slight forward is normal road scanning |
| hip (trunk–thigh) | 95 … 120 | Rebiffé (1969), *Ergonomics* 12(2) |
| knee | 95 … 135 | Rebiffé (1969) |

The desk ranges are the conventional display-screen-equipment values (neck and trunk near
vertical, hips and knees near 90°, shoulders relaxed). Full citations and the reasoning
live in the `driver_model.py` docstring, next to the numbers they justify.

---

## Unknown is not zero

MediaPipe returns a landmark for **every** joint on every frame, whether or not it saw
one — occluded joints are extrapolated from the rest of the body. A driver whose legs are
under the dash still produces knee and hip coordinates, and nothing about the numbers says
they were invented.

`landmark.visibility` is the only signal that separates a joint the model saw from one it
guessed, so this project threads it through the whole pipeline:

- **`landmark_confidence.py`** declares which landmarks each angle is built from, and
  scores each angle as the **minimum** confidence over its inputs — an angle is only as
  good as its worst landmark.
- **An angle below the threshold is `NaN`**, not a number. `geometry_utils.is_measured()`
  is the single gate every consumer checks before comparing.
- **The smoother holds its state** through an unmeasured sample instead of being poisoned
  by it, and passes the gap through as unknown rather than repeating a stale value.
- **Frames are excluded, not averaged.** A frame is an *observation* only if enough of the
  model's weight was measured **and** every required angle was present — for `DRIVING`
  that is the torso line, because a frame that saw only legs is not a weaker reading of
  the seat model, it is a reading of something else.

Every session CSV therefore carries its own audit trail:

```
trunk_signed  knee_L  frame_confidence  measured_fraction  unmeasured                    observation
-18.4         ''      0.05              0.45               left_hip_angle|left_knee_...  False
```

`observation` is the column to filter on. Reports that skip it will quietly average
guesses into their results.

---

## Install

Developed and tested on Python 3.10. MediaPipe is the constraint on newer versions —
check it has a wheel for your interpreter before upgrading.

```bash
git clone https://github.com/aryansharma1305/Car-Posture-analyzer.git
cd Car-Posture-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A webcam is needed for `monitor`. Everything else works on recorded CSVs.

---

## Usage

### Record a session

```bash
python3 main.py monitor \
    --driver-id D017 \
    --seat-id SEAT-PREMIUM-A \
    --vehicle XUV700 \
    --camera-position left-side
```

`--camera-position` is not optional in practice. In 2-D pixel space the sign of a lean
encodes image-left vs image-right, not body-forward vs body-backward; the two only
coincide when the camera is to the **side** of the driver, and which side decides the
sign. `left-side` / `right-side` set it; `front` / `rear` leave it unknown rather than
guessed, and the monitor warns you that the lean sign may be inverted.

Without `--seat-id` the session still records, but it cannot be grouped for seat
comparison and the monitor says so.

Add `--enhanced` for the desk monitor (quality score + break reminders) instead of the
in-car one.

### Score a single posture

```bash
# automotive model — pass trunk/neck SIGNED, negative = reclined
python3 main.py score --reference driving --trunk -20 --neck 5 --hip 105 --knee 115
#   Overall Comfort: 100.0/100   Category: Excellent

# the same driver under the desk model
python3 main.py score --trunk 20 --neck 5 --hip 105 --knee 115
```

Unspecified angles are filled with the midpoint of the chosen model's own ideal range, so
they are neutral rather than a hardcoded guess.

### Analyse

```bash
python3 main.py list                       # sessions in posture_logs/
python3 main.py analyze <session>.csv      # one session, as text
python3 main.py report --days 7            # across sessions
python3 main.py visualize <session>.csv    # charts
```

For the research-grade driver report (seat-adjustment insights, time-series, measurement
coverage):

```bash
python3 driver_comfort_reporter.py <session>.csv --report-type research
```

A session whose frames all fall below the measurement threshold gets a refusal explaining
why, not a report built from guesses.

### Compare seats

The research output. Groups every session by the `seat_id` in its sidecar, scores each
under the model it was recorded with, and excludes unobservable frames:

```console
$ python3 compare_seats.py
seat                    mean     sd    min    max  frames  excluded
SEAT-BASELINE-B         95.2    0.7   94.1   96.4      30        10
SEAT-PREMIUM-A         100.0    0.0  100.0  100.0      40         0

SEAT-PREMIUM-A scores 4.8 points above SEAT-BASELINE-B.
```

Sessions recorded without `--seat-id` are listed as skipped rather than silently pooled.
The difference is a difference of means with no significance test — read
`validate_angles.py report` first to find out whether a gap that size is larger than the
sensor's own error.

### Configure

```bash
python3 main.py config --list-profiles
python3 main.py config --apply-preset driving     # also: office, gaming, student
```

Profiles carry runtime settings (camera, confidence, cadence, alerts) and may *name* a
posture model. They deliberately do **not** restate the ergonomic numbers: a profile
leaves a threshold unset to inherit it from the model, and sets it only to deviate on
purpose. `--config <profile>` is passed through to the monitor.

---

## The session format

One schema, defined once in `session_schema.py`, written by every monitor. 29 columns,
`SCHEMA_VERSION = 3`. Columns a monitor does not compute are written **empty** rather than
omitted, so every file has identical width and `pd.concat` can never produce a ragged
frame.

The contract that matters:

| | |
|---|---|
| `timestamp` | ISO-8601 wall clock. Always absolute, never elapsed. |
| `t_sec` | Float seconds from session start. Always relative, never a clock. |
| `*_signed` | Signed degrees. **Negative = reclined.** |
| `*_deg` | Direction-invariant magnitudes 0–90, derived from the signed values. |
| *(unmeasured)* | An empty cell — never the string `"nan"`. |

Each CSV gets a `<stem>.meta.json` sidecar written **on construction**, so a session is
self-describing even if it crashes mid-run: driver, seat, vehicle, camera position,
resolved lean direction, and which posture model produced its labels.

### Migrating older logs

`migrate_logs.py` converts the five historical log conventions to the canonical schema.
Dry-run by default; originals are never touched (asserted by a test that byte-compares the
source directory before and after).

```bash
python3 migrate_logs.py            # report what would change
python3 migrate_logs.py --apply    # write to posture_logs/migrated/
```

It writes a sidecar per session recording the writer family, the inferred seating context,
any magnitude columns corrected from a pre-fold convention, and the measurement coverage
of the rows. `posture_logs/migrated/` is gitignored as derived data.

**What the historical corpus can and cannot support.** The 36 migrated sessions have no
`driver_id`, `seat_id` or `camera_position`, and no `trunk_signed` — not because the
migration dropped them but because no writer ever captured them, and lean direction is not
recoverable after the fact. They are a valid angle corpus for pipeline and trend work. They
cannot support a seat comparison, and the `observation` column says so per row rather than
leaving you to find out.

---

## Validating the measurement

Every number above is an estimate of a joint angle from a 2-D webcam. Until that
estimate is compared against an independent reference, its accuracy is unknown — and a
seat comparison built on an unvalidated sensor is not a finding. If the trunk estimate
carries ±8° of error, the `DRIVING` ideal range is only 25° wide and a few points of
difference between two seats means nothing.

`validate_angles.py` answers three questions in order.

**1. Is the arithmetic right?** No camera needed. It feeds `compute_angles` skeletons
whose true angle is known exactly, isolating this project's own pixel-to-angle path:

```console
$ python3 validate_angles.py synth
 recline  expected  measured    error
     0.0     -0.00      0.00    0.000
    25.0    -25.00    -25.00   -0.000
    45.0    -45.00    -45.00    0.000

worst torso error 0.0000 deg, worst limb/neck error 0.0000 deg
PASS - the pixel-to-angle path is exact to within 0.01 deg.
```

That is the control. Because it passes, any error measured afterwards belongs to the
camera and MediaPipe rather than to the code reading them.

**2. Is the sensor accurate?** Read the protocol first — it covers camera siting,
reference sources, and the sweep design:

```bash
python3 validate_angles.py protocol

python3 validate_angles.py capture \
    --recline 25 --reference-source sternum \
    --subject-id S1 --camera-distance-cm 150
```

`--recline` is **degrees of recline from vertical, positive** — how a seat is specified.
Upright is 0, a typical car backrest is 20–25. The conversion to `trunk_signed`
(negative = reclined) happens in one function, because getting it backwards would invert
every conclusion while still producing a plausible table.

**3. Is it accurate *enough*?**

```console
$ python3 validate_angles.py report
  bias (mean signed error)   +2.86 deg
  between-trial SD           1.73 deg
  95% limits of agreement    -0.53 to +6.26 deg
  measured = 0.977 x truth + +2.45

  DRIVING ideal trunk range  -30 to -5 deg (25 deg wide)
  95% agreement width        6.8 deg (27% of the range)
  VERDICT  Usable. Agreement is comfortably inside the ideal range.
  with  5 trials per seat, smallest resolvable difference 2.2 deg
```

Three numbers carry the result. **Bias** is a systematic offset — correctable, and it
cancels when comparing two seats, but it shifts a single seat against the cited ideal
range. **SD** is what actually limits a seat comparison. **Slope** should be near 1.0;
away from it means recline is compressed as it grows, which distorts comparisons across
different recline settings even when bias is zero.

Aggregation uses **one value per trial**, never per frame. Frames of a person sitting
still are heavily correlated, so pooling them would shrink the error bar by √frames and
make the sensor look far better than it is. The same reason the minimum-detectable-
difference table counts separate sittings, not repeats without getting out of the seat.

---

## Module map

| Module | Responsibility |
|---|---|
| `main.py` | CLI entry point for every command |
| `pose_core.py` | MediaPipe wrapper; the one landmark index map, coordinates **and** visibility |
| `geometry_utils.py` | Angle primitives, the `NaN` unmeasured sentinel, `is_measured()` |
| `landmark_confidence.py` | Which landmarks each angle needs, and whether they can be trusted |
| `posture_angles.py` | Signed + magnitude angles from landmarks; masks the untrustworthy |
| `smoothing.py` | Per-angle EMA that a lost landmark cannot poison |
| `driver_model.py` | **The one posture model.** `DRIVING` / `DESK`: ranges, risks, thresholds |
| `posture_rules.py` | Seated-posture labels against a model |
| `posture_fullbody_rules.py` | Standing / sitting / lying / squatting, and upper-body labels |
| `ergonomics_scores.py` | Simplified RULA and REBA categories, plain-language advice |
| `posture_scoring.py` | Desk quality score by category |
| `driver_comfort_analyzer.py` | Driver comfort score, ergonomic risk, fatigue drift |
| `driver_comfort_reporter.py` | Research reports, charts, seat-design insights |
| `posture_analytics.py` | Cross-session analytics and visualisation |
| `session_schema.py` | The CSV contract, the writer, the metadata sidecar |
| `migrate_logs.py` | Legacy log migration |
| `config_manager.py` | Runtime settings and profiles |
| `posture_live_full.py` | **In-car monitor** (`DRIVING`) |
| `posture_monitor_enhanced.py` | **Desk monitor** (`DESK`) |
| `posture_live.py` | Early prototype, kept for reference |
| `validate_angles.py` | **Measures the measurement**: bias, scatter, linearity, resolution verdict |
| `validation_geometry.py` | Synthetic skeletons with an exactly known angle — the control |
| `compare_seats.py` | Groups sessions by `seat_id` and compares comfort across seats |

---

## Tests

```bash
python3 -m pytest tests -q
```

238 tests, no camera required. They are written to pin *contracts* rather than current
output — the column schema and its width, the equivalence between the signed and magnitude
angle views, that every module reads its thresholds from `driver_model` rather than a local
copy, that no risk rule fires on a posture the model calls ideal, that an unmeasured angle
never reaches a CSV as text, and that a lost landmark cannot be scored as a measurement.

Several carry the regression they exist to prevent in the docstring, which is usually the
fastest way to understand why a rule is shaped the way it is.

---

## Known limitations

- **The sensor has not been validated yet.** `validate_angles.py` exists to fix this and
  its arithmetic control passes, but no real trials have been recorded. Until they are, the
  accuracy of every angle — and therefore of every comfort score and seat comparison — is
  unknown. Run the protocol before trusting a number.
- **Thresholds are stated, not calibrated.** The comfort category cut-offs (85/70/50), the
  per-degree score slopes, and `MIN_MEASURED_FRACTION = 0.5` are documented judgements. The
  joint *ranges* are cited; these are not, and calibrating them needs measured discomfort
  data.
- **Lean direction needs a side camera.** From the front or rear, leaning forward barely
  moves `x`, so the sign carries little signal. The real fix is MediaPipe
  `pose_world_landmarks` (true 3-D), which this project does not use yet.
- **Confidence masking is binary.** A landmark at 0.51 is trusted exactly as much as one at
  0.99. Weighting an angle's contribution by its confidence would be a better estimator.
- **RULA and REBA are simplified** three-band approximations, and both score trunk flexion
  as a magnitude. Neither instrument was designed for a seat that supports the torso, so
  read the driver comfort score — not RULA — for that judgement.

---

## Licence

No licence file yet; all rights reserved by default. Open an issue if you need one added.
