# Stage 1 — Data Preparation (patient-independent protocol)

Boxes 1–7 of the framework. Produces `ds1.npz` / `ds2.npz`, which are the
only inputs Stage 2 is allowed to read.

## Run it

```bash
pip install -r requirements.txt

python src/download.py         # ~90 MB from PhysioNet, run once
python src/build_dataset.py    # builds artifacts/ds1.npz, ds2.npz
python src/verify_dataset.py   # acceptance tests — must pass before Stage 2
```

This source bundle does not include a synthetic-record generator. Use the real MIT-BIH records for the acceptance run; do not create or report synthetic data as experimental evidence.

## What comes out

| File | Contents |
|---|---|
| `ds1.npz` | `X (n,216)` beats · `rr (n,4)` timing · `y (n,)` labels · `pid (n,)` subject · `record_id (n,)` record |
| `ds2.npz` | same, from 21 different subjects |
| `manifest.json` | per-record counts, class distribution, config echo — paste into the paper's Methods |

## Expected output

Exact beat totals and class distributions must be generated from the corrected patient-grouped split and recorded in `artifacts/manifest.json`. Do not reuse the earlier 50,977 / 49,668 totals: moving record 202 changes both partitions.

## Design decisions, and why

**Records.** 48 total, minus paced 102/104/107/217 as ANSI/AAMI EC57
requires, leaves 44. The conventional record split places 201 in DS1 and 202 in DS2, but PhysioNet confirms both are from the same subject. For strict subject grouping, record 202 is moved into DS1 alongside 201, giving 23 DS1 records (22 subjects) and 21 DS2 records (21 subjects). Subject IDs are stored separately from record IDs, so both recordings remain grouped during cross-validation.

**R-peaks come from the database's expert annotations, not a detector.**
Detector error would confound the imbalance results, and it is a separate
variable we are not studying. Say this explicitly in Methods — reviewers
ask.

**Window: 200 ms before, 400 ms after R (216 samples at 360 Hz).**
Keeps the preceding P wave and the following T wave inside the window,
which Stage 5 needs to aggregate attributions into P/QRS/T regions.
Boundaries live in `config.SEGMENTS`.

**Normalisation is per-record z-score.** Statistics never cross a patient
boundary, so it is leak-free by construction. It also removes the
per-patient amplitude and baseline offset that a model would otherwise
use to identify the patient rather than the disease.

**Four RR features.** `pre_rr`, `post_rr`, `local_ratio` (vs the last 10
beats), `global_ratio` (vs the whole record). Class S is defined by
prematurity, and a window cropped around a beat's own R-peak discards
exactly that. Without these, S is unidentifiable no matter which model is
used. The two ratios are relative to the patient's own rhythm, so they
help across patients too.

**First and last beat of each record are dropped** — no pre-RR or post-RR
exists for them.

**Both `pid` (subject ID) and `record_id` must survive to the end of the project.** Stage 2's `StratifiedGroupKFold`
depends on it. A plain `KFold` would silently re-create the leak this
whole stage exists to remove.

**Q is kept in the data but excluded from metrics.** After paced-record
removal it holds 8 and 7 beats. See the note in `src/config.py`.

## Before you trust any output

`verify_dataset.py` runs three tiers:

- **PROTOCOL** — patient disjointness, paced exclusion, record lists.
  A failure here invalidates everything downstream.
- **SHAPE / NORMALISATION** — array alignment, window length, no NaNs.
- **RR DIAGNOSTIC** — S beats must show a *lower* local RR ratio than N
  beats, since S beats are premature by definition. If this fails, the RR
  features are wired wrong and Class S will stay broken regardless of
  architecture.

The verifier prints the measured separation for the current corrected datasets; do not reuse diagnostic values from the earlier record-wise split.

The SCALE check compares totals against approximately 53,100 (DS1) and 47,500 (DS2) for the corrected partition.
Warnings there are expected on synthetic data and meaningful on real data.

## Sign-off before Stage 2

- [ ] `verify_dataset.py` passes all PROTOCOL and SHAPE checks on real data
- [ ] Guide has signed off on the AAMI mapping table in `config.py`
- [ ] `manifest.json` class distribution recorded in the shared results sheet
- [ ] `ds1.npz` / `ds2.npz` uploaded to the shared Drive folder
- [ ] Repo pushed; `artifacts/` gitignored (regenerate, don't commit)

## Next

Stage 2 reads `ds1.npz` only. `ds2.npz` stays sealed until the single
final evaluation run — no peeking for hyperparameters, no "just checking"
on the test set. That discipline is the paper's main claim.
