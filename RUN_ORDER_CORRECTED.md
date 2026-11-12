# Corrected Stage 1–2 run order

## Important evaluation note
The supplied corrected DS2 has previously been accessed during Stage 4.
It is **not an untouched test set**. Use it only for exploratory/internal
checks with that limitation disclosed. Do not call it a pristine final test
set or use it for model selection and then report it as untouched.

## 1. Put the corrected data in the expected location
Extract the companion corrected-data ZIP into the project root. It contains:
- `artifacts/ds1.npz`
- `artifacts/ds2.npz`
- `artifacts/manifest.json`

The source code expects these exact paths. Do not rename the files manually.

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Verify the data before training
```bash
python src/verify_dataset.py
```
Stop if any hard check fails. Warnings should be reviewed, not ignored.

## 4. Generate the corrected frozen folds exactly once
```bash
python src/make_folds.py
```
This creates `src/folds.json`, recording the DS1 SHA-256 hash and the
subject-to-fold assignment. Keep this file with the source and use the same
copy on every machine.

## 5. Run Stage 2
Smoke test (not reportable):
```bash
python src/run_grid.py --models lr xgb --subset 5000
```
Delete smoke-test outputs before the real run:
```bash
rm -rf artifacts/preds artifacts/preds_seed* artifacts/grid_summary*.json
```
Then run:
```bash
python src/run_grid.py --models lr xgb
python src/run_grid.py --models cnn
```
For multi-seed CNN runs, use the agreed seed list and keep each seed's
predictions separate. Do not mix predictions made with different DS1 hashes
or different `folds.json` files.

## 6. Validate summaries
```bash
python src/verify_summary.py
python src/seed_report.py
```
Only report metrics generated from this corrected DS1 and its matching folds.
The historical tables in `stage2.md` are not corrected-split results.
