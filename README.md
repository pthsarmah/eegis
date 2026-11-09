# EEGIS Imagined-Speech Classification

Eight-class EEG classification for:

`Sí`, `No`, `Baño`, `Hambre`, `Sed`, `Ayuda`, `Dolor`, `Gracias`

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Place the EEGIS dataset under `data/`.

## Run

Production configuration uses the validation-weighted 10-model ensemble and
kNN `k=5`:

```bash
.venv/bin/python -u run_experiments.py \
  --data data \
  --cache .cache/features-8class.npy \
  --cv-folds 5 \
  --max-features 512 \
  --production-k 5
```

The command reports accuracy, weighted/macro precision, recall, F1, model
CV scores, ensemble weights, and an 8×8 normalized confusion matrix.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline diagrams
- [`EXECUTIVE_REPORT.md`](EXECUTIVE_REPORT.md) — results and limitations
