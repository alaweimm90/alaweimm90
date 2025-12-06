# OptiBench (Skeleton)

Files
- `registry.yaml` – list of benchmark definitions to replay (e.g., QAPLIB filters, modes, time limits).
- `replay.py` – runs entries from the registry and writes results to `results/optibench/`.

Usage
```
python -m optibench.replay --entry qaplib-mini --out results/optibench/qaplib-mini.json
```

