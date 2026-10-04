# Two-stage decision head-to-head

Pre-registered test of one claim: does the frozen `quantum_probability` routine in
`insight137-eap` 2.0.0 predict the unknown-condition choice rate in two-stage decisions
better than the classical law of total probability, on studies it was never evaluated on?

The plan lives in the Obsidian vault:
`Work/Insight137/Pre-registration — Two-Stage Decision Head-to-Head (draft).md`.

**Status (2026-10-04): not registered. The held-out set has not been analysed.**

## Files

| File | What it is |
|---|---|
| `h2h_analysis.py` | The analysis. Refuses to touch held-out rows without `--registration-id`. |
| `studies.csv` | One row per triple: 5 development rows, 34 held-out rows, with a source location per row. |
| `screening_log.md` | What was searched, what was included or excluded and why, and what is still pending. |
| `test_h2h_analysis.py` | Tests. Model checks use development rows only; statistics checks use made-up numbers. |
| `requirements.txt` | Pinned dependencies. |
| `.gitattributes` | Stops git from converting line endings, so the recorded file hashes hold on every machine. |

## Run

Use a clean environment so the released 2.0.0 package is imported, not the working-tree
copy at the repo root. The script checks the module's SHA-256 and stops if it differs.

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
python h2h_analysis.py                           # development set only
```

After the plan is registered, and only then:

```bash
python h2h_analysis.py --set heldout --registration-id <registration URL> --out results.json
```

## Expected development-set output

```
Mean absolute error  M_C=0.166  M_Q=0.105  M_K=0.094  M_O=0.102
```

`M_O` is the oracle "best classical mixture", written `M_C*` in the plan.
