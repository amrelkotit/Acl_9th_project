# Source code

Add reusable Python modules here as the project grows. Suggested boundaries:
- `data_loading.py` — locate and load source tables.
- `cleaning.py` — validated transformations on working copies.
- `features.py` — leakage-safe pre-race features and chronological splits.
- `analysis.py` — the three data-engineering questions.
- `models.py` and `evaluation.py` — model pipelines and metrics.

Keep functions small, document assumptions, and avoid paths tied to one person's machine. Notebook sections may import these modules once they are implemented.
