# Contribution workflow

## Before work starts
- Assign a requirement owner and agree on the expected evidence and report section.
- Create a GitHub issue for work that needs tracking.
- Confirm that the notebook and repository agree on inputs, outputs, and assumptions.

## Branches and changes
- Use one focused branch per task: `feat/<topic>`, `fix/<topic>`, or `docs/<topic>`.
- Keep commits small and descriptive. Never commit raw data, credentials, or large generated outputs.
- Leave raw source files unchanged; write cleaned data to a separate working location.

## Pull requests
- Open a pull request against `main` and link the issue or requirement number.
- Summarize what changed, how it was checked, and which notebook/report sections are affected.
- Include figures or metric summaries when relevant; do not add test-set-driven tuning.
- Request review from one teammate. Resolve comments before merging.
- Keep the shared Kaggle notebook aligned with accepted changes on `main`.

## Ownership and review
- Each member owns four of the 16 Milestone 1 requirements. Record owners and status in the team task board or issue list.
- Every member reviews the final notebook and report before submission.
- Keep decisions, caveats, and contribution notes in the repository so the team can reproduce the work.
