# Milestone 1: submission readiness

This is a handoff checklist, not a submission confirmation. Deadline: **18 October 2026 at 11:59 PM Cairo time**, from the supplied milestone brief. Evaluation dates and slot booking require their own official announcement.

## Review artifacts

- `notebooks/milestone1.ipynb`: complete local implementation and checks for plan rows 1–40, with 37 executed code cells.
- `output/pdf/ACL_Milestone_1_Review_Report.pdf`: current 29-page analytical review PDF, including the final accuracy/historical-sensitivity addendum and all 26 figures.
- `reports/milestone1-report.md` and `reports/milestone1-evidence/`: analytical text and measured evidence.
- `reports/batch3-acceptance.md` and `reports/batch3-evidence/`: prior acceptance checkpoint, final feature register/contract, accuracy target, inference checks and row counts.
- `reports/final-compliance.md` and `reports/final-evidence/`: current all-40-row/eight-deliverable reconciliation, sensitivity evidence and final PDF page/figure mapping.
- `reports/final-evidence/final_submission_manifest.json`: current artifact hashes. Earlier batch manifests are historical checkpoints. A changed notebook/report/PDF requires a new manifest and applicable checks.
- `docs/evaluation-walkthrough.md`: mock-evaluation agenda and evidence-based questions for all four members; no attendance/sign-off is implied.
- [Draft PR #2](https://github.com/amrelkotit/Acl_9th_project/pull/2): reviewed branch, not merged or approved by teammates.

The user goal of at least 80% ordinary binary accuracy is met by all three frozen models on the historical held-out test. Always predicting no podium also exceeds 80%; retain podium precision/recall/F1, ROC-AUC, trapezoidal PR-AUC and separately defined AP. Do not report ROC-AUC or F1 as accuracy, and do not tune against the revealed test set.

## External actions still required

| Check | Current evidence | Completion evidence required |
| --- | --- | --- |
| Four-member access | Owner can use the private repository/notebook; others unverified | Each member successfully opens the intended private links |
| Team review | No member sign-off recorded | All four review the notebook, report, decisions and limitations; contributions match actual work |
| Historical semantics | Explicit conservative rules and limitations | Review uncertain shared drives, starters/pit-lane cases, grid cutoff assumptions and mechanical causes; disclose what cannot be resolved |
| PR review/merge | Draft PR; no review/merge claimed | Teammate review and accepted changes, following contribution workflow |
| Kaggle synchronization | Original private version 5 backup; final upload explicitly authorized, but extension still reports file-URL access disabled | Successfully import the verified notebook; preserve the backup and private access settings |
| Kaggle execution | Local Run All only | Restart kernel and Run All in Kaggle with the intended dataset/dependencies; no hidden manual steps; save completed version |
| Output agreement | Local artifact hashes and numeric verification | Confirm Kaggle results match the reviewed report and repository version, allowing documented floating-point tolerance |
| Official submission | Not submitted | Submit accessible Kaggle/repository links and reviewed PDF through the [official form](https://forms.gle/esDxHsvfzffYHVAZ9), then save confirmation |

The user explicitly authorized uploading the final notebook and Run All on 14 October 2026. The authorized upload still failed because the browser extension reports file-URL access disabled, including after the user reported enabling it. No upload or Kaggle execution has been completed. Resolve the extension setup or select the file manually in Kaggle's Import Notebook dialog; do not bypass browser protections through another agent upload route.

The user also authorized inspecting the official form. Browser security still blocks forms.gle through a saved permission preference. No form fields were inspected or submitted. That saved setting must be changed by the user before browser inspection can proceed.

## Sign-off discipline

Keep the shared plan's original owners and internal targets. A passing local check does not establish human review, accessible team links, Kaggle execution or submission. Leave Done cells blank until the relevant owner/reviewer verifies the evidence. This audit does not authorize any Done-cell updates.

All local work for rows 31–40 is prepared, including evaluation material, independent historical recounts/sensitivity, wording coverage, runtime evidence and whole-package reconciliation. Actual team walkthrough/review, final Kaggle execution and official submission remain external completion gates.

To reproduce the final local package: `python build_notebook.py --final`, `python run_batch1.py`, `python verify_batch2.py`, `python render_report.py`, `python package_batch2.py`, then `python verify_final.py`. `requirements-tested.txt` records the tested local versions; it is not a guarantee across operating systems. Preserve the model-choice lock.
