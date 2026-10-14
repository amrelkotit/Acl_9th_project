# Milestone 1: submission readiness

This is a handoff checklist, not a submission confirmation. Deadline: **18 October 2026 at 11:59 PM Cairo time**, from the supplied milestone brief. Evaluation dates and slot booking require their own official announcement.

## Review artifacts

- `notebooks/milestone1.ipynb`: integrated implementation plus acceptance rows 21–30, with local execution outputs.
- `output/pdf/ACL_Milestone_1_Review_Report.pdf`: analytical review PDF from batch 2; unchanged by batch-3 acceptance work.
- `reports/milestone1-report.md` and `reports/milestone1-evidence/`: analytical text and measured evidence.
- `reports/batch3-acceptance.md` and `reports/batch3-evidence/`: current row reconciliation, final feature register/contract, accuracy target, inference checks, row counts and PDF page/figure mapping.
- `reports/batch3-evidence/submission_manifest.json`: exact artifact hashes for this review package. A changed notebook/report/PDF requires a new manifest and applicable checks.
- [Draft PR #2](https://github.com/amrelkotit/Acl_9th_project/pull/2): reviewed branch, not merged or approved by teammates.

The user goal of at least 80% ordinary binary accuracy is met by all three frozen models on the historical held-out test. Always predicting no podium also exceeds 80%; retain podium precision/recall/F1, ROC-AUC, trapezoidal PR-AUC and separately defined AP. Do not report ROC-AUC or F1 as accuracy, and do not tune against the revealed test set.

## External actions still required

| Check | Current evidence | Completion evidence required |
| --- | --- | --- |
| Four-member access | Owner can use the private repository/notebook; others unverified | Each member successfully opens the intended private links |
| Team review | No member sign-off recorded | All four review the notebook, report, decisions and limitations; contributions match actual work |
| Historical semantics | Explicit conservative rules and limitations | Review uncertain shared drives, starters/pit-lane cases, grid cutoff assumptions and mechanical causes; disclose what cannot be resolved |
| PR review/merge | Draft PR; no review/merge claimed | Teammate review and accepted changes, following contribution workflow |
| Kaggle synchronization | Original private version 5 backup; updated upload previously denied | Authorized synchronization of accepted notebook; preserve the backup and privacy/access settings |
| Kaggle execution | Local Run All only | Restart kernel and Run All in Kaggle with the intended dataset/dependencies; no hidden manual steps; save completed version |
| Output agreement | Local artifact hashes and numeric verification | Confirm Kaggle results match the reviewed report and repository version, allowing documented floating-point tolerance |
| Official submission | Not submitted | Submit accessible Kaggle/repository links and reviewed PDF through the [official form](https://forms.gle/esDxHsvfzffYHVAZ9), then save confirmation |

The earlier browser upload denial must not be bypassed through another upload route or by pasting the notebook code. The complete local notebook is ready for later authorized synchronization.

## Sign-off discipline

Keep the shared plan's original owners and internal targets. A passing local check does not establish human review, accessible team links, Kaggle execution or submission. Leave Done cells blank until the relevant owner/reviewer verifies the evidence. This audit does not authorize any Done-cell updates.

Rows 31–40 remain for the final ten-row batch, including evaluation preparation, historical sensitivity, wording clarification, final reproducibility and whole-package reconciliation.
