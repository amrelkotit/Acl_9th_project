# Milestone 1: team walkthrough and mock evaluation

Preparation material is ready; no meeting, attendance, oral pass or member sign-off is implied. Every member must be able to explain the complete project, as required by the brief. Preserve the existing Notion ownership assignments.

## Run the walkthrough

Allow about 45 minutes: 10 for audit/cohort, 10 for the historical questions, 10 for features/models/metrics, 10 for experiments/XAI/inference, and 5 for limitations/submission checks. Rotate the questions among all four members, including topics outside their assigned work. These durations are suggested agenda estimates, not official dates or deadlines.

For each answer: state the decision, show a relevant notebook result or report page, explain the evidence, then give its limitation. “An AI suggested it” is not a justification. Record actual attendance/review in the team's own sign-off record; leave the fields below blank until it occurs.

| Member | Walkthrough attended | Can explain all sections | Reviewed notebook/PDF | Private links open |
| --- | --- | --- | --- | --- |
| Amr Hazem Mohamed Mohamed Hassan Elkotit | Unverified | Unverified | Unverified | Unverified |
| Mark Bassem Mourad Samoel Emira | Unverified | Unverified | Unverified | Unverified |
| Loay Mohamed Sabry Farag Ismail | Unverified | Unverified | Unverified | Unverified |
| Karim Ahmed Mahmoud Abdelmageed | Unverified | Unverified | Unverified | Unverified |

## Questions and evidence to defend

| Question | Expected reasoning and evidence |
| --- | --- |
| What is one prediction? | One reconstructed race starter per driver/race; target positionOrder ≤ 3. Exact finishing position is outside scope. Show unique-key and target reconciliation checks. |
| Why are current-race points/status/laps forbidden? | They become known during/after the event. They are used for retrospective labels/cohort auditing, never in the predictor whitelist. Explain conditional prediction for reconstructed starters. |
| How are missing tokens and durations handled? | Literal backslash-N and blanks become missing on working copies; dates receive date types and durations numeric seconds. Show zero conversion failures, 14 unchanged source hashes and before/after plots. |
| Why not choose the best-finishing duplicate row? | That uses the label to choose the record. The baseline prioritizes positive grid then smallest resultId; the alternate smallest-ID-only sensitivity does not select using outcomes. Discuss shared-drive uncertainty. |
| Is grid zero pole position? | No. It may be a pit-lane start, missing historical grid or handover. Positive-lap evidence is retained with a separate flag; uncertain zero-grid/zero-lap cases are withheld. Not every historical identity is independently recoverable. |
| What defines the front row? | Actual grid positions 1/2 after penalties, not qualifying rank. Show Q1 counts/rates and pre-/post-2014 rankings; distinguish tiny denominators and excluded Indianapolis 500 events. |
| Does home racing cause podiums? | The primary comparison standardizes home/away rates to pooled grid-stratum weights. Exact-grid/era sensitivity and residual driver/team/circuit confounding prevent a causal conclusion. |
| What counts as a mechanical retirement? | Explicit component/system failure; exclude finishers, crashes, driver/regulatory causes and non-starts. Generic Retired and unclear damage are not silently classified. The all-ambiguous scenario is a sensitivity bound, not a true-rate estimate or confidence interval. |
| Why begin modeling in 2003? | Training-only qualifying coverage supports the restriction. Older races still inform historical analysis and prior form. Show coverage by era; Q2/Q3 absence can reflect elimination rather than an observed zero. |
| How are form and standings leakage-safe? | Shift driver outcomes before rolling; aggregate constructor results by race then shift; strictly earlier within-season as-of standings. Show source-event lineage and future-outcome/standings perturbation invariance. |
| What happens when a driver changes teams? | Driver form follows driverId; constructor form uses the current entrant constructorId's earlier history. New constructorIds reset team history rather than silently merging organizational renames. |
| Why these splits? | Train 2003–2019, validation 2020–2021, test 2022–2024; races are disjoint. Earlier validation/test outcomes can inform later events, matching sequential pre-race deployment. |
| Which transforms learn from which rows? | Median imputers, missing indicators, scaling and categorical encoding fit training only. Unknown categories use the fitted encoder's unknown handling. Show training medians, feature contracts and final leakage checks. |
| Why LR as baseline and why RF/NN? | LR is a regularized simple reference; RF models interactions; shallow NN tests nonlinear representation with one/two hidden layers. Show separately exposed trial sections, measured validation comparison and limitations. |
| How was the NN epoch chosen? | Training gradients only; explicit chronological validation AP selects architecture/checkpoint. No internal random validation split. Threshold is subsequently selected using validation F1. |
| Does 90.10% accuracy mean finding 90% of podiums? | No. RF test precision is 69.44%, recall 61.27%, F1 0.6510; always predicting no podium is already 84.93% accurate. RF accuracy is 5.17 percentage points higher. The user's 80% ordinary-accuracy goal is met, without guaranteeing future performance. |
| Why AP as primary, and is it PR-AUC? | Podiums are a minority, so AP evaluates positive-class ranking. AP is step-weighted; the separately reported PR-AUC is trapezoidal. Explain F1 at the locked threshold and complementary ROC-AUC. Six plan metrics plus AP cover named outputs; the brief's five/three wording is flagged. |
| Why not switch to LR after seeing higher test F1? | Selection used validation AP. The revealed test set cannot choose a new winner, threshold or preprocessing. The choice manifest remains unchanged. |
| What does ablation show? | Remove one feature group and retrain a fixed RF on the same rows/settings; measure validation AP/F1. Grid removal has the largest AP loss; context removal slightly improves AP. Correlated substitutes limit isolated-feature/causal interpretations. |
| What changed in the preprocessing-order trials? | Imputation/scaling and clipping/scaling orders alter training moments/ranges. Same rows, fixed LR/threshold and equal experiment dimensions. AP differences are small and F1 unchanged within each comparison. Show the distribution audit and figures. |
| Why did SHAP use a fallback? | TreeSHAP failed an external probability-additivity check in the tested runtime. Permutation SHAP of the actual fitted callable reconstructs probabilities without relaxing tolerance. Explain its sampled background and limited permutation budget. |
| How reliable is the local LIME explanation? | Its neighborhood R² is about 0.6806 and local surrogate differs from model probability by about 0.2941. It is approximate model reliance, not an exact probability decomposition or a causal intervention. |
| Does inference refit or reconstruct history? | Neither. It consumes engineered pre-race features and applies the fitted pipeline. Show complete/missing examples and 25 contract cases, including field order, unknown categories and invalid values. |
| What remains before official completion? | All-member access/review and walkthrough, final Kaggle restart/Run All and saved accessible version, accepted repository work, reviewed PDF/link agreement, official form submission and confirmation. |

## Course announcement and questions to clarify

The reviewed official October 7, 2026 announcement concerns **Milestone 1 office hours**: one 15-minute slot per team, correct team number, booking by 11:59 PM the preceding day, and preparation with specific questions. Multiple reservations may cancel both; a no-show affects later office-hours access. It does not establish the final evaluation date or change the submission deadline. The academic-mail search for a separate evaluation announcement found no match; CMS/evaluation slot details remain unconfirmed.

Questions for course clarification, if needed: which five metrics are intended by the deliverables wording, how to treat independently unverifiable historical starting-car cases, and whether the conservative mechanical-status treatment is acceptable. Both local LIME and SHAP are already provided, covering both XAI wordings. No question has been sent on the team's behalf.
