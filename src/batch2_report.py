"""Integrated report content derived from the complete run's measured evidence."""
from pathlib import Path
import json
import re
import pandas as pd

def write_integrated_report(out, destination):
    out, destination = Path(out), Path(destination)
    def read(name): return pd.read_csv(out/name)
    def get(name): return json.loads((out/name).read_text(encoding='utf-8'))
    def table(frame):
        def cell(v):
            if pd.isna(v): return '-'
            if isinstance(v,float): return f'{v:.4f}'
            return str(v).replace('|','/')
        return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
            '| '+' | '.join(['---']*len(frame.columns))+' |']+
            ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
    def fig(name,caption): return f'![{caption}](figures/{name}.png)\n\n{caption}'
    first = (out/'batch1_report.md').read_text(encoding='utf-8')
    first = first.split('## Statistical-model comparison')[0]
    first = first.replace('# ACL Milestone 1 — batch 1 report draft (work packages 1–10)',
        '# ACL Milestone 1 - Formula 1 Podium Prediction')
    first = re.sub(r'\*\*Status:\*\*[^\n]+',
        '**Status:** integrated implementation of work packages 1-20; team review, Kaggle verification and official submission remain pending.',first,count=1)
    first = first.replace('Batch 2 ablation will assess group contribution.','The ablation section below measures group contribution.')
    first = first.replace('test is reserved at 2022–2024','test is held out at 2022–2024 until the choice-freeze step')
    first = first.replace('../outputs/batch1/figures/','figures/')
    score = read('three_model_metrics.csv')
    short = {'Logistic regression':'LR','Random forest':'RF','Shallow neural network':'NN'}
    shown = score.copy()
    shown['model'] = shown.model.map(short)
    compact = shown[['model','split','accuracy','precision','recall','F1','ROC_AUC','PR_AUC','average_precision']].rename(columns={'average_precision':'AP'})
    frozen = get('model_choices_frozen.json')
    xai = get('xai_summary.json')
    ablation = read('feature_group_ablation.csv')
    orders = read('preprocessing_order_results.csv')
    order_index = orders.set_index('order')
    infer = read('inference_examples.csv')
    infer['model'] = infer.model.map(short)
    trials = read('nn_model_trials.csv')
    nn_params = frozen['models']['Shallow neural network']['parameters']
    test = score.loc[score.split.eq('test')]
    validation = score.loc[score.split.eq('validation')]
    selected = frozen['chosen_model_for_inference']
    selected_test = test.set_index('model').loc[selected]
    full_ap = ablation.loc[ablation.removed_group.eq('none'),'average_precision'].iloc[0]
    strongest = ablation.loc[ablation.removed_group.ne('none')].sort_values('AP_drop_from_full',ascending=False).iloc[0]
    worse = ablation.loc[ablation.removed_group.ne('none') & ablation.AP_drop_from_full.lt(0)]
    q1 = read('q1_front_row_rates.csv')
    q1leaders = q1.groupby('era').head(3)
    supported = q1.loc[q1.starts.ge(10)].groupby('era').head(3)
    contributions = read('local_shap_contributions.csv')
    positive = contributions.sort_values('contribution',ascending=False).head(3)
    negative = contributions.sort_values('contribution').head(3)
    def terms(frame): return ', '.join(f'{r.feature} ({r.contribution:+.4f})' for r in frame.itertuples())
    sections = [
        '## Team and contribution verification',
        'Team listed in the shared plan: Amr Hazem Mohamed Mohamed Hassan Elkotit; Mark Bassem Mourad Samoel Emira; Loay Mohamed Sabry Farag Ismail; Karim Ahmed Mahmoud Abdelmageed. Existing work-package ownership remains unchanged. Implementation assistance does not prove individual contributions or understanding; each member must review and defend the complete notebook. Contribution sign-off is pending.',
        '## Additional detail for Question 1 - unfiltered versus supported leaders',
        'The full ranking answers the question for every observed circuit; the supported display excludes fewer than 10 front-row starts to avoid presenting a tiny denominator as a stable best circuit. The unfiltered leaders below may be based on very few starts:',
        table(q1leaders[['era','circuit_name','starts','podiums','rate']]),
        'Supported leaders (at least 10 starts) are:',table(supported[['era','circuit_name','starts','podiums','rate']]),
        'The Indianapolis 500 event is excluded, not every race at Indianapolis Motor Speedway: the later United States Grand Prix on its road course remains eligible.',
        '## Three-model training, selection and comparison',
        'Logistic regression (LR) is the statistical baseline: regularized linear log-odds, transparent but limited for nonlinear interactions. The random forest (RF) averages 250 trees; it represents interactions but can overfit and is sensitive to distribution shifts. The shallow feed-forward neural network (NN) uses ReLU hidden units and an Adam optimizer with L2 regularization alpha = 0.01. It can represent nonlinearities but is sensitive to scaling, optimization choices, random initialization, and overfitting.',
        'LR candidates use C = 0.1, 1, 10; RF candidates use max_depth = 6 or 12 with min_samples_leaf = 3. NN trials use hidden layers (32,) and (32, 16), each trained for up to 150 epochs. Gradients use 2003-2019 only. Checkpoints at epoch 1 and every 5 epochs are selected by explicit 2020-2021 validation AP, with the earliest checkpoint retained on ties. No internal random validation split is used. Validation F1 selects thresholds from 0.05 to 0.95; the default 0.5 is not assumed optimal.',
        f'Selected NN configuration: hidden layers {nn_params["hidden_layer_sizes"]}, checkpoint epoch {nn_params["selected_epoch"]}. The complete training and validation histories are preserved in nn_training_history.csv. The two architectural trials are:',
        table(trials[['parameters','validation_AP']]),
        fig('09_nn_training','Training loss and chronological validation AP show whether additional training improves generalization rather than only fit to training data.'),
        'All three models use the same driver-race rows, the same 14 canonical input features, and matching encoded feature names. Each main pipeline fits median imputation with missing indicators, numeric scaling and one-hot categorical encoding on training only. Unknown categories map to all-zero one-hot values and remain a documented limitation. Class balancing/resampling is not introduced; the task is assessed with minority-class-aware precision-recall metrics.',
        'Metric definitions: accuracy is the fraction of correct binary decisions; precision is the fraction of predicted podiums that are podiums; recall is the fraction of actual podiums found; F1 is their harmonic mean at the locked threshold. ROC-AUC assesses ranking across false/true-positive tradeoffs. PR-AUC is trapezoidal area under precision versus recall. Average precision (AP) is a separate step-weighted precision-recall summary. AP drives validation selection; PR-AUC, F1 and ROC-AUC supply complementary evidence. Accuracy alone can conceal the podium minority. The brief lists three named metrics but also refers to five; all six plan metrics plus AP are provided, and the wording remains for course verification.',
        '### Decisions fixed before test evaluation',
        f'The selected inference model is {selected}, chosen using validation AP only. All models keep the full predeclared feature set and baseline preprocessing. Ablation and order experiments below are diagnostic, not a basis for test-driven changes. Fitted-pipeline hashes, row hashes, hyperparameters, NN checkpoint and thresholds were recorded before any test predictions. Choice manifest SHA-256: {frozen["choice_sha256"]}.',
        'The 2022-2024 results below were computed only after that freeze. They cannot be used to select a different winner, threshold, feature set or preprocessing order. Re-running an identical frozen pipeline checks reproducibility; it does not make the already revealed test set new evidence for further tuning.',
        table(pd.DataFrame([{'model':short[name],'threshold':values['threshold'],'settings':json.dumps(values['parameters'])} for name,values in frozen['models'].items()])),
        '### Training, validation and held-out test metrics',
        'LR = logistic regression; RF = random forest; NN = shallow neural network. Every metric is computed on the same split rows for all models. Training figures are descriptive and optimistically biased; validation was used for selection. Test figures are the locked-model generalization check.',
        table(compact),
        f'The validation-selected {selected} has test F1 {selected_test.F1:.4f}, ROC-AUC {selected_test.ROC_AUC:.4f}, PR-AUC {selected_test.PR_AUC:.4f}, and AP {selected_test.average_precision:.4f}. This reports performance without claiming perfect accuracy or choosing a new model after observing test results.',
        'RF and NN validation AP are nearly tied (0.739633 versus 0.739587); this small observed difference does not establish a statistically reliable superiority. RF has substantially higher training AP than validation AP, consistent with overfitting. On the held-out test, LR has higher F1/recall at its locked threshold while RF has higher AP/ROC-AUC; the choice of ranking metric versus thresholded decisions therefore changes the apparent ordering. These are descriptive comparisons, not permission to select a new winner using test results.',
        fig('08_statistical_model_curves','Statistical-model validation ROC and precision-recall curves from the initial batch.'),
        fig('logistic_regression_confusion','LR validation decisions at its validation-selected threshold.'),
        fig('random_forest_confusion','RF validation decisions at its validation-selected threshold.'),
        fig('17_frozen_test_curves','Locked three-model test ROC and precision-recall curves. The test prevalence line contextualizes PR performance.'),
    ]
    for name in frozen['models']:
        sections.append(fig('18_test_'+name.lower().replace(' ','_')+'_confusion',name+' test confusion matrix at the locked validation-selected threshold.'))
    sections += [
        '## Controlled experiments',
        '### Feature-group ablation',
        'The probe model is RF with its validation-selected settings and threshold fixed. Retrain once after dropping each of grid, qualifying, form, standings, context, and identity, using identical training/validation rows. Missing-indicator and one-hot columns are regenerated from each reduced training feature set. This measures group reliance in this model under correlation, not a causal contribution or the value of an isolated individual feature.',
        table(ablation[['removed_group','average_precision','F1','AP_drop_from_full']]),
        f'Full-model validation AP is {full_ap:.4f}. The largest AP reduction follows removing {strongest.removed_group} ({strongest.AP_drop_from_full:.4f}). '+
        ('Groups whose removal improved AP: '+', '.join(worse.removed_group)+'. Negative drops are retained rather than hidden.' if len(worse) else 'No group removal improved validation AP in this run.'),
        'Correlated feature groups can substitute for one another. A small ablation drop does not establish irrelevance; apparent improvement may reflect variance or overfitting. The main comparison keeps the full feature set fixed for this milestone rather than making additional validation-guided revisions after revealing test scores.',
        fig('10_feature_group_ablation','AP decrease from dropping each group; negative values mean the reduced model improved validation AP.'),
        '### Preprocessing-order experiment 1',
        'Compare median-impute then standardize against standardize observed values (NaNs ignored during fitting) then median-impute. Both use the same numeric/categorical inputs, fixed LR C and threshold, identical rows, and no numeric missing-indicator expansion for equal experimental dimension. The main three-model pipelines retain their indicators; these experimental scores are not silently substituted into that main comparison.',
        table(orders.loc[orders.experiment.astype(str).eq('1'),['order','average_precision','F1','ROC_AUC','PR_AUC']]),
        f'Observed validation AP changes from {order_index.loc["impute_then_scale","average_precision"]:.6f} to {order_index.loc["scale_then_impute","average_precision"]:.6f} when scaling first. F1 is unchanged at {order_index.loc["scale_then_impute","F1"]:.6f}: the small ranking change does not alter the locked-threshold decisions in this run. This is not evidence of a reliable improvement across future seasons.',
        'Imputing before scaling changes the fitted variance because imputed observations participate in scale statistics. Scaling first estimates moments from observed values; later median imputation does not retroactively change those moments. If a feature has no missing values, the two orders may be equivalent for that feature.',
        fig('11_preprocessing_order_1','Experiment 1 validation AP and training distribution of transformed Q1 gap.'),
        '### Preprocessing-order experiment 2',
        'After training-only median imputation, compare training 1st/99th-percentile clipping then scaling against scaling then clipping with bounds learned in the transformed training space. Same rows, LR settings, categories and threshold are held fixed. Clipping changes extremes; fitting scale before versus after clipping changes centering and variance. No quantile bounds are learned from validation/test.',
        table(orders.loc[orders.experiment.astype(str).eq('2'),['order','average_precision','F1','ROC_AUC','PR_AUC']]),
        f'Observed validation AP changes from {order_index.loc["impute_clip_scale","average_precision"]:.6f} to {order_index.loc["impute_scale_clip","average_precision"]:.6f} when scaling before clipping. Both orders have F1 {order_index.loc["impute_scale_clip","F1"]:.6f}. The range audit shows the clipping orders substantially reduce extreme transformed Q1 gaps relative to the unclipped experiment, while their modest AP difference does not justify a new post-test model choice.',
        fig('11_preprocessing_order_2','Experiment 2 validation AP and transformed training Q1-gap distributions; detailed range/moment effects are saved per feature.'),
        '### Effects on distributions, ranges and values',
        'The numeric transformation audit records each training feature’s transformed mean, standard deviation, minimum and maximum for every order. The following example illustrates the effect; the complete table is delivered alongside the report.',
        table(read('preprocessing_distribution_effects.csv').loc[lambda d:d.feature.eq('q1_gap_to_best')]),
        '## Model explainability',
        f'The explained model is the validation-selected {xai["explained_model"]}. Permutation importance shuffles each canonical input over validation data and measures AP drop, with five seeded repeats. Correlated inputs may share importance; shuffling can produce implausible combinations. The top permutation feature in this run is {xai["permutation_top_feature"]}.',
        fig('13_global_permutation','Global model reliance measured as validation AP loss after feature shuffling; bars may be negative and error bars show repeat SD, not confidence intervals.'),
        f'Global SHAP method: {xai["SHAP_method"]}. Forty-eight validation records are explained using a training-only background. Encoded/OHE and missing-indicator contributions are summed back to original input features; mean absolute contributions summarize that sample. The leading feature is {xai["SHAP_top_feature"]}. Background/sample choices and dependent features affect these attributions.',
        ('The initial TreeSHAP decomposition failed the external probability-additivity check (maximum error '+f'{xai["initial_TreeSHAP_max_additivity_error"]:.6f}'+'). The reported explanations therefore use permutation SHAP of the actual fitted prediction callable; the tolerance was not relaxed. This affects the explanation method, not the fitted classifier or its predictions.' if xai.get('initial_TreeSHAP_max_additivity_error',0) and xai['initial_TreeSHAP_max_additivity_error']>2e-5 else 'SHAP reconstruction was checked directly against the fitted pipeline probabilities.'),
        fig('14_global_shap','Global SHAP: mean absolute probability contribution on the stated validation sample, not a population-wide causal effect.'),
        f'For one complete validation record, predicted podium probability is {xai["local_predicted_probability"]:.4f}, compared with SHAP baseline {xai["local_SHAP_baseline_probability"]:.4f}. Largest positive contributions: {terms(positive)}. Most negative contributions: {terms(negative)}. Maximum additivity error across the explained records is {xai["SHAP_max_additivity_error"]:.3g}.',
        fig('15_local_shap','Local SHAP probability decomposition for the stated record; contributions plus baseline reconstruct the model probability.'),
        f'A local LIME explanation is also supplied to cover the final checklist’s LIME wording. It uses training data, explicit categorical decoding, and 3,000 perturbed neighbors. Surrogate local R-squared is {xai["local_LIME_surrogate_R2"]:.4f}; surrogate probability is {xai["local_LIME_surrogate_probability"]:.4f}. R-squared describes the surrogate’s neighborhood fit, not the classifier’s prediction accuracy. A low value limits confidence in the local approximation. Correlated/engineered variables can still form unrealistic perturbed combinations.',
        f'The surrogate differs from the classifier probability at this record by {abs(xai["local_LIME_surrogate_probability"]-xai["local_predicted_probability"]):.4f}. That material approximation error limits the strength of the local LIME interpretation; its weights must not be treated as an exact probability decomposition.',
        table(read('local_lime_weights.csv')),
        fig('16_local_lime','LIME weights describe a local surrogate; they are neither causal effects nor an exact SHAP-style probability decomposition.'),
        'All explanations describe how the model relies on the supplied data representation. They do not establish that changing a feature causes a podium. SHAP and LIME may differ because they use different neighborhoods, backgrounds and approximation objectives.',
        '## Inference function and missing-data handling',
        'podium_inference accepts a non-empty dictionary of the 14 engineered pre-race features. The caller must construct historical form/standings using only information before the cutoff. Missing numeric fields use the fitted training imputer; missing/unseen categorical identities use the encoder’s unknown-category behavior. Unknown or post-race keys are rejected. The function calls predict_proba without fitting, returns the locked threshold and binary prediction, and reports missing fields. Model hashes are asserted unchanged before/after the demonstrations.',
        'Complete and partially missing examples are evaluated for every model on the same validation record; the partial example omits Q1 gap, qualifying position and prior driver rank. Results are:',
        table(infer),
        'The probability is an uncalibrated model score, not a guaranteed real-world frequency. Each driver is classified independently, so predictions need not sum to exactly three podiums in a race; the brief requests binary classification rather than constrained race rankings.',
        '## Limitations and review items',
        'Historical cohort reconstruction remains uncertain for shared drives, some pit-lane starts and zero-lap start failures. The deterministic rule and withheld cases are documented; independent event-by-event ground-truth verification is not claimed. A modern 2003+ modeling window reduces some historical ambiguity but changes the modeled population. Recorded final grids are interpreted as pre-start positions; the CSVs are not timestamped snapshots of last-minute decisions.',
        'Question 2 controls starting position through broad grid strata; within-bin, driver, team, circuit and era confounding remain. Dual-nationality mapping exclusions and Indianapolis exclusions affect the comparison. Question 3 uses a conservative status taxonomy; vague retirement/damage/tyre/wing causes can undercount mechanical failures. Constructor renames reset identity history rather than being silently merged.',
        'Repeated validation selection can overfit the validation window; no uncertainty intervals or probability calibration are claimed. Validation/test include different sporting/technical eras. Explanations depend on representation, background, sample and perturbation assumptions. Feature evidence and ablations demonstrate association/model reliance, not causation.',
        'The brief’s metric-count wording and local-LIME checklist wording remain explicit verification items. Providing all six plan metrics plus AP and both local SHAP/LIME covers the named outputs but does not invent an official clarification.',
        '## Reproducibility and acceptance state',
        'Source hashes, semantic conversions, PK/FK integrity, unique driver-race keys, chronological source-event lineage, race-disjoint splits, feature whitelists, three-model row/feature contracts, inference immutability and SHAP additivity are checked in code. The notebook executes from a fresh local kernel. Independent perturbation checks verify that current/future outcomes or standings cannot change current/earlier features.',
        'Private GitHub review: [PR #2](https://github.com/amrelkotit/Acl_9th_project/pull/2). Shared Kaggle: [Acl_9th_project](https://www.kaggle.com/code/amrelkotit/acl-9th-project). The existing Kaggle draft was preserved as backup version 5. Upload permission was denied earlier; the updated notebook has not been synchronized or verified in Kaggle. All four members’ access, full review, and oral evaluation preparation remain unverified.',
        'Row 16 is not complete: this PDF is prepared for review, but the official submission form has not been submitted and no confirmation is claimed. Submission deadline in the course brief is 18 October 2026 at 11:59 PM; evaluation dates/slot reservation are separately announced. Do not treat a local successful run as completed team review or official submission.',
        '## Appendix - detailed feature evidence',
        'These larger panels supplement the compact overview and make the training-only feature evidence readable. Category plots show the ten most frequent categories; they do not establish incremental predictive benefit.',
    ]
    for number in range(1,5):
        sections.append(fig(f'12_feature_detail_{number}',f'Training feature-evidence panel {number}; candidate rationale and missingness are documented in feature_register.csv.'))
    sections += [
        '## Sources and methods references',
        '- MS1 Description (Project).pdf, pp. 1-8: official supplied milestone requirements.',
        '- Introduction for Formula 1 Beginners (Project).pdf, pp. 1-7: domain interpretation and historical cautions.',
        '- [Assignment-cited Kaggle/Ergast dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), downloaded snapshot hashes in source_manifest.json.',
        '- [scikit-learn leakage and preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).',
        '- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).',
        '- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) and [SHAP PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html).',
        '- [LIME tabular explanation documentation](https://lime-ml.readthedocs.io/en/latest/lime.html).',
    ]
    text = first+'\n\n'+'\n\n'.join(sections)+'\n'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(text,encoding='utf-8')
    # Every saved analysis plot must occur in the integrated report, not only selected successes.
    referenced = {Path(p).name for p in re.findall(r'!\[[^]]*\]\(([^)]*)\)',text)}
    actual = {p.name for p in (out/'figures').glob('*.png')}
    assert actual==referenced, f'Figure coverage mismatch: missing={actual-referenced}, unknown={referenced-actual}'
    return text
