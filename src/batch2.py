"""Work packages 11–20; embedded verbatim into the self-contained notebook."""
from copy import deepcopy
from datetime import datetime, timezone
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
import shap
from lime.lime_tabular import LimeTabularExplainer
if 'make_preprocessor' not in globals():
    from batch1 import (Path, np, pd, plt, json, hashlib, joblib, sklearn, SEED, KEY,
        clone, Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder,
        average_precision_score, f1_score, metrics, make_preprocessor, save_figure, write_json,
        RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)

class QuantileClipper(TransformerMixin, BaseEstimator):
    """Winsorization bounds learned only from the training input at this stage."""
    def __init__(self, lower=.01, upper=.99):
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None):
        X = np.asarray(X,dtype=float)
        self.lower_ = np.nanquantile(X,self.lower,axis=0)
        self.upper_ = np.nanquantile(X,self.upper,axis=0)
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):
        return np.clip(np.asarray(X,dtype=float),self.lower_,self.upper_)

def choose_threshold(y, p):
    candidates = np.linspace(.05,.95,91)
    scores = [f1_score(y,p>=t,zero_division=0) for t in candidates]
    best = sorted(zip(candidates,scores),key=lambda x:(-x[1],abs(x[0]-.5),x[0]))[0]
    return float(best[0])

def train_shallow_network(model, numeric, categorical, models, out):
    features = numeric+categorical
    tr = model.loc[model.split.eq('train')]
    va = model.loc[model.split.eq('validation')]
    preprocessor = make_preprocessor(numeric,categorical)
    ztr = preprocessor.fit_transform(tr[features])
    zva = preprocessor.transform(va[features])
    assert tuple(preprocessor.get_feature_names_out()) == tuple(models['Random forest']['pipeline'].named_steps['preprocess'].get_feature_names_out())
    history, attempts, best = [], [], None
    # Library-built FFNNs, with one and two hidden layers. All gradients use training rows only.
    # Checkpoints/hyperparameters use the explicit chronological validation set, not an internal random split.
    with threadpool_limits(limits=2):
        for layers in [(32,),(32,16)]:
            nn = MLPClassifier(hidden_layer_sizes=layers,activation='relu',solver='adam',
                alpha=.01,batch_size=128,learning_rate_init=.001,early_stopping=False,
                random_state=SEED,shuffle=True,max_iter=1)
            best_trial = None
            for epoch in range(1,151):
                nn.partial_fit(ztr,tr.podium.to_numpy(),classes=np.array([0,1]))
                if epoch==1 or epoch%5==0:
                    ptr = nn.predict_proba(ztr)[:,1]
                    pva = nn.predict_proba(zva)[:,1]
                    score = average_precision_score(va.podium,pva)
                    history.append({'architecture':str(layers),'epoch':epoch,'training_loss':nn.loss_,
                        'training_AP':average_precision_score(tr.podium,ptr),'validation_AP':score})
                    if best_trial is None or score>best_trial['validation_AP']:
                        best_trial = {'estimator':deepcopy(nn),'epoch':epoch,'validation_AP':score}
            params = {'hidden_layer_sizes':list(layers),'alpha':.01,'solver':'adam',
                'learning_rate_init':.001,'selected_epoch':best_trial['epoch'],'max_checked_epoch':150,
                'checkpoint_selection':'highest validation AP; earliest checkpoint on ties'}
            attempts.append({'model':'Shallow neural network','parameters':json.dumps(params),
                             'validation_AP':best_trial['validation_AP']})
            if best is None or best_trial['validation_AP']>best['validation_AP']:
                best = {**best_trial,'parameters':params}
    pipe = Pipeline([('preprocess',preprocessor),('model',best['estimator'])])
    pva = pipe.predict_proba(va[features])[:,1]
    threshold = choose_threshold(va.podium,pva)
    models['Shallow neural network'] = {'pipeline':pipe,'parameters':best['parameters'],
        'validation_AP':best['validation_AP'],'threshold':threshold}
    records = []
    for name,info in models.items():
        for split,frame in [('train',tr),('validation',va)]:
            p = info['pipeline'].predict_proba(frame[features])[:,1]
            records.append({'model':name,'split':split,'threshold':info['threshold'],**metrics(frame.podium,p,info['threshold'])})
    history = pd.DataFrame(history)
    history.to_csv(Path(out)/'nn_training_history.csv',index=False)
    attempts = pd.DataFrame(attempts)
    attempts.to_csv(Path(out)/'nn_model_trials.csv',index=False)
    comparison = pd.DataFrame(records)
    comparison.to_csv(Path(out)/'three_model_train_validation_metrics.csv',index=False)
    joblib.dump(pipe,Path(out)/'shallow_neural_network.joblib')
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for architecture,g in history.groupby('architecture'):
        axes[0].plot(g.epoch,g.training_loss,label=architecture)
        axes[1].plot(g.epoch,g.training_AP,label=architecture+' training',linestyle='--')
        axes[1].plot(g.epoch,g.validation_AP,label=architecture+' validation')
    axes[0].set(title='NN training loss (training gradients only)',xlabel='Epoch',ylabel='Regularized loss')
    axes[1].set(title='Chronological validation checkpoints',xlabel='Epoch',ylabel='Average precision')
    for ax in axes: ax.legend(fontsize=8)
    save_figure(fig,out,'09_nn_training')
    print(comparison.to_string(index=False))
    return models,comparison,attempts

def feature_group_ablation(model, numeric, categorical, registry, models, out):
    tr,va = [model.loc[model.split.eq(s)] for s in ['train','validation']]
    info = models['Random forest']
    records = []
    groups = sorted(registry.group.unique())
    for removed in ['none']+groups:
        omitted = set(registry.loc[registry.group.eq(removed),'feature'])
        nums = [c for c in numeric if c not in omitted]
        cats = [c for c in categorical if c not in omitted]
        pipe = Pipeline([('preprocess',make_preprocessor(nums,cats)),
                         ('model',clone(info['pipeline'].named_steps['model']))])
        pipe.fit(tr[nums+cats],tr.podium)
        p = pipe.predict_proba(va[nums+cats])[:,1]
        records.append({'removed_group':removed,'removed_features':'; '.join(sorted(omitted)),
            'validation_rows':len(va),'fixed_threshold':info['threshold'],**metrics(va.podium,p,info['threshold'])})
    results = pd.DataFrame(records)
    baseline = results.loc[results.removed_group.eq('none'),'average_precision'].iloc[0]
    results['AP_drop_from_full'] = baseline-results.average_precision
    results.to_csv(Path(out)/'feature_group_ablation.csv',index=False)
    fig,ax = plt.subplots(figsize=(9,4))
    shown = results.loc[results.removed_group.ne('none')].sort_values('AP_drop_from_full')
    ax.barh(shown.removed_group,shown.AP_drop_from_full,color=np.where(shown.AP_drop_from_full.ge(0),'#0f766e','#b45309'))
    ax.axvline(0,color='#64748b')
    ax.set(title='Remove one group and retrain the fixed random forest',xlabel='Validation AP drop (negative = ablation improved AP)')
    save_figure(fig,out,'10_feature_group_ablation')
    print(results[['removed_group','average_precision','F1','AP_drop_from_full']].to_string(index=False))
    return results

def preprocessing_order_experiments(model, numeric, categorical, models, out):
    tr,va = [model.loc[model.split.eq(s)] for s in ['train','validation']]
    features = numeric+categorical
    lr = clone(models['Logistic regression']['pipeline'].named_steps['model'])
    threshold = models['Logistic regression']['threshold']
    impute = lambda:SimpleImputer(strategy='median',keep_empty_features=True)
    orders = [
        ('1','impute_then_scale',[('impute',impute()),('scale',StandardScaler())]),
        ('1','scale_then_impute',[('scale',StandardScaler()),('impute',impute())]),
        ('2','impute_clip_scale',[('impute',impute()),('clip',QuantileClipper()),('scale',StandardScaler())]),
        ('2','impute_scale_clip',[('impute',impute()),('scale',StandardScaler()),('clip',QuantileClipper())])]
    results,distributions,transforms = [],[],{}
    for experiment,order,steps in orders:
        numeric_pipe = Pipeline(steps)
        prep = ColumnTransformer([('numeric',numeric_pipe,numeric),
            ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)],sparse_threshold=0)
        pipe = Pipeline([('preprocess',prep),('model',clone(lr))])
        pipe.fit(tr[features],tr.podium)
        p = pipe.predict_proba(va[features])[:,1]
        results.append({'experiment':experiment,'order':order,'fixed_threshold':threshold,
                        **metrics(va.podium,p,threshold)})
        z = pipe.named_steps['preprocess'].named_transformers_['numeric'].transform(tr[numeric])
        transforms[order] = z
        for i,c in enumerate(numeric):
            distributions.append({'experiment':experiment,'order':order,'feature':c,
                'mean':np.mean(z[:,i]),'std':np.std(z[:,i]),'min':np.min(z[:,i]),'max':np.max(z[:,i])})
    results,distributions = pd.DataFrame(results),pd.DataFrame(distributions)
    results.to_csv(Path(out)/'preprocessing_order_results.csv',index=False)
    distributions.to_csv(Path(out)/'preprocessing_distribution_effects.csv',index=False)
    for experiment in ['1','2']:
        subset = results.loc[results.experiment.eq(experiment)]
        fig,axes = plt.subplots(1,2,figsize=(12,4))
        axes[0].bar(subset.order,subset.average_precision,color=['#64748b','#0f766e'])
        axes[0].set(title='Experiment '+experiment+': validation AP',ylabel='Average precision',ylim=(0,1))
        feature = 'q1_gap_to_best'
        for order in subset.order:
            axes[1].hist(transforms[order][:,numeric.index(feature)],bins=35,alpha=.5,label=order,density=True)
        axes[1].set(title='Effect on training '+feature,xlabel='Transformed value',ylabel='Density')
        axes[1].legend(fontsize=8)
        axes[0].tick_params(axis='x',labelsize=8)
        save_figure(fig,out,'11_preprocessing_order_'+experiment)
    print(results.to_string(index=False))
    return results,distributions

def feature_detail_plots(model, numeric, categorical, out):
    """Readable report appendix panels supplement the dense batch-1 overview."""
    tr = model.loc[model.split.eq('train')]
    features = numeric+categorical
    names = []
    for start in range(0,len(features),4):
        fig,axes = plt.subplots(2,2,figsize=(9,7))
        for ax,c in zip(axes.ravel(),features[start:start+4]):
            if c in numeric and tr[c].nunique()>8:
                rates = tr.groupby(pd.qcut(tr[c],5,duplicates='drop'),observed=True).podium.agg(['mean','size'])
            else:
                rates = tr.groupby(c,observed=True).podium.agg(['mean','size']).sort_values('size',ascending=False).head(10)
            ax.bar(range(len(rates)),rates['mean'])
            ax.set_xticks(range(len(rates)),[str(i) for i in rates.index],rotation=45,ha='right',fontsize=8)
            ax.set(title=c,ylabel='Training podium rate',ylim=(0,1))
        for ax in axes.ravel()[len(features[start:start+4]):]: ax.axis('off')
        name = f'12_feature_detail_{start//4+1}'
        save_figure(fig,out,name)
        names.append(name)
    return names

def podium_inference(record, models, numeric, categorical, model_name):
    """Predict from engineered PRE-RACE features without learning from the input."""
    if model_name not in models:
        raise ValueError(f'Unknown model: {model_name}')
    features = numeric+categorical
    if not isinstance(record,dict) or not record:
        raise ValueError('Supply a non-empty dictionary of engineered pre-race features.')
    extra = set(record)-set(features)
    if extra:
        raise ValueError(f'Unknown/outcome fields are forbidden: {sorted(extra)}')
    row = pd.DataFrame([{c:record.get(c,np.nan) for c in features}])
    missing = [c for c in features if pd.isna(row.at[0,c])]
    for c in numeric:
        row[c] = pd.to_numeric(row[c],errors='raise').astype(float)
        if np.isinf(row[c]).any(): raise ValueError(f'Infinite value: {c}')
    for c in categorical:
        row[c] = row[c].map(lambda x:'__UNKNOWN__' if pd.isna(x) else str(x))
    info = models[model_name]
    p = float(info['pipeline'].predict_proba(row[features])[0,1])
    return {'model':model_name,'podium_probability':p,'threshold':float(info['threshold']),
        'predicted_podium':int(p>=info['threshold']),'missing_features':missing,
        'input_contract':'engineered pre-race features; caller must assemble history before cutoff',
        'probability_note':'Model score; probability calibration is not established.'}

def inference_demonstration(model, numeric, categorical, models, out):
    features = numeric+categorical
    va = model.loc[model.split.eq('validation')]
    complete = va.loc[va[features].notna().all(axis=1)].iloc[0]
    full_record = complete[features].to_dict()
    partial = full_record.copy()
    for c in ['q1_gap_to_best','qualifying_position','driver_standings_rank_before']:
        partial.pop(c)
    examples = []
    before = {n:joblib.hash(i['pipeline']) for n,i in models.items()}
    for name in models:
        for kind,record in [('complete',full_record),('partially_missing',partial)]:
            examples.append({'record_kind':kind,**podium_inference(record,models,numeric,categorical,name)})
    after = {n:joblib.hash(i['pipeline']) for n,i in models.items()}
    assert before==after, 'Inference must not modify fitted pipelines.'
    write_json(Path(out)/'inference_examples.json',{'raceId':int(complete.raceId),
        'driverId':int(complete.driverId),'complete_record':full_record,'partial_record':partial,'predictions':examples})
    # Outcome keys must never be silently accepted at inference time.
    for forbidden in ['positionOrder','podium','statusId','laps','points']:
        try:
            podium_inference({**full_record,forbidden:1},models,numeric,categorical,next(iter(models)))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Inference accepted forbidden outcome field {forbidden}')
    summary = pd.DataFrame([{k:v for k,v in e.items() if k in ['record_kind','model','podium_probability','threshold','predicted_podium']} for e in examples])
    summary.to_csv(Path(out)/'inference_examples.csv',index=False)
    print(summary.to_string(index=False))
    return summary,full_record

def explain_validation_model(model, numeric, categorical, models, out, local_record):
    features = numeric+categorical
    tr,va = [model.loc[model.split.eq(s)] for s in ['train','validation']]
    chosen_name = max(models,key=lambda n:models[n]['validation_AP'])
    pipe = models[chosen_name]['pipeline']
    permutation = permutation_importance(pipe,va[features],va.podium,scoring='average_precision',
        n_repeats=5,random_state=SEED,n_jobs=1)
    importance = pd.DataFrame({'feature':features,'AP_drop_mean':permutation.importances_mean,
                              'AP_drop_std':permutation.importances_std}).sort_values('AP_drop_mean')
    importance.to_csv(Path(out)/'global_permutation_importance.csv',index=False)
    fig,ax = plt.subplots(figsize=(9,6))
    ax.barh(importance.feature,importance.AP_drop_mean,xerr=importance.AP_drop_std,color='#0f766e')
    ax.axvline(0,color='#64748b')
    ax.set(title=chosen_name+': validation permutation importance',xlabel='AP drop after shuffling; error bars = repeat SD')
    save_figure(fig,out,'13_global_permutation')
    prep = pipe.named_steps['preprocess']
    estimator = pipe.named_steps['model']
    ztr = prep.transform(tr[features])
    local = pd.DataFrame([local_record],columns=features)
    sample = pd.concat([local,va[features].sample(n=47,random_state=SEED)],ignore_index=True)
    zsample = prep.transform(sample)
    rng = np.random.default_rng(SEED)
    background = ztr[rng.choice(len(ztr),size=32,replace=False)]
    encoded_names = list(prep.get_feature_names_out())
    tree_additivity_error = None
    if chosen_name=='Random forest':
        explainer = shap.TreeExplainer(estimator,data=background,feature_perturbation='interventional',
                                      model_output='probability',feature_names=encoded_names)
        explanation = explainer(zsample,check_additivity=True)
        phi = explanation.values[:,:,1] if explanation.values.ndim==3 else explanation.values
        base = explanation.base_values[:,1] if np.asarray(explanation.base_values).ndim==2 else explanation.base_values
        method = 'TreeSHAP, interventional probability output; 32 training background rows'
        tree_reconstructed = np.asarray(base).reshape(-1)+phi.sum(axis=1)
        tree_additivity_error = float(np.max(abs(tree_reconstructed-estimator.predict_proba(zsample)[:,1])))
    if chosen_name!='Random forest' or tree_additivity_error>2e-5:
        # Some TreeSHAP/runtime combinations fail probability additivity despite matching
        # internal forest predictions. Explain the actual callable instead; never relax
        # the acceptance tolerance or silently present the inconsistent decomposition.
        explainer = shap.Explainer(lambda z:estimator.predict_proba(z)[:,1],
            shap.maskers.Independent(background),algorithm='permutation',seed=SEED,feature_names=encoded_names)
        with threadpool_limits(limits=2):
            explanation = explainer(zsample,max_evals=2*(2*zsample.shape[1]+1),silent=True)
        phi,base = explanation.values,np.asarray(explanation.base_values)
        method = 'Permutation SHAP, two permutation cycles per row; 32 training background rows'
        if tree_additivity_error is not None:
            method += '; fallback after TreeSHAP failed the external probability-additivity check'
    probability = pipe.predict_proba(sample[features])[:,1]
    reconstructed = np.asarray(base).reshape(-1)+phi.sum(axis=1)
    assert np.allclose(reconstructed,probability,atol=2e-5), 'SHAP output must reconstruct predicted probabilities.'
    pd.DataFrame({'probability':probability,'SHAP_baseline':np.asarray(base).reshape(-1),
        'SHAP_contribution_sum':phi.sum(axis=1),'reconstructed_probability':reconstructed,
        'absolute_error':abs(reconstructed-probability)}).to_csv(Path(out)/'shap_additivity_audit.csv',index=False)
    # Sum encoded/OHE and missing-indicator contributions back to original feature names.
    encoded_to_original = []
    for name in encoded_names:
        suffix = name.split('__',1)[1]
        if suffix.startswith('missingindicator_'):
            original = suffix.removeprefix('missingindicator_')
        elif name.startswith('category__'):
            original = next(c for c in categorical if suffix.startswith(c+'_'))
        else:
            original = suffix
        assert original in features,name
        encoded_to_original.append(original)
    grouped = np.column_stack([phi[:,np.array(encoded_to_original)==c].sum(axis=1) for c in features])
    assert np.allclose(grouped.sum(axis=1),phi.sum(axis=1))
    global_shap = pd.DataFrame({'feature':features,'mean_absolute_probability_contribution':np.abs(grouped).mean(axis=0)}).sort_values('mean_absolute_probability_contribution')
    global_shap.to_csv(Path(out)/'global_shap_importance.csv',index=False)
    pd.DataFrame(grouped,columns=features).to_csv(Path(out)/'sample_shap_contributions.csv',index=False)
    fig,ax = plt.subplots(figsize=(9,6))
    ax.barh(global_shap.feature,global_shap.mean_absolute_probability_contribution,color='#0f766e')
    ax.set(title=chosen_name+': global SHAP on 48 validation records',xlabel='Mean absolute contribution to podium probability')
    save_figure(fig,out,'14_global_shap')
    local_shap = shap.Explanation(values=grouped[0],base_values=float(np.asarray(base).reshape(-1)[0]),
        data=local.iloc[0].to_numpy(),feature_names=features)
    shap.plots.waterfall(local_shap,max_display=12,show=False)
    save_figure(plt.gcf(),out,'15_local_shap')
    pd.DataFrame({'feature':features,'contribution':grouped[0]}).to_csv(Path(out)/'local_shap_contributions.csv',index=False)
    # LIME uses the original feature grain with explicit categorical code/decode; it does not
    # independently toggle one-hot bits into impossible multi-team identities.
    medians = tr[numeric].median()
    lime_train = tr[features].copy()
    lime_local = local[features].copy()
    for c in numeric:
        lime_train[c] = lime_train[c].fillna(medians[c])
        lime_local[c] = lime_local[c].fillna(medians[c])
    category_names,category_maps = {},{}
    for c in categorical:
        levels = sorted(tr[c].unique().tolist())
        if local_record[c] not in levels: levels.append(local_record[c])
        mapping = {s:i for i,s in enumerate(levels)}
        category_maps[c] = levels
        category_names[features.index(c)] = levels
        lime_train[c] = lime_train[c].map(mapping)
        lime_local[c] = lime_local[c].map(mapping)
    category_names[features.index('grid_zero_flag')] = ['normal grid','zero grid']
    def lime_predict(z):
        frame = pd.DataFrame(z,columns=features)
        for c in categorical:
            levels = category_maps[c]
            frame[c] = [levels[int(np.clip(round(v),0,len(levels)-1))] for v in frame[c]]
        for c in ['grid_position','qualifying_position','round','grid_zero_flag']:
            frame[c] = frame[c].round()
        return pipe.predict_proba(frame[features])
    lime_explainer = LimeTabularExplainer(lime_train.to_numpy(dtype=float),feature_names=features,
        class_names=['Not podium','Podium'],categorical_features=sorted(category_names),
        categorical_names=category_names,random_state=SEED,discretize_continuous=True)
    local_lime = lime_explainer.explain_instance(lime_local.iloc[0].to_numpy(dtype=float),lime_predict,
        labels=(1,),num_features=10,num_samples=3000)
    lime_terms = pd.DataFrame(local_lime.as_list(label=1),columns=['condition','weight'])
    lime_terms.to_csv(Path(out)/'local_lime_weights.csv',index=False)
    save_figure(local_lime.as_pyplot_figure(label=1),out,'16_local_lime')
    info = {'explained_model':chosen_name,'SHAP_method':method,'validation_sample_rows':len(sample),
        'initial_TreeSHAP_max_additivity_error':tree_additivity_error,
        'background_source':'training rows only','SHAP_max_additivity_error':float(np.max(abs(reconstructed-probability))),
        'local_predicted_probability':float(probability[0]),'local_SHAP_baseline_probability':float(np.asarray(base).reshape(-1)[0]),
        'local_LIME_surrogate_R2':float(local_lime.score),'local_LIME_surrogate_probability':float(local_lime.local_pred[0]),
        'permutation_top_feature':str(importance.iloc[-1].feature),'SHAP_top_feature':str(global_shap.iloc[-1].feature),
        'limitation':'Explanations describe model reliance, not causal effects. Correlated features and artificial perturbations can distort attribution. SHAP is sample/background dependent; LIME is a local approximation, not an additive probability decomposition.'}
    write_json(Path(out)/'xai_summary.json',info)
    print(json.dumps(info,indent=2))
    return info,importance,global_shap,lime_terms

def freeze_model_choices(model, numeric, categorical, models, out):
    """Called after all validation-driven experiments, before any test predictions."""
    features = numeric+categorical
    contents = {'features':features,'primary_selection_metric':'validation average precision',
        'chosen_model_for_inference':max(models,key=lambda n:models[n]['validation_AP']),
        'models':{n:{'parameters':i['parameters'],'threshold':i['threshold'],
                    'fitted_pipeline_hash':joblib.hash(i['pipeline'])} for n,i in models.items()},
        'row_hashes':{s:hashlib.sha256(model.loc[model.split.eq(s),KEY].to_csv(index=False).encode()).hexdigest()
                      for s in ['train','validation','test_reserved']},
        'decision':'Keep the complete predeclared feature set and baseline preprocessing for the main three-model comparison. Ablation/order experiments are diagnostic; no test-driven adoption.',
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'seed':SEED}
    digest = hashlib.sha256(json.dumps(contents,sort_keys=True,default=str).encode()).hexdigest()
    lock = Path(out)/'model_choices_frozen.json'
    if lock.exists() and (Path(out)/'three_model_metrics.csv').exists():
        old = json.loads(lock.read_text(encoding='utf-8'))
        if old['choice_sha256']!=digest:
            raise RuntimeError('Model choices changed after test evaluation. Stop and document the change; do not silently retune against the revealed test set.')
    contents['choice_sha256'] = digest
    contents['frozen_before_test_predictions'] = True
    write_json(lock,contents)
    print('Frozen model choices:',digest)
    return contents

def evaluate_frozen_models(model, numeric, categorical, models, frozen, out):
    features = numeric+categorical
    records,predictions = [],[]
    for name,info in models.items():
        assert joblib.hash(info['pipeline'])==frozen['models'][name]['fitted_pipeline_hash']
        for split in ['train','validation','test_reserved']:
            frame = model.loc[model.split.eq(split)]
            p = info['pipeline'].predict_proba(frame[features])[:,1]
            records.append({'model':name,'split':'test' if split=='test_reserved' else split,
                'threshold':info['threshold'],**metrics(frame.podium,p,info['threshold'])})
            if split=='test_reserved':
                predictions.append(frame[KEY+['podium']].assign(model=name,probability=p,threshold=info['threshold']))
    result = pd.DataFrame(records)
    result.to_csv(Path(out)/'three_model_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(Path(out)/'frozen_test_predictions.csv',index=False)
    test = model.loc[model.split.eq('test_reserved')]
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    for name,info in models.items():
        p = info['pipeline'].predict_proba(test[features])[:,1]
        RocCurveDisplay.from_predictions(test.podium,p,name=name,ax=axes[0])
        PrecisionRecallDisplay.from_predictions(test.podium,p,name=name,ax=axes[1])
        fig_cm,ax_cm = plt.subplots(figsize=(4,3.5))
        ConfusionMatrixDisplay.from_predictions(test.podium,p>=info['threshold'],ax=ax_cm,
            display_labels=['Not podium','Podium'],colorbar=False)
        ax_cm.set_title(name+' — locked test')
        save_figure(fig_cm,out,'18_test_'+name.lower().replace(' ','_')+'_confusion')
    axes[0].set_title('Test ROC (choices frozen before scoring)')
    axes[1].axhline(test.podium.mean(),color='#64748b',linestyle='--',label='Test class prevalence')
    axes[1].legend(fontsize=7)
    axes[1].set_title('Test precision–recall (no retuning)')
    save_figure(fig,out,'17_frozen_test_curves')
    print(result.to_string(index=False))
    return result

def acceptance_checks_batch2(model, numeric, categorical, models, out):
    features = numeric+categorical
    assert len(models)==3
    assert not model.duplicated(KEY).any()
    assert set(model.podium.unique())=={0,1}
    contracts = [tuple(i['pipeline'].named_steps['preprocess'].get_feature_names_out()) for i in models.values()]
    assert all(c==contracts[0] for c in contracts)
    group_results = pd.read_csv(Path(out)/'feature_group_ablation.csv')
    assert len(group_results)==7
    order_results = pd.read_csv(Path(out)/'preprocessing_order_results.csv')
    assert order_results.groupby('experiment').size().eq(2).all()
    scored = pd.read_csv(Path(out)/'three_model_metrics.csv')
    assert len(scored)==9 and set(scored.split)=={'train','validation','test'}
    names = ['accuracy','precision','recall','F1','ROC_AUC','PR_AUC','average_precision']
    assert np.isfinite(scored[names].to_numpy()).all()
    assert scored[names].ge(0).all().all() and scored[names].le(1).all().all()
    assert not set(features)&{'podium','positionOrder','points','laps','statusId','milliseconds','rank'}
    results = {'same_rows_features_three_models':True,'unique_driver_race_keys':True,
        'six_feature_groups_ablated':True,'two_order_experiments_two_variants_each':True,
        'three_models_all_six_metrics_all_three_splits':True,'model_choices_locked_before_test':True,
        'global_SHAP_permutation_local_SHAP_LIME':True,'complete_and_missing_inference':True,
        'Kaggle_execution_verified':False,'all_team_members_reviewed':False,'official_submission_completed':False,
        'historical_cohort_ground_truth_fully_resolved':False,
        'notes':'Implemented checks pass. Kaggle synchronization remains blocked by the prior upload denial; team access/review and historical source ambiguities require verification. No perfect-score guarantee.'}
    write_json(Path(out)/'batch2_acceptance.json',results)
    print(json.dumps(results,indent=2))
    return results
