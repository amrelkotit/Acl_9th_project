"""Rows 21–30: acceptance evidence only. No model fitting or selection."""
from pathlib import Path
import json
import hashlib
if 'podium_inference' not in globals():
    from batch1 import np, pd, joblib, PK, FK, KEY, markdown_table, file_hash, write_json
    from batch2 import podium_inference

def accuracy_target_audit(model, models, out, destination):
    """Assess the user goal against frozen results, including a no-podium baseline."""
    recorded = pd.read_csv(Path(out)/'three_model_metrics.csv')
    rows = []
    for row in recorded.to_dict('records'):
        split = 'test_reserved' if row['split']=='test' else row['split']
        frame = model.loc[model.split.eq(split)]
        prevalence = float(frame.podium.mean())
        majority_accuracy = 1-prevalence
        assert majority_accuracy>.5
        rows.append({**row,'rows':len(frame),'podium_prevalence':prevalence,
            'always_no_podium_accuracy':majority_accuracy,
            'accuracy_lift_percentage_points':100*(row['accuracy']-majority_accuracy),
            'target_accuracy':.80,'meets_user_accuracy_target':row['accuracy']>=.80})
    results = pd.DataFrame(rows)
    results.to_csv(destination/'accuracy_target_audit.csv',index=False)
    test = results.loc[results.split.eq('test')]
    selected = max(models,key=lambda name:models[name]['validation_AP'])
    info = {'target':.80,'target_source':'User instruction, 14 October 2026; ordinary binary accuracy',
        'selection_metric_unchanged':'validation average precision',
        'selected_model':selected,'selected_test_accuracy':float(test.set_index('model').loc[selected,'accuracy']),
        'all_three_test_models_meet_target':bool(test.meets_user_accuracy_target.all()),
        'always_no_podium_test_accuracy':float(test.always_no_podium_accuracy.iloc[0]),
        'interpretation':'Historical held-out performance, not a guarantee for future races. The majority baseline already exceeds 80%; report podium precision/recall/F1 and ranking metrics as well. No changes are chosen using revealed test scores.'}
    assert info['all_three_test_models_meet_target']
    write_json(destination/'accuracy_target_summary.json',info)
    return results,info

def complete_feature_evidence(numeric, categorical, out, destination):
    register = pd.read_csv(Path(out)/'feature_register.csv')
    ablation = pd.read_csv(Path(out)/'feature_group_ablation.csv').set_index('removed_group')
    assert set(register.feature)==set(numeric+categorical)
    register['calculation'] = register.pre_race_validity
    register['availability_at_cutoff'] = register.group.map({
        'grid':'Actual starting grid after penalties and before the start; retrospective CSV is not a timestamped snapshot.',
        'qualifying':'Current qualifying session completed before the main race.',
        'form':'Source events strictly earlier than the main race; shift before rolling.',
        'standings':'Strictly earlier source race within the same season; reset at season start.',
        'context':'Scheduled round and circuit identity known before the main race.',
        'identity':'Entrant constructor identity known before the main race; reconstructed-starter population.'})
    register['group_ablation_validation_AP_drop'] = register.group.map(ablation.AP_drop_from_full)
    register['detail_plot'] = [f'12_feature_detail_{(numeric+categorical).index(c)//4+1}.png' for c in register.feature]
    register['evidence'] = '04_feature_evidence.png; '+register.detail_plot+'; feature_group_ablation.csv (validation only)'
    register['interpretation_limit'] = 'Training association and group reliance, not causation or an isolated individual-feature effect. Historical cutoff/cohort assumptions require team review.'
    assert register[['calculation','availability_at_cutoff','evidence']].notna().all().all()
    assert not register.evidence.str.contains('pending',case=False).any()
    register.to_csv(destination/'feature_register_final.csv',index=False)
    # Retain the original batch-1 contract as a checkpoint; provide one explicit final contract.
    contract = json.loads((Path(out)/'model_feature_contract.json').read_text())
    frozen = json.loads((Path(out)/'model_choices_frozen.json').read_text())
    contract.update({'phase':'final frozen comparison','test_evaluated':True,
        'test_years':[2022,2023,2024],'models':list(frozen['models']),
        'choice_sha256':frozen['choice_sha256'],
        'checkpoint_note':'model_feature_contract.json is the initial statistical-model checkpoint. This final contract supersedes its test-reserved flag.'})
    write_json(destination/'model_feature_contract_final.json',contract)
    return register

def audit_cleaning_and_joins(model, data_dir, out, destination):
    """Recheck the raw/cleaned files, documented key graph and row conservation."""
    raw,cleaned,counts,keys = {},{},{},[]
    for name,pk in PK.items():
        raw[name] = pd.read_csv(Path(data_dir)/(name+'.csv'),dtype=str,keep_default_na=False)
        cleaned[name] = pd.read_csv(Path(out)/'cleaned'/(name+'.csv'),keep_default_na=False)
        before,after = raw[name],cleaned[name]
        assert len(before)==len(after),name
        assert not before[pk].isin(['',r'\N']).any().any(),name
        assert not before.duplicated(pk).any(),name
        assert not after.astype(str).isin([r'\N']).any().any(),name
        keys.append({'table':name,'primary_key':' + '.join(pk),'rows':len(before),
                     'null_primary_keys':0,'duplicate_primary_keys':0,'literal_sentinels_remaining':0})
        counts[name] = {'stage':name+': raw -> typed working copy','input_rows':len(before),
                       'output_rows':len(after),'row_difference':len(after)-len(before)}
    foreign_keys=[]
    for child,relations in FK.items():
        for column,parent,parent_column in relations:
            values=raw[child].loc[~raw[child][column].isin(['',r'\N']),column]
            orphan=int((~values.isin(raw[parent][parent_column])).sum())
            assert orphan==0,(child,column)
            foreign_keys.append({'child':child,'column':column,'parent':parent,'orphan_rows':orphan})
    conversions=pd.read_csv(Path(out)/'type_conversion_audit.csv')
    assert conversions.new_missing.sum()==0
    assert {'numeric','date','duration_seconds'}<=set(conversions.type)
    for row in conversions.to_dict('records'):
        values=cleaned[row['table']][row['column']].replace('',np.nan)
        if row['type']=='date':
            assert pd.to_datetime(values,errors='coerce').notna().equals(values.notna())
        elif row['type']=='duration_seconds':
            seconds=cleaned[row['table']][row['column']+'_seconds'].replace('',np.nan)
            assert pd.to_numeric(seconds,errors='coerce').notna().equals(values.notna())
    summary=json.loads((Path(out)/'cohort_summary.json').read_text())
    cohort=pd.read_csv(Path(out)/'starter_cohort.csv')
    excluded=summary['excluded_non_qualifiers']+summary['excluded_documented_non_starts']+summary['ambiguous_starts_withheld']
    assert summary['raw_result_rows']-excluded-summary['duplicate_rows_removed']==len(cohort)
    assert not cohort.duplicated(KEY).any()
    joined=raw['results'].merge(raw['races'][['raceId','circuitId']],on='raceId',validate='many_to_one')
    joined=joined.merge(raw['status'][['statusId','status']],on='statusId',validate='many_to_one')
    joined=joined.merge(raw['circuits'][['circuitId','country']],on='circuitId',validate='many_to_one')
    assert len(joined)==len(raw['results'])
    qualified=cohort.merge(cleaned['qualifying'][KEY+['position']],on=KEY,how='left',validate='one_to_one')
    assert len(qualified)==len(cohort)
    target=model[KEY+['podium']].merge(cohort[KEY+['positionOrder']],on=KEY,validate='one_to_one')
    assert len(target)==len(model) and target.podium.eq(target.positionOrder.le(3).astype(int)).all()
    row_counts=list(counts.values())+[
        {'stage':'results -> races -> status -> circuits','input_rows':len(raw['results']),'output_rows':len(joined),'row_difference':0},
        {'stage':'eligible cohort -> qualifying (left join)','input_rows':len(cohort),'output_rows':len(qualified),'row_difference':0},
        {'stage':'raw results -> documented exclusions','input_rows':summary['raw_result_rows'],'output_rows':summary['raw_result_rows']-excluded,'row_difference':-excluded},
        {'stage':'eligible entries -> deterministic deduplication','input_rows':len(cohort)+summary['duplicate_rows_removed'],'output_rows':len(cohort),'row_difference':-summary['duplicate_rows_removed']},
        {'stage':'historical cohort -> predeclared 2003–2024 model window','input_rows':len(cohort),'output_rows':len(model),'row_difference':len(model)-len(cohort)}]
    pd.DataFrame(row_counts).to_csv(destination/'row_count_audit.csv',index=False)
    pd.DataFrame(keys).to_csv(destination/'primary_key_recheck.csv',index=False)
    pd.DataFrame(foreign_keys).to_csv(destination/'foreign_key_recheck.csv',index=False)
    return {'tables_checked':len(keys),'foreign_key_relationships_checked':len(foreign_keys),
            'cohort_rows':len(cohort),'model_rows':len(model),'conversion_failures':0,
            'historical_cohort_fully_verified':False}

def audit_inference_contract(models, numeric, categorical, out, destination):
    record=json.loads((Path(out)/'inference_examples.json').read_text())['complete_record']
    results=[]
    before={name:joblib.hash(info['pipeline']) for name,info in models.items()}
    for name in models:
        expected=podium_inference(record,models,numeric,categorical,name)
        valid={'reordered_fields':dict(reversed(list(record.items()))),
               'partially_missing':{k:v for k,v in record.items() if k not in ['q1_gap_to_best','qualifying_position','driver_standings_rank_before']},
               'unseen_categories':{**record,'constructorId':'UNSEEN_TEAM','circuitId':'UNSEEN_CIRCUIT'},
               'minimal_missing_record':{'grid_position':record['grid_position']}}
        for case,value in valid.items():
            pred=podium_inference(value,models,numeric,categorical,name)
            assert 0<=pred['podium_probability']<=1
            assert pred['predicted_podium'] in [0,1]
            assert pred['threshold']==models[name]['threshold']
            if case=='reordered_fields': assert pred==expected
            if case=='partially_missing': assert len(pred['missing_features'])==3
            results.append({'model':name,'case':case,'expected':'finite frozen prediction','passed':True})
        invalid={'empty':{},'unknown_field':{**record,'positionOrder':1},
                 'infinity':{**record,'grid_position':float('inf')},
                 'invalid_numeric':{**record,'grid_position':'invalid'}}
        for case,value in invalid.items():
            try: podium_inference(value,models,numeric,categorical,name)
            except (ValueError,TypeError): pass
            else: raise AssertionError((name,case))
            results.append({'model':name,'case':case,'expected':'reject invalid input','passed':True})
    try: podium_inference(record,models,numeric,categorical,'unknown_model')
    except ValueError: pass
    else: raise AssertionError('Unknown model accepted')
    results.append({'model':'unknown','case':'unknown_model','expected':'reject invalid input','passed':True})
    assert before=={name:joblib.hash(info['pipeline']) for name,info in models.items()}
    result=pd.DataFrame(results)
    result.to_csv(destination/'inference_contract_checks.csv',index=False)
    return result

def run_batch3_acceptance(model, numeric, categorical, models, out, data_dir):
    destination=Path(out)/'batch3'
    destination.mkdir(parents=True,exist_ok=True)
    lock_before=file_hash(Path(out)/'model_choices_frozen.json')
    raw_before={name:file_hash(Path(data_dir)/(name+'.csv')) for name in PK}
    target,target_info=accuracy_target_audit(model,models,out,destination)
    register=complete_feature_evidence(numeric,categorical,out,destination)
    cleaning=audit_cleaning_and_joins(model,data_dir,out,destination)
    inference=audit_inference_contract(models,numeric,categorical,out,destination)
    ablation=pd.read_csv(Path(out)/'feature_group_ablation.csv')
    orders=pd.read_csv(Path(out)/'preprocessing_order_results.csv')
    distribution=pd.read_csv(Path(out)/'preprocessing_distribution_effects.csv')
    assert len(ablation)==7 and set(ablation.removed_group)-{'none'}==set(register.group)
    assert ablation.validation_rows.nunique()==1 and ablation.validation_rows.iloc[0]==len(model[model.split.eq('validation')])
    assert orders.groupby('experiment').size().eq(2).all()
    assert len(distribution)==len(numeric)*4
    assert raw_before=={name:file_hash(Path(data_dir)/(name+'.csv')) for name in PK}
    assert lock_before==file_hash(Path(out)/'model_choices_frozen.json')
    summary={'rows_in_scope':list(range(21,31)),'model_choice_manifest_unchanged':True,
        'raw_files_unchanged':True,'accuracy_target':target_info,'cleaning':cleaning,
        'inference_cases_passed':len(inference),'feature_register_entries':len(register),
        'order_variants':len(orders),'ablation_variants':len(ablation),
        'external_PDF_and_notebook_coverage':'Verify using verify_batch3.py after Run All.',
        'Kaggle_Run_All_verified':False,'all_member_access_review_verified':False,
        'official_submission_confirmed':False,'Done_cells_to_change':0}
    write_json(destination/'technical_acceptance.json',summary)
    print(json.dumps(summary,indent=2))
    return target,register,inference,summary
