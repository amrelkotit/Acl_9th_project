"""Final plan rows 31–40: descriptive sensitivity and final evidence, no tuning."""
from pathlib import Path
import json
import platform
import importlib.metadata as package_metadata
if 'COUNTRY' not in globals():
    from batch1 import np, pd, KEY, COUNTRY, MECHANICAL, file_hash, write_json, markdown_table

def final_historical_checks(cohort, clean, model, out):
    dest=Path(out)/'batch4'
    dest.mkdir(parents=True,exist_ok=True)
    # Independent recount of the three historical questions, using their documented joins/grain.
    a=cohort.loc[~cohort.name.str.contains('Indianapolis',case=False)].copy()
    a['era']=np.where(a.year.lt(2014),'1950–2013','2014–2024')
    front=a.loc[a.grid.isin([1,2])].groupby(['circuitId','era']).podium.agg(starts='size',podiums='sum',rate='mean').reset_index()
    saved=pd.read_csv(Path(out)/'q1_front_row_rates.csv')
    columns=['circuitId','era','starts','podiums','rate']
    pd.testing.assert_frame_equal(front[columns].sort_values(columns[:2]).reset_index(drop=True),
        saved[columns].sort_values(columns[:2]).reset_index(drop=True),check_dtype=False)
    duplicates=pd.read_csv(Path(out)/'duplicate_resolution.csv')
    retained=duplicates.loc[duplicates.retained.astype(str).str.lower().eq('true')]
    alternative=duplicates.sort_values('resultId').drop_duplicates(KEY).copy()
    alternative['podium']=alternative.positionOrder.le(3).astype(int)
    joined=retained[KEY+['resultId','positionOrder','year']].merge(
        alternative[KEY+['resultId','positionOrder']],on=KEY,suffixes=('_chosen','_alternative'),validate='one_to_one')
    changed=joined.resultId_chosen.ne(joined.resultId_alternative)
    changed_target=joined.positionOrder_chosen.le(3).ne(joined.positionOrder_alternative.le(3))
    duplicate_info={'duplicate_driver_race_groups':len(joined),'alternate_rule':'Smallest resultId only; no target-based selection',
        'changed_source_rows':int(changed.sum()),'changed_podium_labels':int(changed_target.sum()),
        'changed_rows_in_2003_2024_model_window':int((changed & joined.year.between(2003,2024)).sum()),
        'limitation':'Sensitivity is not independent ground-truth reconstruction of the historical starting car.'}
    write_json(dest/'duplicate_rule_sensitivity.json',duplicate_info)
    home=a.merge(clean['drivers'][['driverId','nationality']],on='driverId',validate='many_to_one').merge(
        clean['circuits'][['circuitId','country']],on='circuitId',validate='many_to_one')
    home['home_country']=home.nationality.str.strip().map(COUNTRY)
    home['country']=home.country.replace({'United States':'USA'})
    eligible=home.loc[home.home_country.notna() & home.grid.gt(0)].copy()
    eligible['home']=eligible.home_country.eq(eligible.country)
    sensitivity=[]
    for label,frame in [('1950–2024',eligible),('1950–2013',eligible[eligible.year.lt(2014)]),('2014–2024',eligible[eligible.year.ge(2014)])]:
        for control in ['coarse_grid_strata','exact_grid_position']:
            f=frame.copy()
            f['stratum']=pd.cut(f.grid,[0,2,5,10,15,np.inf],labels=['1–2','3–5','6–10','11–15','16+']) if control=='coarse_grid_strata' else f.grid
            g=f.groupby(['stratum','home'],observed=True).podium.agg(['size','mean']).reset_index()
            rates=g.pivot(index='stratum',columns='home',values='mean').reindex(columns=[False,True]).dropna()
            counts=g.pivot(index='stratum',columns='home',values='size').loc[rates.index]
            weights=counts.sum(axis=1)/counts.sum().sum()
            adjusted=rates.mul(weights,axis=0).sum()
            sensitivity.append({'era':label,'control':control,'eligible_rows':len(frame),
                'common_support_rows':int(counts.sum().sum()),'common_strata':len(rates),
                'away_rate':adjusted[False],'home_rate':adjusted[True],
                'home_minus_away_percentage_points':100*(adjusted[True]-adjusted[False])})
    sensitivity=pd.DataFrame(sensitivity)
    original=pd.read_csv(Path(out)/'q2_home_summary.csv').set_index('group')
    overall=sensitivity.iloc[0]
    assert np.isclose(overall.home_rate,original.loc['Home','grid_standardized_rate'])
    assert np.isclose(overall.away_rate,original.loc['Away','grid_standardized_rate'])
    sensitivity.to_csv(dest/'home_grid_era_sensitivity.csv',index=False)
    mechanical=cohort.loc[cohort.year.between(2014,2024)].copy()
    mapping=pd.read_csv(Path(out)/'q3_status_mapping.csv').set_index('status')['group']
    assert not set(MECHANICAL)&set(mapping[mapping.eq('finished')].index)
    mechanical['period']=np.where(mechanical.year.le(2021),'2014–2021','2022–2024')
    mechanical['mechanical']=mechanical.status_text.isin(MECHANICAL).astype(int)
    mechanical['ambiguous']=mechanical.status_text.map(mapping).eq('ambiguous_or_other').astype(int)
    grouped=mechanical.groupby(['constructorId','period']).agg(starts=('podium','size'),mechanical_retirements=('mechanical','sum'),ambiguous_statuses=('ambiguous','sum')).reset_index()
    recorded=pd.read_csv(Path(out)/'q3_mechanical_rates.csv')
    columns=['constructorId','period','starts','mechanical_retirements','ambiguous_statuses']
    pd.testing.assert_frame_equal(grouped[columns].sort_values(columns[:2]).reset_index(drop=True),
        recorded[columns].sort_values(columns[:2]).reset_index(drop=True),check_dtype=False)
    uncertainty=recorded.copy()
    uncertainty['all_ambiguous_counted_rate']=(uncertainty.mechanical_retirements+uncertainty.ambiguous_statuses)/uncertainty.starts
    uncertainty.to_csv(dest/'mechanical_ambiguity_sensitivity.csv',index=False)
    coverage=cohort[KEY+['year']].merge(clean['qualifying'][KEY+['position']],on=KEY,how='left',validate='one_to_one')
    coverage['observed']=coverage.position.notna()
    periods=[]
    for label,first,last in [('1950–1995',1950,1995),('1996–2002',1996,2002),('training 2003–2019',2003,2019),('validation 2020–2021',2020,2021),('test 2022–2024',2022,2024)]:
        f=coverage[coverage.year.between(first,last)]
        periods.append({'period':label,'driver_race_rows':len(f),'qualifying_observed_rows':int(f.observed.sum()),'qualifying_coverage':f.observed.mean()})
    periods=pd.DataFrame(periods)
    periods.to_csv(dest/'qualifying_era_coverage.csv',index=False)
    ordered=cohort.sort_values(['driverId','event_idx'])
    previous=ordered.groupby('driverId').constructorId.shift()
    changes=ordered.loc[previous.notna() & ordered.constructorId.ne(previous),KEY+['year','constructorId']].copy()
    changes['previous_constructorId']=previous.loc[changes.index].astype(int)
    changes.to_csv(dest/'driver_constructor_changes.csv',index=False)
    constructors=cohort.groupby('constructorId').agg(first_year=('year','min'),last_year=('year','max'),starts=('podium','size')).reset_index().merge(
        clean['constructors'][['constructorId','name']],on='constructorId',validate='one_to_one')
    constructors.to_csv(dest/'constructor_history.csv',index=False)
    lineage=pd.read_csv(Path(out)/'feature_lineage.csv')
    for c in [c for c in lineage if c.endswith('_source_event')]:
        assert (lineage.loc[lineage[c].notna(),c]<lineage.loc[lineage[c].notna(),'event_idx']).all()
    versions={'python':platform.python_version(),'platform':platform.platform()}
    for package in ['numpy','pandas','scikit-learn','scipy','matplotlib','shap','lime','numba','joblib','nbformat','nbclient','ipykernel','reportlab','pypdf']:
        versions[package]=package_metadata.version(package)
    write_json(dest/'tested_runtime.json',versions)
    (dest/'requirements-tested.txt').write_text('\n'.join(f'{name}=={version}' for name,version in versions.items() if name not in ['python','platform'])+'\n')
    summary={'Q1_recount_matches':True,'Q2_standardized_recount_matches':True,'Q3_count_recount_matches':True,
        'strictly_earlier_feature_lineage':True,'duplicate_sensitivity':duplicate_info,
        'driver_constructor_change_events':len(changes),'constructors_in_history':len(constructors),
        'home_sensitivity_rows':len(sensitivity),'sprint_predictors_used':False,
        'model_choices_changed':False,'historical_ground_truth_fully_resolved':False,
        'evaluation_walkthrough_held':False,'Kaggle_Run_All_verified':False,'submission_confirmed':False}
    write_json(dest/'final_technical_checks.json',summary)
    print(json.dumps(summary,indent=2))
    return sensitivity,periods,uncertainty,summary

def write_final_report(out):
    out=Path(out)
    path=out/'milestone1_report.md'
    core=path.read_text(encoding='utf-8').split('## Final acceptance addendum')[0]
    core=core.replace('integrated implementation of work packages 1-20; team review, Kaggle verification and official submission remain pending.',
        'local implementation and acceptance evidence for all 40 plan rows; team walkthrough/review, Kaggle verification and official submission remain pending.')
    accuracy=pd.read_csv(out/'batch3/accuracy_target_audit.csv')
    test=accuracy[accuracy.split.eq('test')].copy()
    test['accuracy_percent']=test.accuracy*100
    test['majority_baseline_percent']=test.always_no_podium_accuracy*100
    home=pd.read_csv(out/'batch4/home_grid_era_sensitivity.csv')
    coverage=pd.read_csv(out/'batch4/qualifying_era_coverage.csv')
    mechanical=pd.read_csv(out/'batch4/mechanical_ambiguity_sensitivity.csv')
    summary=json.loads((out/'batch4/final_technical_checks.json').read_text())
    duplicate=summary['duplicate_sensitivity']
    text='\n\n'.join([core,
        '## Final acceptance addendum',
        'This addendum reconciles the final checks without changing any model, threshold or feature choice after test evaluation. The final feature register and test-evaluated contract are in the batch-3 evidence; initial registers/contracts remain historical checkpoints.',
        '### User accuracy goal and majority baseline',
        'The user requested at least 80% ordinary binary accuracy. All three frozen models meet that target on historical 2022-2024 test data. This is not a required score in the milestone brief and does not guarantee future-race accuracy. The always-no-podium baseline also exceeds 80%, so podium precision/recall/F1 and ranking metrics remain essential.',
        markdown_table(test[['model','accuracy_percent','majority_baseline_percent','accuracy_lift_percentage_points','precision','recall','F1']]),
        'The validation-selected random forest scores 90.10% accuracy, 5.17 percentage points above the 84.93% no-podium baseline. Its podium precision is 69.44% and recall 61.27%; a high overall accuracy does not mean 90% of actual podiums are recovered.',
        '### Historical cohort and duplicate sensitivity',
        f'A second deterministic rule keeps the smallest resultId without prioritizing positive grid. Across {duplicate["duplicate_driver_race_groups"]} duplicate driver-race groups, it changes {duplicate["changed_source_rows"]} retained source rows and {duplicate["changed_podium_labels"]} podium labels. Changed rows in the 2003-2024 modeling window: {duplicate["changed_rows_in_2003_2024_model_window"]}. Neither rule uses the target to select a row. This sensitivity does not identify historical starting-car ground truth; unresolved cases remain disclosed.',
        'Question 1 was independently recounted from actual grid 1/2, using the documented circuit/era grain, and matches all saved counts/rates. Qualifying is not substituted for actual grid. Indianapolis 500 events remain excluded; later road-course US Grands Prix remain eligible. Existing front-row figures and complete circuit rankings provide the visual evidence.',
        '### Home-race sensitivity to grid control and era',
        'The following descriptive sensitivity compares broad grid strata with exact recorded grid positions using common pooled weights and only strata represented by both home and away starts. It also separates pre-2014 and hybrid-era outcomes. The primary broad-stratum result is independently reproduced. Coverage of common support is reported because exact matching can exclude unsupported grid positions.',
        markdown_table(home),
        'A change in the home-away difference across matching definitions or eras is evidence of sensitivity, not a causal home advantage. Exact grid control still leaves driver, team, circuit, season and selection confounding. Nationality mappings, positive-grid restriction and ambiguous dual-nationality exclusions remain unchanged.',
        '### Mechanical-status uncertainty',
        'Question 3 was independently recounted for every constructor/period and matches the saved mechanical and ambiguous counts. Finished/lapped finishers, crashes, driver causes, regulatory decisions and documented non-starts are not counted as proven mechanical failures. The table shows each period’s five highest strict-rate constructors and a deliberately extreme scenario where every ambiguous status is counted as mechanical. This scenario is not an estimated true rate or confidence interval.',
        markdown_table(mechanical.groupby('period',sort=False).head(5)[['constructor_name','period','starts','mechanical_retirements','ambiguous_statuses','mechanical_rate','all_ambiguous_counted_rate']]),
        '### Qualifying coverage and constructor history',
        markdown_table(coverage),
        'Training-only qualifying coverage supports the predeclared 2003 model-window start; older races remain descriptive/history inputs. Later-session Q2/Q3 absence can indicate non-advancement and is not treated as an observed zero time. The existing coverage and feature figures show the drift. Validation/test coverage above is descriptive after choices were fixed.',
        f'The cohort contains {summary["driver_constructor_change_events"]} observed driver constructor-change events and {summary["constructors_in_history"]} constructor identities. Constructor form follows constructorId, while driver form follows driverId; a driver moving teams receives the destination constructor’s prior history. New constructorIds reset team history rather than silently merging renamed organizations. Source-event checks confirm all form/standings histories are strictly earlier than the predicted event.',
        '### Evaluation preparation and external completion',
        'The repository provides an evidence-based walkthrough and question set for every member. A mock evaluation has not been held or passed merely because the guide exists. The course brief says evaluation dates and slot reservations are announced separately. A reviewed official October 7 office-hours announcement concerns one 15-minute office-hours slot per team, reserved by 11:59 PM the preceding day; this is not the milestone submission deadline or a confirmed evaluation date.',
        'All six plan metrics plus separately defined AP, and both local SHAP/LIME, cover the named metric/XAI outputs while preserving the brief’s wording ambiguity. No official clarification is invented. Local runtime versions and a tested dependency list are recorded. Final Kaggle restart/Run All, four-member access and review, accepted-work synchronization, and official submission/confirmation remain external requirements. No Done cells or member contribution approvals are inferred from code execution.',
        'The final compliance matrix links all 40 plan rows and the eight official deliverable groups to notebook/report evidence. Plan rows are team work packages/checks, not the brief’s numbering. The submission deadline remains 18 October 2026 at 11:59 PM.'])+'\n'
    path.write_text(text,encoding='utf-8')
    return text
