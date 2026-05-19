#!/usr/bin/env python3
"""
Run all 25 milestone tests for the Water Quality Prediction project.
Usage: python3 run_tests.py
All tests should pass if the notebooks have been executed in order (01 → 05).
"""

import os, sys, json
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
passed = failed = 0

def check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f'  PASS  {name}')
        passed += 1
    else:
        print(f'  FAIL  {name}' + (f' — {detail}' if detail else ''))
        failed += 1

print('\n── M1: Setup & Dataset Load ──────────────────────')
try:
    df = pd.read_csv(os.path.join(BASE, 'data', 'water_potability.csv'))
    check('T1.1  Dataset loads', True)
    check('T1.2  Shape (3276, 10)', df.shape == (3276, 10), str(df.shape))
    check('T1.3  Potability column exists', 'Potability' in df.columns)
    check('T1.4  Missing values in ph/Sulfate/THM',
          df[['ph','Sulfate','Trihalomethanes']].isnull().any().all())
except Exception as e:
    check('T1.1  Dataset loads', False, str(e))
    for t in ['T1.2','T1.3','T1.4']: check(t, False, 'skipped')

print('\n── M2: EDA ───────────────────────────────────────')
figs = ['fig2_1_distributions.png','fig2_2_class_dist.png',
        'fig2_3_heatmap.png','fig2_4_boxplots.png','fig2_5_violin_plots.png']
fig_dir = os.path.join(BASE, 'reports', 'figures')
for f in figs:
    check(f'T2.x  {f}', os.path.isfile(os.path.join(fig_dir, f)))

print('\n── M3: Preprocessing ────────────────────────────')
proc = os.path.join(BASE, 'data', 'processed')
for f in ['train.csv','test.csv','train_unscaled.csv','test_unscaled.csv','imputation_medians.json']:
    check(f'T3.x  {f}', os.path.isfile(os.path.join(proc, f)))

print('\n── M4: Model Training ───────────────────────────')
models_dir = os.path.join(BASE, 'models')
for f in ['logistic_regression.pkl','random_forest.pkl','xgboost.pkl','neural_network.pkl']:
    check(f'T4.x  {f}', os.path.isfile(os.path.join(models_dir, f)))

print('\n── M5: Model Evaluation ─────────────────────────')
try:
    with open(os.path.join(models_dir, 'training_metrics.json')) as f:
        metrics = json.load(f)
    check('T5.1  training_metrics.json exists', True)
    check('T5.2  XGBoost AUC ≥ 0.60',
          metrics.get('metrics',{}).get('XGBoost',{}).get('test_auc', 0) >= 0.60,
          str(metrics.get('metrics',{}).get('XGBoost',{}).get('test_auc')))
    eval_figs = ['fig5_1_confusion_matrix.png','fig5_2_roc_curves.png','fig5_3_feature_importance.png']
    for f in eval_figs:
        check(f'T5.x  {f}', os.path.isfile(os.path.join(fig_dir, f)))
except Exception as e:
    for t in range(5): check(f'T5.x', False, str(e))

print('\n── M6: Prediction Pipeline ──────────────────────')
try:
    model  = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    check('T6.1  best_model.pkl loads', True)
    check('T6.2  scaler.pkl loads', True)
    sample = np.array([[7.0, 204.0, 20791.0, 7.3, 368.5, 564.0, 10.4, 86.0, 2.96]])
    pred = int(model.predict_proba(scaler.transform(sample))[0][1] >= 0.44)
    check('T6.3  End-to-end prediction returns 0 or 1', pred in (0, 1), str(pred))
    check('T6.4  All 7 required figures on disk',
          all(os.path.isfile(os.path.join(fig_dir, f))
              for f in ['fig2_1_distributions.png','fig2_2_class_dist.png',
                        'fig2_3_heatmap.png','fig2_4_boxplots.png','fig2_5_violin_plots.png',
                        'fig5_1_confusion_matrix.png','fig5_2_roc_curves.png']))
except Exception as e:
    for t in range(4): check(f'T6.x', False, str(e))

print(f'\n{"─"*50}')
print(f'  {passed} passed  |  {failed} failed  |  {passed+failed} total')
print('  ALL TESTS PASSED ✓' if failed == 0 else f'  {failed} TEST(S) FAILED')
sys.exit(0 if failed == 0 else 1)
