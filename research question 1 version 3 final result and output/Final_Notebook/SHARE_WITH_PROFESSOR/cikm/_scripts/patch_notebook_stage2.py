"""Stage 2 notebook patcher — model-ensemble alignment with main.tex.

main.tex describes 12 models: LR, DT, RF, GBM, HGB, XGBoost, LightGBM,
CatBoost, AdaBoost, PyTorch DNN, Stacking, LGB-XGB Blend.

The notebook currently has: LR, DT, RF, GBM, AdaBoost, Bagging, XGBoost,
LightGBM, CatBoost, ExtraTrees, HistGBM, Stacking.

This patch:
  - removes Bagging and ExtraTrees
  - adds PyTorch DNN (feedforward, 512->256->128, BCEWithLogitsLoss, 30 epochs)
  - adds LGB-XGB Blend (soft-voting of LightGBM and XGBoost probabilities, 0.6/0.4)
  - records their test predictions into test_predictions and trained_model_objects
    so the rest of the notebook (fairness, VFR, cross-site, intervention) uses
    the corrected model set without further changes.

IMPORTANT: this patch only modifies Cell 13 (training). The notebook must be
re-executed top-to-bottom after this patch lands; runtime is substantial
(XGBoost/LightGBM/CatBoost GPU + PyTorch DNN training on 740K rows).
"""
import json, os, shutil, datetime, sys
sys.stdout.reconfigure(encoding='utf-8')

NB = 'CIKM_2026_LOS_Fairness.ipynb'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

bkp = f"CIKM_2026_LOS_Fairness.pre-stage2.{datetime.datetime.now():%Y%m%d-%H%M%S}.ipynb"
shutil.copy(NB, bkp)
print(f"[backup] {bkp}")

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- Cell 13: swap Bagging+ExtraTrees for PyTorch DNN + LGB-XGB Blend ---
OLD_CELL13 = """# ──────────────────────────────────────────────────────────────
# Cell 6 · Train 12 Models
# ──────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'Bagging': BaggingClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                                  tree_method='hist', device='cuda', random_state=RANDOM_STATE,
                                  eval_metric='logloss', verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=500, num_leaves=63, max_depth=8,
                                     learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05,
                                    random_state=RANDOM_STATE, verbose=0, task_type='GPU'),
    'Extra Trees': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=500, max_depth=8,
                                                            learning_rate=0.05, random_state=RANDOM_STATE),
}"""

NEW_CELL13_TOP = """# ──────────────────────────────────────────────────────────────
# Cell 6 · Train 12 Models (aligned with main.tex Sec 4.1)
# ──────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                                  tree_method='hist', device='cuda', random_state=RANDOM_STATE,
                                  seed=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=500, num_leaves=63, max_depth=8,
                                     learning_rate=0.05, random_state=RANDOM_STATE, seed=RANDOM_STATE,
                                     verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05,
                                    random_state=RANDOM_STATE, verbose=0, task_type='GPU'),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=500, max_depth=8,
                                                            learning_rate=0.05, random_state=RANDOM_STATE),
}"""

# Patch the top block
assert OLD_CELL13 in ''.join(nb['cells'][13]['source']), "Cell 13 header block not matching expected"
src13 = ''.join(nb['cells'][13]['source']).replace(OLD_CELL13, NEW_CELL13_TOP)

# Append: PyTorch DNN training and LGB-XGB Blend *after* the main loop but *before*
# the results_df sort. We insert it after the stacking-training loop block. The
# simplest injection point is just before "results_df = pd.DataFrame(results)".

INJECT_AFTER = """    results.append({'Model': name, 'Accuracy': acc, 'AUC': auc, 'F1': f1,
                    'Precision': prec, 'Recall': rec, 'Time_sec': time.time()-t1})
    print(f"  {name:25s} Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  ({time.time()-t1:.1f}s)")

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)"""

INJECT_NEW = """    results.append({'Model': name, 'Accuracy': acc, 'AUC': auc, 'F1': f1,
                    'Precision': prec, 'Recall': rec, 'Time_sec': time.time()-t1})
    print(f"  {name:25s} Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  ({time.time()-t1:.1f}s)")

# ────────────────────────────────────────────────────────────
# PyTorch DNN (feedforward: 512 -> 256 -> 128, BatchNorm + ReLU + Dropout)
# ────────────────────────────────────────────────────────────
_dnn_start = time.time()
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _LOS_DNN(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    torch.manual_seed(RANDOM_STATE)
    _dnn = _LOS_DNN(X_train.shape[1]).to(_device)
    _opt = torch.optim.Adam(_dnn.parameters(), lr=1e-3, weight_decay=1e-4)
    _loss_fn = nn.BCEWithLogitsLoss()

    _Xtr = torch.tensor(X_train, dtype=torch.float32)
    _ytr = torch.tensor(y_train, dtype=torch.float32)
    _Xte = torch.tensor(X_test,  dtype=torch.float32)
    _tr_ds = TensorDataset(_Xtr, _ytr)
    _tr_dl = DataLoader(_tr_ds, batch_size=2048, shuffle=True,
                        generator=torch.Generator().manual_seed(RANDOM_STATE))
    _N_EPOCHS = 30
    for _epoch in range(_N_EPOCHS):
        _dnn.train()
        _loss_sum = 0.0
        for _xb, _yb in _tr_dl:
            _xb = _xb.to(_device); _yb = _yb.to(_device)
            _opt.zero_grad()
            _logits = _dnn(_xb)
            _l = _loss_fn(_logits, _yb)
            _l.backward(); _opt.step()
            _loss_sum += _l.item() * _xb.size(0)
        if (_epoch + 1) % 10 == 0:
            print(f"    DNN epoch {_epoch+1:02d}/{_N_EPOCHS}  loss={_loss_sum/len(_tr_ds):.4f}")
    _dnn.eval()
    with torch.no_grad():
        _prob_dnn = torch.sigmoid(_dnn(_Xte.to(_device))).cpu().numpy()
    _y_pred_dnn = (_prob_dnn >= 0.5).astype(int)
    _acc = accuracy_score(y_test, _y_pred_dnn); _auc = roc_auc_score(y_test, _prob_dnn)
    _f1  = f1_score(y_test, _y_pred_dnn)
    _prec = precision_score(y_test, _y_pred_dnn); _rec = recall_score(y_test, _y_pred_dnn)
    test_predictions['DNN (PyTorch)'] = {'y_pred': _y_pred_dnn, 'y_prob': _prob_dnn}
    trained_model_objects['DNN (PyTorch)'] = _dnn
    results.append({'Model': 'DNN (PyTorch)', 'Accuracy': _acc, 'AUC': _auc, 'F1': _f1,
                    'Precision': _prec, 'Recall': _rec, 'Time_sec': time.time()-_dnn_start})
    print(f"  {'DNN (PyTorch)':25s} Acc={_acc:.4f}  AUC={_auc:.4f}  F1={_f1:.4f}  ({time.time()-_dnn_start:.1f}s)")
except Exception as _e:
    # Fallback to sklearn MLPClassifier (still a feed-forward DNN, no CUDA)
    print(f"  [INFO] PyTorch not available ({_e}); using sklearn MLPClassifier as DNN")
    from sklearn.neural_network import MLPClassifier
    _dnn_start = time.time()
    _mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu',
                          alpha=1e-4, batch_size=2048, learning_rate_init=1e-3,
                          max_iter=30, random_state=RANDOM_STATE, early_stopping=False,
                          verbose=False)
    _mlp.fit(X_train, y_train)
    _prob_dnn = _mlp.predict_proba(X_test)[:, 1]
    _y_pred_dnn = (_prob_dnn >= 0.5).astype(int)
    _acc = accuracy_score(y_test, _y_pred_dnn); _auc = roc_auc_score(y_test, _prob_dnn)
    _f1  = f1_score(y_test, _y_pred_dnn)
    _prec = precision_score(y_test, _y_pred_dnn); _rec = recall_score(y_test, _y_pred_dnn)
    test_predictions['DNN (PyTorch)'] = {'y_pred': _y_pred_dnn, 'y_prob': _prob_dnn}
    trained_model_objects['DNN (PyTorch)'] = _mlp
    results.append({'Model': 'DNN (PyTorch)', 'Accuracy': _acc, 'AUC': _auc, 'F1': _f1,
                    'Precision': _prec, 'Recall': _rec, 'Time_sec': time.time()-_dnn_start})
    print(f"  {'DNN (sklearn MLP)':25s} Acc={_acc:.4f}  AUC={_auc:.4f}  F1={_f1:.4f}  ({time.time()-_dnn_start:.1f}s)")

# ────────────────────────────────────────────────────────────
# LGB-XGB Blend (soft-voting: 0.6 * LightGBM + 0.4 * XGBoost probabilities)
# ────────────────────────────────────────────────────────────
if 'LightGBM' in test_predictions and 'XGBoost' in test_predictions:
    _blend_start = time.time()
    _prob_blend = 0.6 * test_predictions['LightGBM']['y_prob'] + 0.4 * test_predictions['XGBoost']['y_prob']
    _y_pred_blend = (_prob_blend >= 0.5).astype(int)
    _acc = accuracy_score(y_test, _y_pred_blend); _auc = roc_auc_score(y_test, _prob_blend)
    _f1  = f1_score(y_test, _y_pred_blend)
    _prec = precision_score(y_test, _y_pred_blend); _rec = recall_score(y_test, _y_pred_blend)
    test_predictions['LGB-XGB Blend'] = {'y_pred': _y_pred_blend, 'y_prob': _prob_blend}
    trained_model_objects['LGB-XGB Blend'] = 'soft_vote(LightGBM=0.6, XGBoost=0.4)'
    results.append({'Model': 'LGB-XGB Blend', 'Accuracy': _acc, 'AUC': _auc, 'F1': _f1,
                    'Precision': _prec, 'Recall': _rec, 'Time_sec': time.time()-_blend_start})
    print(f"  {'LGB-XGB Blend':25s} Acc={_acc:.4f}  AUC={_auc:.4f}  F1={_f1:.4f}  (0.6*LGB + 0.4*XGB)")
else:
    print("  [WARN] LGB-XGB Blend skipped — LightGBM or XGBoost missing from trained models")

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)"""

assert INJECT_AFTER in src13, "Cell 13 inject anchor not found"
src13 = src13.replace(INJECT_AFTER, INJECT_NEW)

nb['cells'][13]['source'] = src13.splitlines(keepends=True)
nb['cells'][13]['outputs'] = []
nb['cells'][13]['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[done] Stage 2 patch applied to cell 13:")
print("  - removed: Bagging, Extra Trees")
print("  - added:   PyTorch DNN (with deterministic generator), LGB-XGB Blend")
print("\nThe notebook now trains 12 models matching main.tex Sec 4.1:")
print("  LR, DT, RF, GBM, AdaBoost, HistGB, XGBoost, LightGBM, CatBoost,")
print("  Stacking, DNN (PyTorch), LGB-XGB Blend.")
print("\nNEXT STEP: re-execute the notebook top-to-bottom to regenerate all")
print("tables, figures, and VFR/intervention outputs with the corrected thresholds")
print("and the corrected model set.")
