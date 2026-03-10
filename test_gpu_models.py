"""Quick test of GPU models to find which one hangs."""
import time, sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Testing LightGBM GPU...", flush=True)
try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=10, device='gpu', verbose=-1)
    start = time.time()
    model.fit(X_train, y_train)
    print(f"  OK in {time.time()-start:.1f}s, acc={model.score(X_test, y_test):.3f}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("Testing XGBoost CUDA...", flush=True)
try:
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=10, device='cuda', tree_method='hist', eval_metric='logloss')
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  OK in {time.time()-start:.1f}s, acc={model.score(X_test, y_test):.3f}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("Testing RandomForest n_jobs=1...", flush=True)
try:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, n_jobs=1, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    print(f"  OK in {time.time()-start:.1f}s, acc={model.score(X_test, y_test):.3f}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("Testing PyTorch CUDA...", flush=True)
try:
    import torch
    import torch.nn as nn
    x = torch.randn(100, 20).cuda()
    model = nn.Linear(20, 1).cuda()
    out = model(x)
    print(f"  OK, output shape: {out.shape}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("All tests done.", flush=True)
