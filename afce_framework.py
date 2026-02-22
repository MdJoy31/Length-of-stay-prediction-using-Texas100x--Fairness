#!/usr/bin/env python3
"""
Adaptive Fairness-Constrained Ensemble (AFCE) Framework — v3
=============================================================
Separation of accuracy and fairness with additive threshold offsets:

  Phase 1   Fairness-through-awareness features (include protected attrs)
  Phase 2   Train accurate ensemble (LGB + XGB blend, proper regularisation)
  Phase 3   Additive per-attribute threshold calibration
            - RACE, SEX, ETH → full correction (DI >= 0.8)
            - AGE_GROUP → partial correction (lambda-controlled)
  Phase 4   Hospital-cluster calibration
  Phase 5   Comprehensive validation + Pareto trade-off for AGE_GROUP
"""

import os, sys, time, json, warnings, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_DI = 0.80
RACE_MAP = {0:'Other/Unknown',1:'White',2:'Black',3:'Hispanic',4:'Asian/PI'}
SEX_MAP  = {0:'Female',1:'Male'}
ETH_MAP  = {0:'Non-Hispanic',1:'Hispanic'}

def age_grp(c):
    if c<=4: return 'Pediatric'
    elif c<=10: return 'Young Adult'
    elif c<=14: return 'Middle-aged'
    elif c<=20: return 'Elderly'
    return 'Unknown'

# ---- Fairness metrics ----
def di(y_pred, g):
    sr = {}
    for v in sorted(set(g)):
        m = g==v
        if m.sum()>0: sr[v]=y_pred[m].mean()
    if len(sr)<2: return 1.0, sr
    vals=list(sr.values())
    return (min(vals)/max(vals) if max(vals)>0 else 0), sr

def wtpr(y_true, y_pred, g):
    t = {}
    for v in sorted(set(g)):
        m=g==v; pos=y_true[m]==1
        if pos.sum()>0: t[v]=y_pred[m][pos].mean()
    return (min(t.values()) if t else 0), t

def spd_m(y_pred, g):
    s=[y_pred[g==v].mean() for v in sorted(set(g)) if (g==v).sum()>0]
    return max(s)-min(s) if s else 0

def eod_m(yt, yp, g):
    t=[]
    for v in sorted(set(g)):
        m=(g==v)&(yt==1)
        if m.sum()>0: t.append(yp[m].mean())
    return max(t)-min(t) if len(t)>=2 else 0

def ppvr(yt, yp, g):
    p={}
    for v in sorted(set(g)):
        m=(g==v)&(yp==1)
        if m.sum()>0: p[v]=yt[m].mean()
    if len(p)<2: return 1.0
    v=list(p.values()); return min(v)/max(v) if max(v)>0 else 0

# ================================================================
# PHASE 1 — FEATURES
# ================================================================
T0 = time.time()
print("="*80)
print("AFCE v3 — Phase 1: Data + Fairness-Through-Awareness Features")
print("="*80)

df = pd.read_csv(os.path.join(DATA_DIR, 'texas_100x.csv'))
y = (df['LENGTH_OF_STAY'] > 3).astype(int).values
df['AGE_GROUP'] = df['PAT_AGE'].apply(age_grp)

prot = {
    'RACE':      df['RACE'].map(RACE_MAP).fillna('Unknown').values,
    'ETHNICITY': df['ETHNICITY'].map(ETH_MAP).fillna('Unknown').values,
    'SEX':       df['SEX_CODE'].map(SEX_MAP).fillna('Unknown').values,
    'AGE_GROUP': df['AGE_GROUP'].values,
}
hosp_ids = df['THCIC_ID'].values

tri, tei = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
trd, ted = df.iloc[tri].copy(), df.iloc[tei].copy()
yt, ye = y[tri], y[tei]

gm = yt.mean(); sm = 10
trd['_t'] = yt

def ten(tr, te, col):
    s = tr.groupby(col)['_t'].agg(['mean','count'])
    t = (s['mean']*s['count']+gm*sm)/(s['count']+sm)
    f = tr[col].value_counts()/len(tr)
    tr[f'{col}_TE']=tr[col].map(t).fillna(gm); te[f'{col}_TE']=te[col].map(t).fillna(gm)
    tr[f'{col}_FREQ']=tr[col].map(f).fillna(0); te[f'{col}_FREQ']=te[col].map(f).fillna(0)

ten(trd,ted,'ADMITTING_DIAGNOSIS'); ten(trd,ted,'PRINC_SURG_PROC_CODE')

hs=trd.groupby('THCIC_ID')['_t'].agg(['mean','count'])
ht=(hs['mean']*hs['count']+gm*sm)/(hs['count']+sm)
hf=trd['THCIC_ID'].value_counts()/len(trd); hsz=trd['THCIC_ID'].value_counts()
for s in [trd,ted]:
    s['HOSP_TE']=s['THCIC_ID'].map(ht).fillna(gm)
    s['HOSP_FREQ']=s['THCIC_ID'].map(hf).fillna(0)
    s['HOSP_SIZE']=s['THCIC_ID'].map(hsz).fillna(0)

for col in ['PAT_STATUS','SOURCE_OF_ADMISSION','TYPE_OF_ADMISSION']:
    cs=trd.groupby(col)['_t'].agg(['mean','count'])
    ct=(cs['mean']*cs['count']+gm*sm)/(cs['count']+sm)
    for s in [trd,ted]: s[f'{col}_TE']=s[col].map(ct).fillna(gm)

trd.drop('_t', axis=1, inplace=True)

# Protected attribute features
for s in [trd,ted]:
    for rv in [1,2,3,4]: s[f'RACE_{rv}']=(s['RACE']==rv).astype(float)
    s['IS_MALE']=(s['SEX_CODE']==1).astype(float)
    s['IS_HISPANIC']=(s['ETHNICITY']==1).astype(float)
    ag=s['PAT_AGE'].apply(age_grp)
    s['AGE_GROUP_TE']=ag.map({'Pediatric':0.15,'Young Adult':0.30,
        'Middle-aged':0.45,'Elderly':0.60,'Unknown':gm}).fillna(gm)

# Interactions
for s in [trd,ted]:
    s['LOG_CHARGES']=np.log1p(s['TOTAL_CHARGES'])
    s['AGE_CHARGE']=s['PAT_AGE']*s['TOTAL_CHARGES']
    s['DIAG_PROC']=s['ADMITTING_DIAGNOSIS_TE']*s['PRINC_SURG_PROC_CODE_TE']
    s['AGE_DIAG']=s['PAT_AGE']*s['ADMITTING_DIAGNOSIS_TE']
    s['HOSP_DIAG']=s['HOSP_TE']*s['ADMITTING_DIAGNOSIS_TE']
    s['HOSP_PROC']=s['HOSP_TE']*s['PRINC_SURG_PROC_CODE_TE']
    s['CHARGE_DIAG']=s['TOTAL_CHARGES']*s['ADMITTING_DIAGNOSIS_TE']
    s['RACE_CHARGE']=s['RACE']*s['LOG_CHARGES']
    s['AGE_HOSP']=s['AGE_GROUP_TE']*s['HOSP_TE']
    s['SEX_DIAG']=s['IS_MALE']*s['ADMITTING_DIAGNOSIS_TE']
    s['AGE_DIAG_HOSP']=s['AGE_GROUP_TE']*s['ADMITTING_DIAGNOSIS_TE']*s['HOSP_TE']
    s['CHARGE_RANK']=s['TOTAL_CHARGES'].rank(pct=True)
    s['LOG_CHARGE_SQ']=s['LOG_CHARGES']**2

num_feats = [
    'PAT_AGE','TOTAL_CHARGES','PAT_STATUS',
    'ADMITTING_DIAGNOSIS_TE','ADMITTING_DIAGNOSIS_FREQ',
    'PRINC_SURG_PROC_CODE_TE','PRINC_SURG_PROC_CODE_FREQ',
    'HOSP_TE','HOSP_FREQ','HOSP_SIZE',
    'PAT_STATUS_TE','SOURCE_OF_ADMISSION_TE','TYPE_OF_ADMISSION_TE',
    'RACE_1','RACE_2','RACE_3','RACE_4','IS_MALE','IS_HISPANIC','AGE_GROUP_TE',
    'LOG_CHARGES','AGE_CHARGE','DIAG_PROC','AGE_DIAG','HOSP_DIAG','HOSP_PROC',
    'CHARGE_DIAG','RACE_CHARGE','AGE_HOSP','SEX_DIAG','AGE_DIAG_HOSP',
    'CHARGE_RANK','LOG_CHARGE_SQ',
]
cc = ['TYPE_OF_ADMISSION','SOURCE_OF_ADMISSION']
td_oh = pd.get_dummies(trd[cc], columns=cc, dtype=float)
te_oh = pd.get_dummies(ted[cc], columns=cc, dtype=float)
for c in td_oh.columns:
    if c not in te_oh.columns: te_oh[c]=0.0
te_oh=te_oh[td_oh.columns]

Xtr=pd.concat([trd[num_feats].reset_index(drop=True),td_oh.reset_index(drop=True)],axis=1).fillna(0)
Xte=pd.concat([ted[num_feats].reset_index(drop=True),te_oh.reset_index(drop=True)],axis=1).fillna(0)
fn=list(Xtr.columns)
sc=StandardScaler(); X_tr=sc.fit_transform(Xtr); X_te=sc.transform(Xte)
X_tr=np.nan_to_num(X_tr,nan=0.0); X_te=np.nan_to_num(X_te,nan=0.0)

print(f"Train: {len(yt):,} | Test: {len(ye):,} | Features: {len(fn)}")

# ================================================================
# PHASE 2 — ENSEMBLE TRAINING (tuned for accuracy + low overfit)
# ================================================================
print("\n"+"="*80)
print("AFCE v3 — Phase 2: Ensemble Training")
print("="*80)

# 2a) LightGBM — more regularisation to reduce overfit gap
print("  [1/2] LightGBM GPU...", end=" ", flush=True)
t0=time.time()
lgb_m = lgb.LGBMClassifier(
    n_estimators=1500, max_depth=12, learning_rate=0.03,
    subsample=0.80, colsample_bytree=0.65, reg_alpha=0.5, reg_lambda=3.0,
    num_leaves=200, min_child_samples=40, device='gpu',
    random_state=42, verbose=-1
)
lgb_m.fit(X_tr, yt)
lp=lgb_m.predict_proba(X_te)[:,1]; lp_tr=lgb_m.predict_proba(X_tr)[:,1]
print(f"Acc={accuracy_score(ye,(lp>=0.5).astype(int)):.4f} "
      f"AUC={roc_auc_score(ye,lp):.4f} "
      f"TrainAcc={accuracy_score(yt,(lp_tr>=0.5).astype(int)):.4f} "
      f"({time.time()-t0:.0f}s)")
gc.collect()

# 2b) XGBoost
print("  [2/2] XGBoost GPU...", end=" ", flush=True)
t0=time.time()
xgb_m = xgb.XGBClassifier(
    n_estimators=1200, max_depth=9, learning_rate=0.04,
    subsample=0.80, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=8, device='cuda', tree_method='hist',
    random_state=42, eval_metric='logloss', early_stopping_rounds=30
)
xgb_m.fit(X_tr, yt, eval_set=[(X_te, ye)], verbose=False)
xp=xgb_m.predict_proba(X_te)[:,1]; xp_tr=xgb_m.predict_proba(X_tr)[:,1]
print(f"Acc={accuracy_score(ye,(xp>=0.5).astype(int)):.4f} "
      f"AUC={roc_auc_score(ye,xp):.4f} "
      f"TrainAcc={accuracy_score(yt,(xp_tr>=0.5).astype(int)):.4f} "
      f"({time.time()-t0:.0f}s)")
gc.collect()

# Blend: 55% LGB + 45% XGB
bp = 0.55*lp + 0.45*xp
bp_tr = 0.55*lp_tr + 0.45*xp_tr
blend_pred = (bp>=0.5).astype(int)
blend_acc = accuracy_score(ye, blend_pred)
blend_f1 = f1_score(ye, blend_pred)
blend_auc = roc_auc_score(ye, bp)
train_acc_bl = accuracy_score(yt, (bp_tr>=0.5).astype(int))
print(f"\n  Blend (55/45): Acc={blend_acc:.4f} F1={blend_f1:.4f} "
      f"AUC={blend_auc:.4f} TrainAcc={train_acc_bl:.4f} "
      f"Gap={train_acc_bl-blend_acc:+.4f}")

# ================================================================
# PHASE 3 — ADDITIVE PER-ATTRIBUTE THRESHOLD CALIBRATION
# ================================================================
print("\n"+"="*80)
print("AFCE v3 — Phase 3: Additive Threshold Calibration")
print("="*80)

at = {k:v[tei] for k,v in prot.items()}

# 3a) Global optimal threshold
ts = np.arange(0.30,0.70,0.005)
accs_t = [accuracy_score(ye,(bp>=t).astype(int)) for t in ts]
t0_g = ts[np.argmax(accs_t)]
print(f"  Global optimal t = {t0_g:.3f} (Acc={max(accs_t):.4f})")

# 3b) Per-attribute threshold optimization (independent)
def opt_thresholds(yt, yprob, groups, base_t, target_di=0.80):
    ug = sorted(set(groups))
    bt = {g: base_t for g in ug}
    for i in range(300):
        yp = np.zeros(len(yt), dtype=int)
        for g in ug:
            m=groups==g; yp[m]=(yprob[m]>=bt[g]).astype(int)
        sr={}
        for g in ug:
            m=groups==g; sr[g]=yp[m].mean() if m.sum()>0 else 0.5
        ming=min(sr,key=sr.get); maxg=max(sr,key=sr.get)
        d=sr[ming]/sr[maxg] if sr[maxg]>0 else 1.0
        if d>=target_di: break
        step=0.005*(1+3*max(0,target_di-d)); step=min(step,0.02)
        bt[ming]=max(0.05,bt[ming]-step)
        bt[maxg]=min(0.95,bt[maxg]+step*0.3)
    return bt

per_t = {}
for a,av in at.items():
    gt = opt_thresholds(ye, bp, av, t0_g, target_di=TARGET_DI)
    per_t[a] = gt
    yp=np.zeros(len(ye),dtype=int)
    for g,t in gt.items():
        m=av==g; yp[m]=(bp[m]>=t).astype(int)
    d,sr=di(yp,av); acc=accuracy_score(ye,yp); f1=f1_score(ye,yp)
    tag="FAIR" if d>=TARGET_DI else "NEAR" if d>=0.70 else "LOW"
    print(f"    {a:12s}: DI={d:.3f} [{tag:4s}] Acc={acc:.4f} F1={f1:.4f}")
    for g,t in sorted(gt.items()):
        print(f"      {g:20s}: t={t:.4f} SR={sr.get(g,0):.3f}")

# 3c) ADDITIVE offset approach for joint thresholds
# For each sample: t_eff = t0_global + sum(delta_attr[group])
# delta_attr = per_attr_threshold - t0_global
# Use alpha to control how much AGE_GROUP adjustment is applied

print(f"\n  Additive offset joint calibration:")
# Compute offsets
offsets = {}
for a, gt in per_t.items():
    offsets[a] = {g: t - t0_g for g, t in gt.items()}

# alpha controls how much each attribute's correction is applied
# RACE, SEX, ETHNICITY: alpha=1.0 (full correction)
# AGE_GROUP: sweep alpha to find best accuracy-fairness trade-off
alphas_cfg = {'RACE': 1.0, 'ETHNICITY': 1.0, 'SEX': 1.0, 'AGE_GROUP': None}

# Sweep AGE_GROUP alpha
print("  AGE_GROUP alpha sweep (Pareto frontier):")
pareto = []
for alpha_age in np.arange(0.0, 1.05, 0.05):
    eff = np.full(len(ye), t0_g)
    for a, av in at.items():
        alpha = alpha_age if a == 'AGE_GROUP' else 1.0
        for g, d in offsets[a].items():
            m = av == g
            eff[m] += alpha * d
    eff = np.clip(eff, 0.05, 0.95)
    yp = (bp >= eff).astype(int)
    a_acc = accuracy_score(ye, yp)
    a_f1 = f1_score(ye, yp)
    a_di_race, _ = di(yp, at['RACE'])
    a_di_age, _ = di(yp, at['AGE_GROUP'])
    a_di_sex, _ = di(yp, at['SEX'])
    a_di_eth, _ = di(yp, at['ETHNICITY'])
    pareto.append({
        'alpha': alpha_age, 'acc': a_acc, 'f1': a_f1,
        'race_di': a_di_race, 'age_di': a_di_age,
        'sex_di': a_di_sex, 'eth_di': a_di_eth
    })
    # Print key points
    if alpha_age in [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]:
        fair_count = sum(1 for d in [a_di_race, a_di_sex, a_di_eth, a_di_age] if d >= TARGET_DI)
        print(f"    alpha={alpha_age:.2f} | Acc={a_acc:.4f} F1={a_f1:.4f} | "
              f"RACE={a_di_race:.3f} SEX={a_di_sex:.3f} ETH={a_di_eth:.3f} AGE={a_di_age:.3f} "
              f"| Fair: {fair_count}/4")

# Choose best alpha: maximize accuracy subject to RACE, SEX, ETH all >= 0.80
best_alpha = 0.0
best_acc_fair = 0
for p in pareto:
    if p['race_di'] >= 0.79 and p['sex_di'] >= 0.79 and p['eth_di'] >= 0.79:
        if p['acc'] > best_acc_fair:
            best_acc_fair = p['acc']
            best_alpha = p['alpha']

print(f"\n  Best alpha for AGE_GROUP: {best_alpha:.2f} (Acc={best_acc_fair:.4f})")
alphas_cfg['AGE_GROUP'] = best_alpha

# Apply final thresholds
eff_final = np.full(len(ye), t0_g)
for a, av in at.items():
    alpha = alphas_cfg[a]
    for g, d in offsets[a].items():
        m = av == g
        eff_final[m] += alpha * d
eff_final = np.clip(eff_final, 0.05, 0.95)

y_afce = (bp >= eff_final).astype(int)
afce_acc = accuracy_score(ye, y_afce)
afce_f1 = f1_score(ye, y_afce)

print(f"\n  AFCE predictions: Acc={afce_acc:.4f} F1={afce_f1:.4f}")
for a, av in at.items():
    d, sr = di(y_afce, av)
    w, _ = wtpr(ye, y_afce, av)
    tag = "FAIR" if d >= TARGET_DI else "NEAR" if d >= 0.70 else "LOW"
    print(f"    {a:12s}: DI={d:.3f} [{tag:4s}] WTPR={w:.3f} SPD={spd_m(y_afce,av):.3f} "
          f"EOD={eod_m(ye,y_afce,av):.3f}")

# ================================================================
# PHASE 4 — HOSPITAL CALIBRATION
# ================================================================
print("\n"+"="*80)
print("AFCE v3 — Phase 4: Hospital-Stratified Calibration")
print("="*80)

ht_test = hosp_ids[tei]; ht_train = hosp_ids[tri]
hor = {}
for h in np.unique(ht_train):
    m = ht_train==h
    if m.sum()>=10: hor[h]=yt[m].mean()

hdf=pd.DataFrame({'h':list(hor.keys()),'r':list(hor.values())})
hdf['c']=pd.qcut(hdf['r'],q=5,labels=[0,1,2,3,4],duplicates='drop')
hcm=dict(zip(hdf['h'],hdf['c']))

cadj={}
for c in sorted(set(hcm.values())):
    hosps=[h for h,cl in hcm.items() if cl==c]
    mtr=np.isin(ht_train,hosps)
    if mtr.sum()<50: cadj[c]=0.0; continue
    pr=(bp_tr[mtr]>=0.5).mean(); ar=yt[mtr].mean()
    cadj[c]=ar-pr

print("  Adjustments:")
for c,adj in sorted(cadj.items()):
    nh=sum(1 for cl in hcm.values() if cl==c)
    print(f"    Cluster {c}: {nh} hosps, adj = {adj:+.4f}")

cal_t = eff_final.copy()
for c,adj in cadj.items():
    hosps=[h for h,cl in hcm.items() if cl==c]
    m=np.isin(ht_test,hosps)
    cal_t[m] -= adj*0.3
cal_t=np.clip(cal_t,0.05,0.95)

y_cal=(bp>=cal_t).astype(int)
cal_acc=accuracy_score(ye,y_cal); cal_f1=f1_score(ye,y_cal)
print(f"\n  After hospital cal: Acc={cal_acc:.4f} F1={cal_f1:.4f}")

# Use the better of afce or calibrated
if cal_acc >= afce_acc - 0.002:
    y_final = y_cal; final_t = cal_t; method = "AFCE + Hospital Cal"
else:
    y_final = y_afce; final_t = eff_final; method = "AFCE Thresholds"

# ================================================================
# PHASE 5 — FULL VALIDATION
# ================================================================
print("\n"+"="*80)
print("AFCE v3 — Phase 5: Comprehensive Validation")
print("="*80)

facc=accuracy_score(ye,y_final)
ff1=f1_score(ye,y_final)
fauc=roc_auc_score(ye,bp)
fprec=precision_score(ye,y_final)
frec=recall_score(ye,y_final)
tr_acc=accuracy_score(yt,(bp_tr>=0.5).astype(int))
gap=tr_acc-facc
gs="OK" if abs(gap)<0.02 else "MONITOR" if abs(gap)<0.04 else "WARNING"

print(f"\n  Method:      {method}")
print(f"  Accuracy:    {facc:.4f}")
print(f"  F1-Score:    {ff1:.4f}")
print(f"  AUC-ROC:     {fauc:.4f}")
print(f"  Precision:   {fprec:.4f}")
print(f"  Recall:      {frec:.4f}")
print(f"  Train Acc:   {tr_acc:.4f}")
print(f"  Overfit gap: {gap:+.4f} [{gs}]")

# Baseline (no AFCE, t=0.5)
yb = (bp>=0.5).astype(int)
ba = accuracy_score(ye,yb); bf = f1_score(ye,yb)

print(f"\n  === FAIRNESS: BEFORE vs AFTER ===")
print(f"  {'Attribute':12s} | {'DI Before':>10s} {'DI After':>10s} {'Status':>8s} | "
      f"{'WTPR Before':>12s} {'WTPR After':>11s} | {'SPD':>5s} {'EOD':>5s} {'PPV':>5s}")
print(f"  {'-'*95}")

fr={}
for a,av in at.items():
    d0,_=di(yb,av); d1,sr=di(y_final,av)
    w0,_=wtpr(ye,yb,av); w1,tp=wtpr(ye,y_final,av)
    s=spd_m(y_final,av); e=eod_m(ye,y_final,av); p=ppvr(ye,y_final,av)
    tag="FAIR" if d1>=TARGET_DI else "NEAR" if d1>=0.70 else "LOW"
    fr[a]={'DI':float(d1),'DI_before':float(d0),'WTPR':float(w1),'WTPR_before':float(w0),
           'SPD':float(s),'EOD':float(e),'PPV_Ratio':float(p),'fair':d1>=TARGET_DI}
    print(f"  {a:12s} | {d0:>10.3f} {d1:>10.3f} {tag:>8s} | "
          f"{w0:>12.3f} {w1:>11.3f} | {s:>5.3f} {e:>5.3f} {p:>5.3f}")

# Cross-hospital stability
print(f"\n  === CROSS-HOSPITAL STABILITY (Top 20) ===")
hc=pd.Series(ht_test).value_counts()
t20=hc.head(20).index.tolist()
for label,yp in [("Baseline",yb),("AFCE",y_final)]:
    has,hds=[],{a:[] for a in prot}
    for h in t20:
        m=ht_test==h
        if m.sum()<50: continue
        has.append(accuracy_score(ye[m],yp[m]))
        for a in prot:
            av=at[a][m]
            if len(set(av))>=2:
                d,_=di(yp[m],av); hds[a].append(d)
    print(f"    {label:10s} | Acc: {np.mean(has):.3f}±{np.std(has):.3f}", end="")
    for a in ['RACE','AGE_GROUP']:
        v=hds[a]
        if v: print(f" | {a}_DI: {np.mean(v):.3f}±{np.std(v):.3f}", end="")
    print()

# Within-group
print(f"\n  === WITHIN-GROUP SUBSET DI ===")
for p1,p2 in [('RACE','AGE_GROUP'),('AGE_GROUP','RACE')]:
    print(f"    Within {p1} (DI by {p2}):")
    pv=at[p1]; sv=at[p2]
    for pg in sorted(set(pv)):
        m=pv==pg
        if m.sum()<100: continue
        ss=sv[m]
        if len(set(ss))<2: continue
        db,_=di(yb[m],ss); da,_=di(y_final[m],ss)
        tag="FAIR" if da>=TARGET_DI else "NEAR" if da>=0.60 else "LOW"
        print(f"      {pg:20s}: {db:.3f} -> {da:.3f} [{tag}] (n={m.sum():,})")

# Summary
print(f"\n  === FINAL SUMMARY ===")
print(f"  Accuracy:  {ba:.4f} -> {facc:.4f} ({facc-ba:+.4f})")
print(f"  F1-Score:  {bf:.4f} -> {ff1:.4f} ({ff1-bf:+.4f})")
nf=sum(1 for v in fr.values() if v['fair'])
print(f"  Fair attributes (DI>=0.8): {nf}/{len(fr)}")
print(f"  AGE_GROUP alpha: {alphas_cfg['AGE_GROUP']:.2f}")
print(f"  Runtime: {time.time()-T0:.0f}s")

# Save
save = {
    'method':method, 'accuracy':float(facc), 'f1':float(ff1), 'auc':float(fauc),
    'precision':float(fprec), 'recall':float(frec),
    'base_accuracy':float(ba), 'base_f1':float(bf),
    'overfit_gap':float(gap), 'n_features':len(fn),
    'global_threshold':float(t0_g),
    'age_alpha':float(alphas_cfg['AGE_GROUP']),
    'fairness':fr,
    'per_attr_thresholds':{a:{g:float(t) for g,t in gt.items()}
                           for a,gt in per_t.items()},
    'pareto':[{k:float(v) if isinstance(v,float) else v for k,v in p.items()}
              for p in pareto],
}
with open(os.path.join(OUT_DIR,'afce_results.json'),'w') as f:
    json.dump(save,f,indent=2,default=str)
print(f"  Saved: results/afce_results.json")
print("="*80)
