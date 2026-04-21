"""Memory-optimize Cell 10 (feature engineering): encode in-place to avoid df.copy()."""
import json, os, shutil, datetime, sys
sys.stdout.reconfigure(encoding='utf-8')
NB = 'CIKM_2026_LOS_Fairness.ipynb'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bkp = f"CIKM_2026_LOS_Fairness.pre-mem.{datetime.datetime.now():%Y%m%d-%H%M%S}.ipynb"
shutil.copy(NB, bkp)
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

old = """# Encode categoricals
le_dict = {}
df_enc = df.copy()
for col in feature_cols:
    if df_enc[col].dtype == 'object':
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le

# Split
X = df_enc[feature_cols].fillna(0).values
y = df_enc[target].values
hospital_ids = df_enc['THCIC_ID'].values"""

new = """# Encode categoricals in-place to avoid a full DataFrame copy (memory-heavy on 925K rows)
le_dict = {}
for col in feature_cols:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
df_enc = df  # alias; no copy

# Downcast dtypes to shrink memory before materialising X
for col in feature_cols:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')
    elif df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')

# Split
X = df_enc[feature_cols].fillna(0).values.astype('float32')
y = df_enc[target].values
hospital_ids = df_enc['THCIC_ID'].values"""

src10 = ''.join(nb['cells'][10]['source'])
assert old in src10, "expected Cell 10 block not found"
nb['cells'][10]['source'] = src10.replace(old, new).splitlines(keepends=True)
nb['cells'][10]['outputs'] = []
nb['cells'][10]['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('[done] Cell 10 memory-optimized (no df.copy, dtype downcast).')
