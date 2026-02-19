#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 1: DATA PREPROCESSING (UPDATED)
═══════════════════════════════════════════════════════════════════════════════════

Author: Md Jannatul Rakib Joy
Supervisors: Dr. Caslon Chua, Dr. Viet Vo
Institution: Swinburne University of Technology

PURPOSE: Load and preprocess Texas-100X data (handles both CSV and pickle files)

DATA FILES:
    - texas_100x.csv (925,128 records) - Main dataset
    - texas_100x_features.p - Preprocessed features
    - texas_100x_labels.p - Labels  
    - texas_100x_feature_desc.p - Feature descriptions

RUN: python script1_data_preprocessing.py
NEXT: python script2_model_training.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'data_dir': Path('./data'),
    'output_dir': Path('./processed_data'),
    'los_threshold': 3,  # LOS > 3 days = Extended Stay
    'test_size': 0.2,
    'random_state': 42,
    'age_bins': [0, 18, 40, 65, 120],
    'age_labels': ['Pediatric (0-18)', 'Adult (19-40)', 'Middle-aged (41-65)', 'Elderly (65+)'],
    'protected_attributes': ['RACE', 'ETHNICITY', 'SEX', 'AGE_GROUP']
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)


def load_pickle_file(filepath):
    """Load a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def explore_data_files():
    """Explore all available data files."""
    print("=" * 70)
    print("EXPLORING DATA FILES")
    print("=" * 70)
    
    data_dir = CONFIG['data_dir']
    files_found = {}
    
    # Check for CSV
    csv_path = data_dir / 'texas_100x.csv'
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Found: texas_100x.csv ({size_mb:.1f} MB)")
        files_found['csv'] = csv_path
    
    # Check for pickle files
    pickle_files = ['texas_100x_features.p', 'texas_100x_labels.p', 'texas_100x_feature_desc.p']
    for pf in pickle_files:
        pf_path = data_dir / pf
        if pf_path.exists():
            size_mb = pf_path.stat().st_size / (1024 * 1024)
            print(f"✅ Found: {pf} ({size_mb:.1f} MB)")
            files_found[pf.replace('.p', '')] = pf_path
    
    return files_found


def load_from_csv(csv_path):
    """Load data from CSV file."""
    print(f"\n📂 Loading from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def load_from_pickle(files_found):
    """Load data from pickle files."""
    print(f"\n📂 Loading from pickle files...")
    
    features = None
    labels = None
    feature_desc = None
    
    if 'texas_100x_features' in files_found:
        features = load_pickle_file(files_found['texas_100x_features'])
        print(f"   Features shape: {features.shape if hasattr(features, 'shape') else type(features)}")
    
    if 'texas_100x_labels' in files_found:
        labels = load_pickle_file(files_found['texas_100x_labels'])
        print(f"   Labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
    
    if 'texas_100x_feature_desc' in files_found:
        feature_desc = load_pickle_file(files_found['texas_100x_feature_desc'])
        print(f"   Feature descriptions: {type(feature_desc)}")
        if isinstance(feature_desc, (list, dict)):
            print(f"   Features: {feature_desc}")
    
    return features, labels, feature_desc


def load_data():
    """Load Texas-100X data from available files."""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    files_found = explore_data_files()
    
    df = None
    features = None
    labels = None
    
    # Priority: CSV file (contains all columns including protected attributes)
    if 'csv' in files_found:
        df = load_from_csv(files_found['csv'])
        print(f"\n✅ Loaded {len(df):,} records from CSV")
        
        # Show first few rows
        print(f"\n📋 First 5 rows:")
        print(df.head())
        
        # Show column info
        print(f"\n📋 Column types:")
        print(df.dtypes)
        
    # Also load pickle files for reference
    if any(k.startswith('texas_100x_') for k in files_found.keys()):
        features, labels, feature_desc = load_from_pickle(files_found)
    
    if df is None and features is not None:
        print("\n⚠️ No CSV found, using pickle files")
        # Create DataFrame from features
        if feature_desc:
            df = pd.DataFrame(features, columns=feature_desc if isinstance(feature_desc, list) else None)
        else:
            df = pd.DataFrame(features)
        
        if labels is not None:
            df['LABEL'] = labels
    
    if df is None:
        raise FileNotFoundError("No data files found in ./data/ folder!")
    
    return df


def identify_columns(df):
    """Identify key columns in the dataset."""
    print("\n" + "=" * 70)
    print("STEP 2: IDENTIFYING COLUMNS")
    print("=" * 70)
    
    columns = df.columns.tolist()
    column_mapping = {}
    
    # Common column name variations
    los_names = ['LENGTH_OF_STAY', 'LOS', 'length_of_stay', 'DAYS', 'los']
    age_names = ['AGE', 'age', 'PAT_AGE', 'PATIENT_AGE']
    sex_names = ['SEX', 'GENDER', 'sex', 'SEX_CODE', 'FEMALE']
    race_names = ['RACE', 'race', 'RACE_CD', 'ETHNCTY']
    ethnicity_names = ['ETHNICITY', 'ethnicity', 'ETHNIC', 'HISPANIC']
    hospital_names = ['HOSPITAL_ID', 'PROVIDER_ID', 'THCIC_ID', 'FAC_ID', 'hospital_id']
    
    # Find columns
    for col in columns:
        col_upper = col.upper()
        
        if any(name.upper() == col_upper for name in los_names):
            column_mapping['LENGTH_OF_STAY'] = col
        elif any(name.upper() == col_upper for name in age_names):
            column_mapping['AGE'] = col
        elif any(name.upper() == col_upper for name in sex_names):
            column_mapping['SEX'] = col
        elif any(name.upper() == col_upper for name in race_names):
            column_mapping['RACE'] = col
        elif any(name.upper() == col_upper for name in ethnicity_names):
            column_mapping['ETHNICITY'] = col
        elif any(name.upper() == col_upper for name in hospital_names):
            column_mapping['HOSPITAL_ID'] = col
    
    print(f"\n📋 Column mapping:")
    for standard, actual in column_mapping.items():
        print(f"   {standard}: {actual}")
    
    # Check for missing critical columns
    required = ['LENGTH_OF_STAY', 'AGE', 'SEX', 'RACE']
    missing = [c for c in required if c not in column_mapping]
    
    if missing:
        print(f"\n⚠️ Could not find columns for: {missing}")
        print(f"   Available columns: {columns}")
        print(f"\n   Please check your data file and update column names if needed.")
    
    return column_mapping


def preprocess_data(df, column_mapping):
    """Preprocess the data."""
    print("\n" + "=" * 70)
    print("STEP 3: PREPROCESSING")
    print("=" * 70)
    
    # Rename columns to standard names
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    df = df.rename(columns=reverse_mapping)
    
    # Handle missing values
    print(f"\n📊 Missing values before cleaning:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   No missing values!")
    
    # Create binary target: LOS > 3 days
    if 'LENGTH_OF_STAY' in df.columns:
        df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > CONFIG['los_threshold']).astype(int)
        print(f"\n✅ Created LOS_BINARY: LOS > {CONFIG['los_threshold']} days")
        print(f"   Extended Stay (1): {df['LOS_BINARY'].mean():.1%}")
        print(f"   Normal Stay (0): {1 - df['LOS_BINARY'].mean():.1%}")
    
    # Create AGE_GROUP
    if 'AGE' in df.columns:
        df['AGE_GROUP'] = pd.cut(
            df['AGE'].astype(float),
            bins=CONFIG['age_bins'],
            labels=CONFIG['age_labels'],
            include_lowest=True
        ).astype(str)
        print(f"\n✅ Created AGE_GROUP:")
        print(df['AGE_GROUP'].value_counts().sort_index())
    
    # Show protected attribute distributions
    print(f"\n📊 Protected Attribute Distributions:")
    for attr in ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']:
        if attr in df.columns:
            print(f"\n   {attr}:")
            for val, count in df[attr].value_counts().items():
                pct = count / len(df) * 100
                print(f"      {val}: {count:,} ({pct:.1f}%)")
    
    # Show hospital distribution if available
    if 'HOSPITAL_ID' in df.columns:
        n_hospitals = df['HOSPITAL_ID'].nunique()
        print(f"\n✅ Found {n_hospitals} unique hospitals")
    
    return df


def encode_features(df):
    """Encode categorical features for ML."""
    print("\n" + "=" * 70)
    print("STEP 4: ENCODING FEATURES")
    print("=" * 70)
    
    label_encoders = {}
    categorical_cols = ['SEX', 'RACE', 'ETHNICITY', 'AGE_GROUP']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_ENC'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"   {col}: {list(le.classes_)}")
    
    return df, label_encoders


def create_ml_data(df):
    """Create train/test splits and feature matrices."""
    print("\n" + "=" * 70)
    print("STEP 5: CREATING ML DATA")
    print("=" * 70)
    
    # Identify feature columns
    feature_cols = []
    
    # Numerical features
    for col in ['AGE', 'TOTAL_CHARGES', 'CHARGES']:
        if col in df.columns:
            feature_cols.append(col)
    
    # Encoded categorical features
    for col in ['SEX_ENC', 'RACE_ENC', 'ETHNICITY_ENC', 'AGE_GROUP_ENC']:
        if col in df.columns:
            feature_cols.append(col)
    
    # If we have very few features, add more
    if len(feature_cols) < 3:
        # Try to find any numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in feature_cols and col not in ['LOS_BINARY', 'LENGTH_OF_STAY']:
                feature_cols.append(col)
                if len(feature_cols) >= 10:
                    break
    
    print(f"\n📊 Feature columns: {feature_cols}")
    
    # Handle any remaining missing values
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    X = df[feature_cols].values
    y = df['LOS_BINARY'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)),
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    print(f"\n✅ Train/Test split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing: {len(X_test):,} samples")
    
    # Extract protected attributes
    protected = {}
    subgroups = {}
    for attr in CONFIG['protected_attributes']:
        if attr in df.columns:
            protected[attr] = df[attr].values
            subgroups[attr] = sorted(df[attr].dropna().unique().tolist())
            print(f"\n✅ {attr}: {len(subgroups[attr])} subgroups")
    
    # Hospital IDs
    hospital_ids = df['HOSPITAL_ID'].values if 'HOSPITAL_ID' in df.columns else None
    if hospital_ids is not None:
        print(f"\n✅ Hospital IDs: {len(np.unique(hospital_ids))} unique hospitals")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'idx_train': idx_train, 'idx_test': idx_test,
        'X_scaled': X_scaled, 'y': y,
        'protected': protected, 'subgroups': subgroups,
        'hospital_ids': hospital_ids,
        'feature_cols': feature_cols, 'scaler': scaler
    }


def save_data(data, df, label_encoders):
    """Save all processed data."""
    print("\n" + "=" * 70)
    print("STEP 6: SAVING DATA")
    print("=" * 70)
    
    out = CONFIG['output_dir']
    
    # Save numpy arrays
    np.save(out / 'X_train.npy', data['X_train'])
    np.save(out / 'X_test.npy', data['X_test'])
    np.save(out / 'y_train.npy', data['y_train'])
    np.save(out / 'y_test.npy', data['y_test'])
    np.save(out / 'idx_train.npy', data['idx_train'])
    np.save(out / 'idx_test.npy', data['idx_test'])
    np.save(out / 'X_scaled.npy', data['X_scaled'])
    np.save(out / 'y.npy', data['y'])
    
    if data['hospital_ids'] is not None:
        np.save(out / 'hospital_ids.npy', data['hospital_ids'])
    
    # Save protected attributes
    with open(out / 'protected_attributes.pkl', 'wb') as f:
        pickle.dump({'protected': data['protected'], 'subgroups': data['subgroups']}, f)
    
    # Save preprocessing info
    with open(out / 'preprocessing_info.pkl', 'wb') as f:
        pickle.dump({
            'feature_cols': data['feature_cols'],
            'scaler': data['scaler'],
            'label_encoders': label_encoders,
            'config': CONFIG
        }, f)
    
    # Save processed DataFrame
    df.to_csv(out / 'processed_data.csv', index=False)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(data['y']),
        'n_train': len(data['y_train']),
        'n_test': len(data['y_test']),
        'positive_rate': float(data['y'].mean()),
        'features': data['feature_cols'],
        'protected_attributes': list(data['protected'].keys()),
        'subgroups': data['subgroups'],
        'n_hospitals': int(len(np.unique(data['hospital_ids']))) if data['hospital_ids'] is not None else 0
    }
    
    with open(out / 'data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ All data saved to {out}/")
    for f in sorted(out.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name}: {size_kb:.1f} KB")


def main():
    print("\n" + "🏥" * 30)
    print("\n  SCRIPT 1: DATA PREPROCESSING")
    print("  Texas-100X Fairness Analysis")
    print("\n" + "🏥" * 30)
    
    start = datetime.now()
    
    # Load data
    df = load_data()
    
    # Identify columns
    column_mapping = identify_columns(df)
    
    # Preprocess
    df = preprocess_data(df, column_mapping)
    
    # Encode
    df, label_encoders = encode_features(df)
    
    # Create ML data
    data = create_ml_data(df)
    
    # Save
    save_data(data, df, label_encoders)
    
    elapsed = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"Total samples: {len(data['y']):,}")
    print(f"Protected attributes: {list(data['protected'].keys())}")
    print(f"Total subgroups: {sum(len(v) for v in data['subgroups'].values())}")
    
    if data['hospital_ids'] is not None:
        print(f"Hospitals: {len(np.unique(data['hospital_ids']))}")
    
    print("\n👉 NEXT: python script2_model_training.py")


if __name__ == "__main__":
    main()
