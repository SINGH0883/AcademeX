"""
Data Science Fundamentals and Statistical Intuition
Project-1: Predictive Analytics for Student Performance in Exams
Module: Preprocessing and Data Cleaning
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_raw_data(filepath='data/student-mat.csv'):
    """Load raw dataset with semicolon separator."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath, sep=';')
    return df

def clean_and_preprocess(df):
    """
    Cleans data, handles missing values, encodes categorical variables,
    and engineers relevant academic metrics.
    """
    df_clean = df.copy()

    # 1. Missing Value Check & Imputation (if any exist)
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 2. Binary categorical mapping (0 / 1)
    binary_maps = {
        'school': {'GP': 1, 'MS': 0},
        'sex': {'F': 0, 'M': 1},
        'address': {'U': 1, 'R': 0},       # Urban: 1, Rural: 0
        'famsize': {'GT3': 1, 'LE3': 0},   # Greater than 3: 1, Less or equal: 0
        'Pstatus': {'T': 1, 'A': 0},       # Together: 1, Apart: 0
        'schoolsup': {'yes': 1, 'no': 0},
        'famsup': {'yes': 1, 'no': 0},
        'paid': {'yes': 1, 'no': 0},
        'activities': {'yes': 1, 'no': 0},
        'nursery': {'yes': 1, 'no': 0},
        'higher': {'yes': 1, 'no': 0},
        'internet': {'yes': 1, 'no': 0},
        'romantic': {'yes': 1, 'no': 0}
    }

    for col, mapping in binary_maps.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping)

    # 3. Nominal multi-class categorical encoding using one-hot encoding
    nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
    df_clean = pd.get_dummies(df_clean, columns=nominal_cols, drop_first=True, dtype=int)

    # 4. Feature Engineering:
    # Prior average grade (G1 + G2) / 2
    df_clean['prior_avg_grade'] = (df_clean['G1'] + df_clean['G2']) / 2.0
    # Academic support index
    df_clean['support_index'] = df_clean['schoolsup'] + df_clean['famsup'] + df_clean['paid']
    # Pass status (classification target: G3 >= 10 out of 20)
    df_clean['passed'] = (df_clean['G3'] >= 10).astype(int)

    return df_clean

def get_train_test_data(data_path='data/student-mat.csv', test_size=0.2, random_state=42):
    """
    Returns train and test splits for regression and classification evaluation.
    """
    raw_df = load_raw_data(data_path)
    clean_df = clean_and_preprocess(raw_df)
    
    # Save clean dataset
    os.makedirs('data', exist_ok=True)
    clean_df.to_csv('data/student_clean.csv', index=False)

    # Features (exclude target G3 and binary passed)
    feature_cols = [c for c in clean_df.columns if c not in ['G3', 'passed']]
    X = clean_df[feature_cols]
    y_reg = clean_df['G3']
    y_cls = clean_df['passed']

    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=test_size, random_state=random_state, stratify=y_cls
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'y_train_cls': y_train_cls,
        'y_test_cls': y_test_cls,
        'feature_names': feature_cols,
        'clean_df': clean_df,
        'raw_df': raw_df
    }

if __name__ == '__main__':
    data_dict = get_train_test_data()
    print("Preprocessing completed successfully!")
    print(f"Dataset shape: {data_dict['clean_df'].shape}")
    print(f"Training samples: {len(data_dict['X_train'])}, Testing samples: {len(data_dict['X_test'])}")
    print(f"Total features: {len(data_dict['feature_names'])}")
    print(f"Cleaned dataset saved to 'data/student_clean.csv'")
