"""Validation and leakage-conscious preprocessing helpers."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import FEATURES, TARGET

def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='number').columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if TARGET in df:
        df = df[df[TARGET].between(0, 120) | df[TARGET].isna()]
    return df.reset_index(drop=True)

def validate_features(frame: pd.DataFrame, features=FEATURES):
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError('Missing required model features: ' + ', '.join(missing))
    return frame.loc[:, features].copy()

def make_preprocessor(features=FEATURES):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    return ColumnTransformer([('numeric', numeric, list(features))], remainder='drop')

def health_label(soh):
    if pd.isna(soh): return 'Unknown'
    if soh >= 80: return 'Healthy'
    if soh >= 60: return 'Moderate'
    return 'Degraded'
