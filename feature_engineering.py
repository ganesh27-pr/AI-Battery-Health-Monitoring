"""Feature engineering for discharge-cycle summary measurements."""
import pandas as pd
from .config import FEATURES

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f'Dataset does not contain required measured features: {missing}')
    return frame.loc[:, FEATURES].copy()
