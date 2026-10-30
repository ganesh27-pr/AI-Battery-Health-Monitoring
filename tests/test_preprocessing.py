import pandas as pd
import pytest
from src.preprocessing import clean_frame, validate_features, health_label
from src.config import FEATURES

def test_clean_frame_removes_duplicates_and_invalid_soh():
    df=pd.DataFrame({'soh_percent':[99,99,140], 'x':[1,1,2]})
    out=clean_frame(df)
    assert len(out)==1

def test_validate_features_reports_missing():
    with pytest.raises(ValueError,match='Missing required'):
        validate_features(pd.DataFrame({'cycle_index':[1]}))

def test_health_thresholds():
    assert health_label(80)=='Healthy'
    assert health_label(79.9)=='Moderate'
    assert health_label(59.9)=='Degraded'
