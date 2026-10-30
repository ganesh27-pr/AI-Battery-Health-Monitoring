import pandas as pd
import pytest
from src.predict import load_model, predict_frame
from src.config import FEATURES

def test_missing_model_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError,match='Run python scripts/train_all.py'):
        load_model('regression',tmp_path)

def test_prediction_rejects_missing_features():
    with pytest.raises(ValueError,match='Missing required'):
        predict_frame(pd.DataFrame({'cycle_index':[1]}),model_dir='/missing')
