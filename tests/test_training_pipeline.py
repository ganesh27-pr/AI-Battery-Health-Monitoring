import numpy as np
import pandas as pd
from src.config import FEATURES
import src.train_regression as tr
import src.train_classification as tc

def _synthetic_unit_fixture():
    # Artificial values used only to verify software mechanics; never treated as NASA observations.
    rows=[]
    rng=np.random.default_rng(4)
    for b in range(4):
        for cyc in range(25):
            soh=98-1.7*cyc-2*b
            row={'battery_id':f'fixture_{b}','cycle_index':cyc+1,'soh_percent':soh,'discharge_capacity_ah':soh/50}
            for f in FEATURES:
                if f=='cycle_index': continue
                if f.startswith('voltage'): row[f]=3.7-0.002*cyc+rng.normal(0,.01)
                elif f.startswith('current'): row[f]=-1.2+rng.normal(0,.03)
                elif f.startswith('temperature'): row[f]=25+0.01*cyc+rng.normal(0,.1)
            rows.append(row)
    return pd.DataFrame(rows)

def test_regression_training_mechanics(monkeypatch,tmp_path):
    monkeypatch.setattr(tr,'MODEL_DIR',tmp_path/'models')
    monkeypatch.setattr(tr,'METRICS_DIR',tmp_path/'metrics')
    payload,model,test=tr.train_regression(_synthetic_unit_fixture())
    assert payload['selected_model'] in payload['results']
    assert (tmp_path/'models'/'regression_model.joblib').exists()
    assert len(test[0])>0

def test_classification_training_mechanics(monkeypatch,tmp_path):
    monkeypatch.setattr(tc,'MODEL_DIR',tmp_path/'models')
    monkeypatch.setattr(tc,'METRICS_DIR',tmp_path/'metrics')
    payload,model,test=tc.train_classification(_synthetic_unit_fixture())
    assert payload['selected_model'] in payload['results']
    assert (tmp_path/'models'/'classification_model.joblib').exists()
