"""Load saved estimators and make single or batch predictions."""
from pathlib import Path
import joblib
import pandas as pd
from .config import MODEL_DIR, FEATURES
from .preprocessing import validate_features, health_label
from .anomaly_detection import detect_anomalies

def load_model(task='regression', model_dir=MODEL_DIR):
    path = Path(model_dir) / f'{task}_model.joblib'
    if not path.exists():
        raise FileNotFoundError(f'Model not found: {path}. Run python scripts/train_all.py after preparing the dataset.')
    return joblib.load(path)

def predict_frame(frame, model_dir=MODEL_DIR):
    x = validate_features(frame)
    reg = load_model('regression', model_dir)
    clf_path = Path(model_dir) / 'classification_model.joblib'
    result = frame.copy()
    result['predicted_soh_percent'] = reg.predict(x).clip(0, 120)
    result['predicted_health_class'] = result['predicted_soh_percent'].map(health_label)
    result['warnings'] = [ '; '.join(detect_anomalies(r, s)) for r, s in zip(result.to_dict('records'), result['predicted_soh_percent']) ]
    if clf_path.exists():
        clf = joblib.load(clf_path)
        result['model_health_class'] = clf.predict(x)
    return result
