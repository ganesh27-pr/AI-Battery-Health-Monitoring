"""Train regression candidates with a battery-group held-out test split."""
from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from .config import FEATURES, TARGET, MODEL_DIR, METRICS_DIR
from .preprocessing import make_preprocessor, clean_frame
from .evaluate import regression_metrics

def train_regression(df):
    df = clean_frame(df).dropna(subset=[TARGET]).copy()
    x, y = df[FEATURES], df[TARGET]
    groups = df['battery_id'] if 'battery_id' in df else pd.Series(range(len(df)))
    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=1/groups.nunique(), random_state=42)
        tr, te = next(splitter.split(x,y,groups))
    else:
        from sklearn.model_selection import train_test_split
        tr,te = train_test_split(range(len(df)),test_size=.2,random_state=42)
    candidates = {'Dummy':DummyRegressor(strategy='mean'), 'LinearRegression':LinearRegression(),
      'DecisionTree':DecisionTreeRegressor(max_depth=5,random_state=42),
      'RandomForest':RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42),
      'GradientBoosting':GradientBoostingRegressor(random_state=42)}
    results, fitted = {}, {}
    for name, est in candidates.items():
        pipe=Pipeline([('preprocess',make_preprocessor()),('model',est)])
        pipe.fit(x.iloc[tr],y.iloc[tr]); pred=pipe.predict(x.iloc[te])
        results[name]=regression_metrics(y.iloc[te],pred); fitted[name]=pipe
    best=min(results,key=lambda k:results[k]['MAE'])
    MODEL_DIR.mkdir(parents=True,exist_ok=True); METRICS_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump(fitted[best],MODEL_DIR/'regression_model.joblib')
    payload={'task':'SOH regression','selected_model':best,'split':'battery-group holdout when battery_id is available',
      'test_batteries':sorted(groups.iloc[te].unique().tolist()),'feature_names':FEATURES,'target':TARGET,'target_units':'percent','results':results}
    (METRICS_DIR/'regression_metrics.json').write_text(json.dumps(payload,indent=2))
    pd.DataFrame({'actual_soh':y.iloc[te].values,'predicted_soh':fitted[best].predict(x.iloc[te])}).to_csv(METRICS_DIR/'regression_test_predictions.csv',index=False)
    return payload, fitted[best], (x.iloc[te],y.iloc[te])
