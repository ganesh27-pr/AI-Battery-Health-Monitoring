"""Train battery health classifier from capacity-derived SOH labels."""
from pathlib import Path
import json, joblib
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from .config import FEATURES, TARGET, MODEL_DIR, METRICS_DIR
from .preprocessing import make_preprocessor, clean_frame, health_label
from .evaluate import classification_metrics

def train_classification(df):
    df=clean_frame(df).dropna(subset=[TARGET]).copy(); df['health_class']=df[TARGET].map(health_label)
    if df.health_class.nunique()<2: return {'status':'Skipped: dataset contains fewer than two health classes'}, None, None
    x,y=df[FEATURES],df.health_class
    groups=df['battery_id'] if 'battery_id' in df else None
    if groups is not None and groups.nunique()>=2:
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=1/groups.nunique(),random_state=42).split(x,y,groups))
    else: tr,te=train_test_split(range(len(df)),test_size=.2,random_state=42,stratify=y)
    candidates={'Dummy':DummyClassifier(strategy='most_frequent'), 'DecisionTree':DecisionTreeClassifier(max_depth=5,class_weight='balanced',random_state=42),
      'RandomForest':RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced',random_state=42)}
    # Gradient boosting is included only when the training partition contains all needed classes.
    if y.iloc[tr].nunique()>=2: candidates['GradientBoosting']=GradientBoostingClassifier(random_state=42)
    results,fitted={},{}
    for name,est in candidates.items():
        pipe=Pipeline([('preprocess',make_preprocessor()),('model',est)])
        pipe.fit(x.iloc[tr],y.iloc[tr]); results[name]=classification_metrics(y.iloc[te],pipe.predict(x.iloc[te])); fitted[name]=pipe
    best=max((k for k in results if k!='Dummy'),key=lambda k:results[k]['f1_macro'])
    MODEL_DIR.mkdir(parents=True,exist_ok=True); METRICS_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump(fitted[best],MODEL_DIR/'classification_model.joblib')
    payload={'task':'health classification','selected_model':best,'split':'battery-group holdout when available','test_batteries':sorted(groups.iloc[te].unique().tolist()) if groups is not None else [],'results':results}
    (METRICS_DIR/'classification_metrics.json').write_text(json.dumps(payload,indent=2))
    return payload,fitted[best],(x.iloc[te],y.iloc[te])
