"""Load raw NASA MATLAB files, train models and save outputs."""
import json
from pathlib import Path
import pandas as pd
from src.data_loader import load_nasa_batteries
from src.train_regression import train_regression
from src.train_classification import train_classification
from src.config import PROCESSED_DIR, MODEL_DIR

def main():
    df=load_nasa_batteries(); PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    df.to_csv(PROCESSED_DIR/'nasa_discharge_cycles.csv',index=False)
    reg,_,_=train_regression(df); clf,_,_=train_classification(df)
    metadata={'dataset':'NASA PCoE Li-ion Battery Data Set','dataset_source':'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip','target':'capacity-derived SOH (%)','features':list(__import__('src.config',fromlist=['FEATURES']).FEATURES),'regression':reg,'classification':clf,'limitations':['Small number of independent batteries; battery-group test split is essential.','SOH labels are calculated from discharge capacity and nominal 2 Ah rating.','Voltage/current/temperature summaries are cycle-level features, not a certified BMS.']}
    (MODEL_DIR/'metadata.json').write_text(json.dumps(metadata,indent=2)); print(json.dumps(metadata,indent=2))
if __name__=='__main__': main()
