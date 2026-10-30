"""Streamlit dashboard for battery SOH prediction."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.config import FEATURES, DEMO_LIMITS
from src.predict import predict_frame, load_model
from src.anomaly_detection import detect_anomalies

st.set_page_config(page_title='Battery Health Monitor',page_icon='🔋',layout='wide')
st.title('🔋 AI-Based Battery Health Monitoring System')
st.caption('Machine-learning demonstration using capacity-derived State of Health (SOH). SOH is not State of Charge (SOC).')
with st.sidebar:
    page=st.radio('Navigation',['Overview','Single Prediction','CSV Batch Prediction','Model Performance','About & Methodology'])
    st.info('Research demonstration only. Not a certified battery safety system.')

def model_ready():
    return (Path('models/regression_model.joblib').exists())

if page=='Overview':
    st.subheader('Project overview')
    st.write('Analyze discharge-cycle measurements and estimate battery State of Health. Upload a prepared cycle-summary CSV or enter one measured cycle.')
    st.metric('Required model features',len(FEATURES))
    st.write('**First-time setup:** install requirements, download/extract the NASA dataset, then run `python scripts/train_all.py`.')
    if not model_ready(): st.warning('Trained model not found. Follow the setup instructions before making predictions.')
    else: st.success('Saved regression model found.')
elif page=='Single Prediction':
    st.subheader('Enter one cycle summary')
    with st.form('single'):
        vals={}
        defaults={'cycle_index':1,'voltage_mean_v':3.7,'voltage_min_v':2.8,'voltage_max_v':4.1,'current_mean_a':-1.5,'current_min_a':-2.0,'current_max_a':-0.5,'temperature_mean_c':25.0,'temperature_max_c':27.0,'voltage_std_v':0.2,'current_std_a':0.3,'temperature_std_c':0.5}
        cols=st.columns(3)
        for i,f in enumerate(FEATURES): vals[f]=cols[i%3].number_input(f, value=float(defaults[f]))
        submitted=st.form_submit_button('Estimate battery health')
    if submitted:
        try:
            result=predict_frame(pd.DataFrame([vals])); st.metric('Predicted SOH',f"{result.predicted_soh_percent.iloc[0]:.1f}%")
            st.write('Condition:',result.predicted_health_class.iloc[0]); st.warning(result.warnings.iloc[0]); st.dataframe(result)
        except Exception as e: st.error(str(e))
elif page=='CSV Batch Prediction':
    st.subheader('Batch prediction')
    st.write('CSV must contain all 12 required model features. Extra columns are preserved.')
    file=st.file_uploader('Upload cycle-summary CSV',type=['csv'])
    if file:
        try:
            df=pd.read_csv(file); st.dataframe(df.head())
            chart_cols=[c for c in ['voltage_mean_v','current_mean_a','temperature_mean_c'] if c in df.columns]
            if chart_cols:
                st.subheader('Uploaded measurement trends')
                st.line_chart(df[chart_cols])
            missing=[f for f in FEATURES if f not in df.columns]
            if missing: st.error('Missing required columns: '+', '.join(missing))
            elif st.button('Run batch prediction'):
                result=predict_frame(df); st.dataframe(result)
                st.subheader('Predicted SOH distribution')
                st.bar_chart(result['predicted_soh_percent'])
                st.download_button('Download predictions CSV',result.to_csv(index=False).encode(),file_name='battery_predictions.csv',mime='text/csv')
        except Exception as e: st.error(f'Could not process CSV: {e}')
elif page=='Model Performance':
    for name in ['regression_metrics.json','classification_metrics.json']:
        path=Path('outputs/metrics')/name
        st.subheader(name.replace('_',' ').replace('.json','').title())
        if path.exists(): st.json(json.loads(path.read_text()))
        else: st.info('Metrics are not available yet. Train the models first.')
elif page=='About & Methodology':
    st.subheader('Methodology')
    st.markdown('''1. Load NASA PCoE battery cycling data.\n2. Keep discharge cycles and summarize measured voltage, current and temperature.\n3. Calculate SOH = discharge capacity / nominal rated capacity × 100.\n4. Split by battery identity to reduce battery-level leakage.\n5. Train regression and classification pipelines.\n6. Use saved pipelines for dashboard predictions.''')
    st.warning('The 2.7–4.2 V, 5 A and 45 °C warning settings are demonstration values only. Use the cell manufacturer specifications for real equipment. Predictions do not replace a BMS, protection circuitry, or professional inspection.')
