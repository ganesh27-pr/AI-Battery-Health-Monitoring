from src.anomaly_detection import detect_anomalies

def test_normal_demo_values():
    rec={'voltage_mean_v':3.7,'current_mean_a':-1.2,'temperature_mean_c':25}
    assert detect_anomalies(rec,90)==['Normal Operating Conditions']

def test_temperature_warning():
    rec={'voltage_mean_v':3.7,'current_mean_a':-1.2,'temperature_mean_c':50}
    assert 'High Temperature Warning' in detect_anomalies(rec,90)

def test_missing_input():
    assert detect_anomalies({},None)==['Insufficient Data for Reliable Monitoring']
