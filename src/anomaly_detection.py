"""Rule-based warnings; configured values are demonstrations, not safety limits."""
from .config import DEMO_LIMITS

def detect_anomalies(record, soh=None, limits=None):
    limits = {**DEMO_LIMITS, **(limits or {})}
    warnings = []
    required = ['voltage_mean_v', 'current_mean_a', 'temperature_mean_c']
    if any(k not in record or record[k] is None for k in required):
        return ['Insufficient Data for Reliable Monitoring']
    try:
        v, i, t = (float(record[k]) for k in required)
    except (TypeError, ValueError):
        return ['Invalid or Non-numeric Measurement']
    if not all(map(lambda x: x == x and abs(x) != float('inf'), [v, i, t])):
        return ['Invalid or Non-finite Measurement']
    if not limits['voltage_min_v'] <= v <= limits['voltage_max_v']:
        warnings.append('Voltage Outside Configured Range')
    if abs(i) > limits['current_abs_max_a']:
        warnings.append('Current Outside Configured Range')
    if t > limits['temperature_max_c']:
        warnings.append('High Temperature Warning')
    if soh is not None and float(soh) < 60:
        warnings.append('Battery Health Degraded')
    return warnings or ['Normal Operating Conditions']
