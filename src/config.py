"""Shared project configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
MODEL_DIR = ROOT / 'models'
METRICS_DIR = ROOT / 'outputs' / 'metrics'
FIGURES_DIR = ROOT / 'outputs' / 'figures'
RATED_CAPACITY_AH = 2.0  # NASA cells' nominal capacity used in this project definition
SOH_HEALTHY = 80.0
SOH_MODERATE = 60.0
# Demonstration operating limits only; replace with manufacturer/chemistry-specific limits.
DEMO_LIMITS = {'voltage_min_v': 2.7, 'voltage_max_v': 4.2,
               'current_abs_max_a': 5.0, 'temperature_max_c': 45.0}
FEATURES = ['cycle_index', 'voltage_mean_v', 'voltage_min_v', 'voltage_max_v',
            'current_mean_a', 'current_min_a', 'current_max_a',
            'temperature_mean_c', 'temperature_max_c', 'voltage_std_v',
            'current_std_a', 'temperature_std_c']
TARGET = 'soh_percent'
