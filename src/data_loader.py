"""Load NASA PCoE MATLAB battery cycles and convert discharge cycles to tabular rows."""
from pathlib import Path
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .config import RAW_DIR, RATED_CAPACITY_AH

NASA_URL = 'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip'

def download_nasa_dataset(destination: Path = RAW_DIR / 'NASA_Battery_Data_Set.zip') -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            urllib.request.urlretrieve(NASA_URL, destination)
        except Exception as exc:
            raise RuntimeError(f'Could not download NASA dataset. Download it manually from {NASA_URL} and place the ZIP at {destination}. Original error: {exc}') from exc
    return destination

def _scalar(x):
    return float(np.asarray(x).squeeze())

def _field(obj, name):
    """Read a MATLAB struct field across scipy's common representations."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name].item()
    raise KeyError(name)

def _cycles_from_mat(mat_path: Path, rated_capacity_ah: float = RATED_CAPACITY_AH):
    raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    battery_key = mat_path.stem
    if battery_key not in raw:
        candidates = [k for k in raw if not k.startswith('__')]
        if not candidates:
            raise ValueError(f'No battery struct found in {mat_path}')
        battery_key = candidates[0]
    battery = raw[battery_key]
    cycles = _field(battery, 'cycle')
    rows = []
    for i, cycle in enumerate(np.atleast_1d(cycles), start=1):
        try:
            kind = str(_field(cycle, 'type')).lower()
            data = _field(cycle, 'data')
            if 'discharge' not in kind:
                continue
            voltage = np.asarray(_field(data, 'Voltage_measured'), dtype=float).reshape(-1)
            current = np.asarray(_field(data, 'Current_measured'), dtype=float).reshape(-1)
            temp = np.asarray(_field(data, 'Temperature_measured'), dtype=float).reshape(-1)
            capacity = _scalar(_field(data, 'Capacity'))
            n = min(len(voltage), len(current), len(temp))
            voltage, current, temp = voltage[:n], current[:n], temp[:n]
            valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temp)
            voltage, current, temp = voltage[valid], current[valid], temp[valid]
            if not len(voltage) or not np.isfinite(capacity) or capacity <= 0:
                continue
            rows.append({'battery_id': mat_path.stem, 'cycle_index': i,
                'voltage_mean_v': voltage.mean(), 'voltage_min_v': voltage.min(), 'voltage_max_v': voltage.max(),
                'current_mean_a': current.mean(), 'current_min_a': current.min(), 'current_max_a': current.max(),
                'temperature_mean_c': temp.mean(), 'temperature_max_c': temp.max(),
                'voltage_std_v': voltage.std(), 'current_std_a': current.std(), 'temperature_std_c': temp.std(),
                'discharge_capacity_ah': capacity, 'soh_percent': capacity / rated_capacity_ah * 100.0})
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return rows

def load_nasa_batteries(raw_dir: Path = RAW_DIR, extract: bool = True) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    mat_files = list(raw_dir.rglob('*.mat'))
    if not mat_files:
        zips = list(raw_dir.glob('*.zip'))
        if not zips:
            raise FileNotFoundError(f'No NASA .mat files or ZIP found in {raw_dir}. Run python scripts/download_data.py.')
        if extract:
            with zipfile.ZipFile(zips[0]) as zf:
                zf.extractall(raw_dir / 'NASA_Battery_Data_Set')
            mat_files = list((raw_dir / 'NASA_Battery_Data_Set').rglob('*.mat'))
    rows = []
    for path in mat_files:
        rows.extend(_cycles_from_mat(path))
    if not rows:
        raise ValueError('No usable discharge cycles found. Check that the official NASA ZIP was extracted correctly.')
    return pd.DataFrame(rows).sort_values(['battery_id', 'cycle_index']).reset_index(drop=True)

def load_csv(path):
    return pd.read_csv(path)
