
import os, yaml, pathlib
from typing import Any, Dict

_DEFAULT = {
  "thresholds": {
    "refrigerator": {
      "air_return_c_max": 7.0,
      "milk_surface_c_max": 6.0,
      "chill_mw_surface_c_max": 8.0,
      "rise_c_per_min_max": 0.5
    },
    "freezer": {
      "air_return_c_max": -15.0,
      "mw_freeze_surface_c_max": -12.0,
      "rise_c_per_min_max": 0.4
    }
  },
  "defrost": {"grace_min": 5, "penalty_factor": 0.8},
  "weights": {
    "w_time": 0.35, "w_energy": 0.25, "w_risk": 0.20,
    "w_open": 0.05, "w_dload": 0.10, "w_defrost": 0.05
  }
}

def load_config() -> Dict[str, Any]:
    cfg_path = os.environ.get("CONFIG_PATH")
    if not cfg_path:
        # default to project config/config.yaml
        root = pathlib.Path(__file__).resolve().parents[2]
        cfg_path = root / "config" / "config.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # merge minimal defaults
        return _merge(_DEFAULT, data)
    except Exception:
        return _DEFAULT

def _merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _merge(base.get(k), v) if k in base else v
        return out
    return override if override is not None else base
