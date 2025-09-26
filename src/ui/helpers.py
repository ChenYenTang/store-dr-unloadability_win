import os, json, yaml
import gradio as gr
from datetime import datetime, timezone
from src.io.schema import EvaluateRequest, CabinetInput

DEFAULT_CONFIG_YAML = """
thresholds:
  refrigerator:
    air_return_c_max: 7.0
    milk_surface_c_max: 6.0
    chill_mw_surface_c_max: 8.0
    rise_c_per_min_max: 0.5
  freezer:
    air_return_c_max: -15.0
    mw_freeze_surface_c_max: -12.0
    rise_c_per_min_max: 0.4
defrost:
  grace_min: 5
  penalty_factor: 0.8
weights:
  w_time: 0.35
  w_energy: 0.25
  w_risk: 0.20
  w_open: 0.05
  w_dload: 0.10
  w_defrost: 0.05
""".strip()

CABINET_COLUMNS = [
    "cabinet_id","type","air_supply_c","air_return_c",
    "prod_t_mw_chill_c","prod_t_milk_c","prod_t_mw_freeze_c",
    "defrost_status","time_since_defrost_min",
]
CABINET_DTYPES = ["str","str","number","number","number","number","number","number","number"]

def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def load_yaml_from_file(file_obj) -> str:
    if file_obj is None: return DEFAULT_CONFIG_YAML
    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except Exception as e:
        return f"# Failed to read YAML: {e}\\n\\n{DEFAULT_CONFIG_YAML}"

def validate_config_yaml(yaml_text: str):
    try:
        data = yaml.safe_load(yaml_text)
        assert "thresholds" in data and "defrost" in data and "weights" in data
        return {"ok": True, "message": "YAML OK"}
    except Exception as e:
        return {"ok": False, "message": f"Invalid YAML: {e}"}

def save_config_yaml(yaml_text: str) -> str:
    out_dir = os.environ.get("CONFIG_DIR", "config")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return path

def gen_default_rows(num_ref: int, num_fz: int):
    rows = []
    for i in range(int(num_ref)):
        rows.append([f"R-{i+1:02d}", "refrigerator", None, None, None, None, None, 0, 0])
    for j in range(int(num_fz)):
        rows.append([f"F-{j+1:02d}", "freezer", None, None, None, None, None, 0, 0])
    return rows

def generate_table(num_ref: float, num_fz: float):
    return gen_default_rows(int(num_ref), int(num_fz))

def _coerce_float(v):
    if v is None: return None
    try:
        fv = float(v)
        if fv != fv: return None
        return fv
    except Exception:
        return None

def assemble_payload(store_id: str, business_flag: str, timestamp: str):
    """組合 payload，來源是 input/latest.json"""
    latest_path = os.path.join("input", "latest.json")
    if not os.path.exists(latest_path):
        return gr.update(value=None), gr.update(value=None), "⚠️ 找不到 input/latest.json"

    try:
        with open(os.path.join("input","config.json"), "r", encoding="utf-8") as f:
            _cfg = json.load(f) or {}
        _store_id_for_latest = (
            _cfg.get("Active_Store_ID")
            or (_cfg.get("Store_IDs") or [None])[0]
            or _cfg.get("Store_ID")
        )
        _latest_path = os.path.join("input","latest", f"{_store_id_for_latest}.json")
        with open(_latest_path,"r",encoding="utf-8") as f:
        # 分店快照根節點：{"Store_ID": "...", "generated_at": "...", "devices":[...]}
            devices = (json.load(f) or {}).get("devices", [])
    except Exception as e:
        return gr.update(value=None), gr.update(value=None), f"⚠️ 讀取 latest.json 失敗: {e}"

    cabinets = []
    for d in devices:
        try:
            cab = CabinetInput(
                cabinet_id=d.get("Device_ID", ""),
                type=d.get("Device_type", ""),
                air_supply_c=None,  # latest.json 沒有，保留 None
                air_return_c=None,
                prod_t_mw_chill_c=None,
                prod_t_milk_c=None,
                prod_t_mw_freeze_c=None,
                defrost_status=int(d.get("Defrost_status", 0)),
                time_since_defrost_min=0,  # 計算模組才有
            )
            cabinets.append(cab.dict())
        except Exception as e:
            continue

    if not timestamp.strip():
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    req = EvaluateRequest(
        store_id=store_id.strip() or "S001",
        timestamp=timestamp.strip(),
        business_hours_flag=int(business_flag),
        cabinets=cabinets,
    )

    payload = json.loads(req.json())

    out_path = os.path.abspath("payload.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload, out_path, "OK"

