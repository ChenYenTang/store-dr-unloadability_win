# src/io/company_api.py
# v0.6.0 新增：封裝公司 API 的 3 GET / 1 POST 與欄位映射
import os, json
import httpx
from typing import Dict, List, Any, Tuple

# ========== 小工具：dot path 取值（a.b.c） ==========
def _get_by_dot(obj: Any, path: str):
    """從巢狀 dict 依 'a.b.c' 取值；取不到回 None"""
    if not path:
        return None
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

# ========== 建立 HTTP client（處理認證/逾時/SSL） ==========
def _build_client(params: Dict) -> httpx.Client:
    headers = {}
    auth_type = (params.get("auth_type") or "api_key").lower()  # 預設 api_key
    if auth_type == "bearer":
        token = params.get("token") or os.environ.get("COMPANY_API_TOKEN", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "api_key":
        key_name = params.get("api_key_header") or "X-API-Key"
        key_val = params.get("api_key") or os.environ.get("COMPANY_API_KEY", "")
        if key_val:
            headers[key_name] = key_val
    timeout = int(params.get("timeout_sec") or 10)
    verify = bool(params.get("verify_ssl", True))
    base = (params.get("base_url") or os.environ.get("COMPANY_API_BASE_URL","")).rstrip("/")
    return httpx.Client(base_url=base, headers=headers, timeout=timeout, verify=verify)

# ========== 測試連線（以 /stores smoke test） ==========
def test_connection(params: Dict) -> Tuple[bool, str]:
    try:
        with _build_client(params) as cli:
            r = cli.get("/stores")
            return (r.status_code < 500, f"status={r.status_code}")
    except Exception as e:
        return (False, f"error={e}")

# ========== GET：列門市 ==========
def fetch_stores(params: Dict) -> List[Dict[str, Any]]:
    ep = params.get("endpoints",{}).get("list_stores", "/stores")
    with _build_client(params) as cli:
        r = cli.get(ep); r.raise_for_status()
        data = r.json()
    items = data["stores"] if isinstance(data, dict) and "stores" in data else data
    out = []
    for it in (items or []):
        _id = it.get("id") or it.get("store_id") or it.get("Store_ID")
        _name = it.get("name") or _id
        if _id:
            out.append({"id": _id, "name": _name})
    return out

# ========== GET：門市設備 ==========
def fetch_devices(params: Dict, store_id: str) -> List[Dict[str, Any]]:
    tpl = params.get("endpoints",{}).get("list_devices_of_store", "/stores/{store_id}/devices")
    path = tpl.replace("{store_id}", str(store_id))
    with _build_client(params) as cli:
        r = cli.get(path); r.raise_for_status()
        data = r.json()
    return data["devices"] if isinstance(data, dict) and "devices" in data else (data if isinstance(data, list) else [])

# ========== GET：設備即時（→ 轉成系統 devices 欄位） ==========
def fetch_realtime(params: Dict, store_id: str, device_ids: List[str], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    # 預設改為路徑版；若你的端點沒有 {store_id} 也沒關係，format 只會替換存在的 placeholder
    tpl = params.get("endpoints",{}).get("realtime_of_device", "/stores/{store_id}/devices/{device_id}/realtime")
    out = []
    with _build_client(params) as cli:
        for did in device_ids:
            # 正確同時套入 store_id 與 device_id（即使模板沒有某個 placeholder 也不會出錯）
            path = tpl.format(store_id=store_id, device_id=str(did))
            try:
                r = cli.get(path); r.raise_for_status()
                raw = r.json()
                payload = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
            except Exception:
                continue
            dev = {
                "Device_ID": _get_by_dot(payload, mapping.get("device_id","")) or str(did),
                "Device_type": _get_by_dot(payload, mapping.get("device_type","")),
                "Temp_current": _get_by_dot(payload, mapping.get("temp_current","")),
                "T_room": _get_by_dot(payload, mapping.get("t_room","")),
                "Power": _get_by_dot(payload, mapping.get("power","")),
                "Defrost_status": int(_get_by_dot(payload, mapping.get("defrost_status","")) or 0),
                "timestamp": _get_by_dot(payload, mapping.get("timestamp","")),
            }
            out.append(dev)
    return out

# ========== POST：送出卸載優先順序 ==========
def post_priority(params: Dict, payload: Dict) -> Tuple[bool, int, str]:
    """
    payload 結構：{"Store_ID":"S001","devices":[...]}
    回傳：(ok, status_code, text_snippet)
    """
    ep = params.get("endpoints",{}).get("post_priority", "/output/priority.json")
    try:
        with _build_client(params) as cli:
            r = cli.post(ep, json=payload)
            ok = r.status_code < 500
            txt = (r.text or "")[:200]
            return (ok, r.status_code, txt)
    except Exception as e:
        return (False, 0, f"error={e}")

# ========== 寫回 config.yaml 的 api 區塊（敏感值以 env 名保存） ==========
def save_api_settings(params: Dict, prefer_env: bool = True):
    import yaml
    os.makedirs("config", exist_ok=True)
    path = os.path.join("config","config.yaml")
    cfg = {}
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    api = {
        "base_url": params.get("base_url"),
        "auth": {"type": params.get("auth_type","api_key")},  # 預設 api_key
        "timeout_sec": int(params.get("timeout_sec") or 10),
        "verify_ssl": bool(params.get("verify_ssl", True)),
        "endpoints": params.get("endpoints", {
            "list_stores": "/stores",
            "list_devices_of_store": "/stores/{store_id}/devices",
            "realtime_of_device": "/devices/{device_id}/realtime",
            "post_priority": "/output/priority.json",
        })},