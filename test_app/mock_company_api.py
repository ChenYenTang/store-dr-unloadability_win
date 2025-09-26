"""
test_app/mock_company_api.py
目的：提供 3 個 GET Mock API，模擬公司後端（每分鐘重生資料、部分店/設備沒資料）

端點：
  GET /stores
  GET /stores/{store_id}/devices
  GET /stores/{store_id}/devices/{device_id}/realtime

行為重點：
- 店號固定 S001..S050（name：店001..店050）
- 每「分鐘」決定一批「沒資料」的店（預設 15%）
- 有資料的店：每店 4–12 台櫃類（freezer/refrigerator/open_shelf）+ 0–2 台 hvac
- 單台裝置本輪 5% 機率 204（模擬缺測）
- 櫃類 realtime：中文 Device_type（冷凍/冷藏/開放櫃），溫度欄位 Temp_current；冷凍/冷藏有 Defrost_status
- hvac realtime：中文 Device_type（空調），溫度欄位 T_room

除霜週期：
- 每 4-6 小時發生一次除霜
- 每次持續約 20 分鐘
- 只有冷凍和冷藏設備會除霜

可調參數（環境變數）：
- MOCK_SEED       預設 42（決定亂數可重現）
- NO_DATA_RATE    預設 0.15（店級沒資料機率）
- MISS_RATE       預設 0.05（設備級 204 機率）
"""

from fastapi import FastAPI, HTTPException, Response
from typing import List, Dict, Tuple, Any
from random import Random
from datetime import datetime, timezone, timedelta
import os

# ========== 參數與常數 ==========
SEED = int(os.environ.get("MOCK_SEED", "42"))
NO_DATA_RATE = float(os.environ.get("NO_DATA_RATE", "0.001"))
MISS_RATE = float(os.environ.get("MISS_RATE", "0.005"))

# 除霜相關常數
DEFROST_CYCLE_MIN = 4 * 60      # 4小時 = 240分鐘
DEFROST_CYCLE_MAX = 6 * 60      # 6小時 = 360分鐘
DEFROST_DURATION = 20           # 除霜持續20分鐘

STORE_IDS = [f"S{str(i).zfill(3)}" for i in range(1, 51)]
STORE_LIST = [{"id": sid, "name": f"店{sid[1:]}"} for sid in STORE_IDS]

TYPE_LABEL = {
    "freezer": "冷凍",
    "refrigerator": "冷藏",
    "open_show_case": "開放櫃",
    "hvac": "空調",
}

# ========== 工具：以「分鐘」為單位產生可重現亂數 ==========
def minute_epoch() -> int:
    """回傳整數分鐘（UNIX 秒 // 60）"""
    return int(datetime.now(timezone.utc).timestamp() // 60)

def seeded_rng(*salts) -> Random:
    """以全域 SEED + salts 建一個 Random，確保同一分鐘結果一致、跨分鐘改變"""
    s = SEED
    for x in salts:
        s = (s * 1315423911) ^ hash(x)
    return Random(s & 0xFFFFFFFFFFFF)

# ========== 除霜狀態計算 ==========
def get_defrost_status(store_id: str, device_id: str, dev_type: str, current_minute: int) -> int:
    """
    計算設備在當前分鐘是否處於除霜狀態
    
    邏輯：
    1. 每個設備有自己的除霜週期（4-6小時隨機）
    2. 週期內前20分鐘為除霜狀態
    3. 只有冷凍和冷藏設備會除霜
    """
    if dev_type not in ["freezer", "refrigerator"]:
        return 0
    
    # 使用設備ID作為種子，確保每個設備有固定的除霜模式
    rng = seeded_rng("defrost_cycle", store_id, device_id)
    
    # 為每個設備分配一個固定的除霜週期（4-6小時）
    device_cycle = rng.randint(DEFROST_CYCLE_MIN, DEFROST_CYCLE_MAX)
    
    # 計算設備的除霜起始偏移（避免所有設備同時除霜）
    start_offset = rng.randint(0, device_cycle - 1)
    
    # 計算當前在週期中的位置
    cycle_position = (current_minute + start_offset) % device_cycle
    
    # 前20分鐘為除霜狀態
    return 1 if cycle_position < DEFROST_DURATION else 0

# ========== 每分鐘：決定「哪些店沒資料」 ==========
def store_has_data(store_id: str, minute_key: int) -> bool:
    rng = seeded_rng("no_data", minute_key, store_id)
    return rng.random() >= NO_DATA_RATE  # True -> 有資料

# ========== 店鋪設備配置快取 ==========
STORE_CONFIG_CACHE = {}


# 修正後的設備分配邏輯
def get_store_device_config(store_id: str) -> Dict:
    if store_id in STORE_CONFIG_CACHE:
        return STORE_CONFIG_CACHE[store_id]
    
    config_rng = seeded_rng("store_config", store_id)
    
    # 櫃類總數 4~12 台，分配到三種 type，每種至少1台
    total_cabinets = config_rng.randint(4, 12)
    
    # 先給每種櫃類分配1台（確保至少有1台）
    n_freezer = 1
    n_refrigerator = 1  
    n_open_show_case = 1
    
    # 剩餘的櫃類數量隨機分配
    remaining = total_cabinets - 3  # 已分配3台，剩餘的數量
    
    if remaining > 0:
        # 將剩餘數量隨機分配給三種櫃類
        for _ in range(remaining):
            choice = config_rng.randint(0, 2)  # 0=freezer, 1=refrigerator, 2=open_show_case
            if choice == 0:
                n_freezer += 1
            elif choice == 1:
                n_refrigerator += 1
            else:
                n_open_show_case += 1
    
    # hvac 0~2 台
    n_hvac = config_rng.randint(0, 2)
    
    config = {
        "freezer_count": n_freezer,
        "refrigerator_count": n_refrigerator,
        "open_show_case_count": n_open_show_case,
        "hvac_count": n_hvac
    }
    
    # 快取配置
    STORE_CONFIG_CACHE[store_id] = config
    return config

# ========== 每店：產生該店固定設備清單 ==========
def generate_devices_for_store(store_id: str, minute_key: int) -> List[Dict]:
    """
    基於固定的店鋪配置生成設備清單
    設備配置不會因時間改變，但可用性仍受 minute_key 影響
    """
    config = get_store_device_config(store_id)
    
    def make_ids(prefix: str, n: int) -> List[str]:
        return [f"{prefix}{str(i).zfill(3)}" for i in range(1, n + 1)]

    devs = []
    devs += [{"id": did, "type": "freezer"} for did in make_ids("F", config["freezer_count"])]
    devs += [{"id": did, "type": "refrigerator"} for did in make_ids("R", config["refrigerator_count"])]
    devs += [{"id": did, "type": "open_show_case"} for did in make_ids("O", config["open_show_case_count"])]
    devs += [{"id": did, "type": "hvac"} for did in make_ids("H", config["hvac_count"])]

    # 使用固定種子打散順序（保持每家店的設備順序一致）
    config_rng = seeded_rng("store_order", store_id)
    config_rng.shuffle(devs)
    return devs

# ========== 單台設備：產生本分鐘即時資料 or 204 ==========
def device_realtime(store_id: str, device_id: str, dev_type: str, minute_key: int) -> Tuple[int, Dict]:
    """
    依 type 生成中文欄位格式；MISS_RATE 機率回 204（缺測）
    回傳 (status_code, json_body)
    """
    rng = seeded_rng("rt", minute_key, store_id, device_id)
    if rng.random() < MISS_RATE:
        return 204, {}

    # 使用 timezone.utc → 轉 +08:00（不依賴系統時區）
    utc_minute = datetime.fromtimestamp(minute_key * 60, tz=timezone.utc)
    ts_utc = utc_minute.replace(second=0, microsecond=0)
    ts_tpe = ts_utc + timedelta(hours=8)
    ts_iso = ts_tpe.isoformat()

    if dev_type == "hvac":
        # 空調：T_room 18 ~ 30 ℃
        val = round(rng.uniform(18.0, 30.0), 1)
        body = {
            "timestamp": ts_iso,
            "Device_ID": device_id,
            "Device_type": TYPE_LABEL["hvac"],
            "T_room": val,
        }
        return 200, body

    # 櫃類：溫度範圍依類型、冷凍/冷藏含 Defrost_status
    if dev_type == "freezer":
        temp = round(rng.uniform(-25.0, -15.0), 1)
        defrost = get_defrost_status(store_id, device_id, dev_type, minute_key)
        dtype = TYPE_LABEL["freezer"]
    elif dev_type == "refrigerator":
        temp = round(rng.uniform(0.0, 8.0), 1)
        defrost = get_defrost_status(store_id, device_id, dev_type, minute_key)
        dtype = TYPE_LABEL["refrigerator"]
    elif dev_type == "open_show_case":
        temp = round(rng.uniform(2.0, 12.0), 1)
        defrost = 0  # 開放櫃不除霜
        dtype = TYPE_LABEL["open_show_case"]
    else:
        # 不認得的 type（理論不會發生）
        return 404, {}

    body = {
        "timestamp": ts_iso,
        "Device_ID": device_id,
        "Device_type": dtype,
        "Temp_current": temp,
        "Defrost_status": defrost,
    }
    return 200, body

# ========== FastAPI 應用 ==========
app = FastAPI(title="Mock Company API", version="0.7.0-mock")

@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Mock Company API (3 GET) is running.",
        "endpoints": [
            "GET /stores",
            "GET /stores/{store_id}/devices",
            "GET /stores/{store_id}/devices/{device_id}/realtime",
            "POST /output/priority.json"
        ],
        "seed": SEED,
        "no_data_rate": NO_DATA_RATE,
        "miss_rate": MISS_RATE,
        "device_config": {
            "stable_per_store": True,
            "total_stores": len(STORE_IDS),
            "cabinets_per_store": "4-12",
            "hvac_per_store": "0-2",
            "device_types": {
                "freezer": {"prefix": "F", "temp_range": "-25~-15°C", "defrost": True},
                "refrigerator": {"prefix": "R", "temp_range": "0~8°C", "defrost": True},
                "open_show_case": {"prefix": "O", "temp_range": "2~12°C", "defrost": True},
                "hvac": {"prefix": "H", "temp_range": "18~30°C", "defrost": False}
            }
        },
        "defrost_info": {
            "cycle_hours": f"{DEFROST_CYCLE_MIN//60}-{DEFROST_CYCLE_MAX//60}",
            "duration_minutes": DEFROST_DURATION,
            "applies_to": ["freezer", "refrigerator"]
        }
    }

@app.get("/stores")
def list_stores():
    # 固定 50 間
    return STORE_LIST

@app.get("/stores/{store_id}/devices")
def list_devices_of_store(store_id: str):
    # store_id 驗證
    if store_id not in STORE_IDS:
        raise HTTPException(status_code=404, detail="store not found")

    mkey = minute_epoch()

    # 本輪「沒資料」
    if not store_has_data(store_id, mkey):
        return {"devices": []}

    devs = generate_devices_for_store(store_id, mkey)
    return {"devices": devs}

@app.get("/stores/{store_id}/devices/{device_id}/realtime")
def realtime_of_device(store_id: str, device_id: str):
    # store_id 驗證
    if store_id not in STORE_IDS:
        raise HTTPException(status_code=404, detail="store not found")

    mkey = minute_epoch()

    # 本輪店沒資料 → 204
    if not store_has_data(store_id, mkey):
        return Response(status_code=204)

    # 取得該店本輪設備清單，確認 device 是否存在於該店（避免混淆）
    devs = generate_devices_for_store(store_id, mkey)
    dev_map = {d["id"]: d["type"] for d in devs}
    dev_type = dev_map.get(device_id)
    if not dev_type:
        # 裝置不屬於該店（或編號不存在）→ 404
        raise HTTPException(status_code=404, detail="device not found in this store")

    status, body = device_realtime(store_id, device_id, dev_type, mkey)
    if status == 204:
        return Response(status_code=204)
    return body

@app.post("/output/priority.json")
def post_priority(payload: dict):
    """
    接收優先順序；不做驗證，單純回 200 並回覆統計
    期望格式：{"Store_ID":"S012","devices":[...]}
    """
    store_id = payload.get("Store_ID") or payload.get("store_id")
    devices = payload.get("devices") or []
    count = len(devices) if isinstance(devices, list) else 0

    # 可選：標記未知的 Store_ID（不擋）
    note = None
    if store_id and store_id not in STORE_IDS:
        note = "unknown_store_id"

    return {
        "ok": True,
        "received_store": store_id,
        "device_count": count,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }