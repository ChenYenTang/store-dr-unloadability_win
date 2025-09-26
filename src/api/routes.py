from fastapi import APIRouter

router = APIRouter()

# 健康檢查 (測試用)
@router.get("/health")
async def health():
    return {"ok": True}

# 假資料 (後面可以換成真實 /api/devices/feed)
@router.get("/devices/feed")
async def devices_feed():
    return {
        "devices": [
            {
                "Device_ID": "F001",
                "Device_type": "freezer",
                "Device_size": "大",
                "Risk_penalty": "高",
                "Shared_compressor_flag": 0,
                "timestamp": "2025-08-19T12:00:00",
                "Business_hours_flag": 1,
                "Temp_current": -18,
                "Power": 1.2,
                "T_room": 27.5,
                "Defrost_status": 0,
                "Time_since_defrost": 120,
                "Unloadable_time_min": 15,
                "Unloadable_energy_kWh": 0.8,
                "Priority_score": 0.75,
                "Eligibility": "eligible",
            },
            {
                "Device_ID": "C002",
                "Device_type": "fridge",
                "Device_size": "中",
                "Risk_penalty": "中",
                "Shared_compressor_flag": 1,
                "timestamp": "2025-08-19T12:00:00",
                "Business_hours_flag": 1,
                "Temp_current": 5,
                "Power": 0.6,
                "T_room": 27.5,
                "Defrost_status": 0,
                "Time_since_defrost": 80,
                "Unloadable_time_min": 10,
                "Unloadable_energy_kWh": 0.3,
                "Priority_score": 0.55,
                "Eligibility": "eligible",
            },
        ]
    }
