import os
import json
import re
import gradio as gr
import pandas as pd
import yaml
from src.io.company_api import post_priority
from datetime import datetime
import threading
import time

# === Auto Push：避免重入（新增） ===
import threading
_AUTO_PUSH_LOCK = threading.Lock()

# 檔案路徑
CONFIG_PATH = "input/config.json"
PRIORITY_PATH = "output/priority.json"
HISTORY_PATH = "output/push_history.json"

# 全域狀態
_priority_data = None
_auto_push_state = {"enabled": False, "frequency": 15, "last_push": 0}
_push_lock = threading.Lock()

def _pred_read_selected():
    """讀取 config.json -> Selected_Store_IDs list"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            sel = cfg.get("Selected_Store_IDs") or []
            if isinstance(sel, list) and sel:
                return sel
            # 相容舊欄位
            for k in ("All_Store_IDs", "Store_IDs"):
                v = cfg.get(k) or []
                if isinstance(v, list) and v:
                    return v
            sid = cfg.get("Store_ID")
            return [sid] if sid else []
    except Exception:
        pass
    return []

def _load_api_params_from_yaml():
    """從 config/config.yaml 讀取 api 區塊"""
    p = os.path.join("config", "config.yaml")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        
        api = cfg.get("api", {})
        mapping = cfg.get("mapping", {})
        
        # 處理 base_url 格式
        base = api.get("base_url")
        if base and not re.match(r"^https?://", base, flags=re.IGNORECASE):
            base = "http://" + base
        
        return {
            "base_url": base.rstrip("/") if base else None,
            "auth_type": api.get("auth", {}).get("type", "api_key"),
            "timeout_sec": api.get("timeout_sec", 10),
            "verify_ssl": api.get("verify_ssl", True),
            "endpoints": api.get("endpoints", {}),
            "api_key_header": api.get("auth", {}).get("api_key_header", "X-API-Key"),
            "mapping": mapping,
        }
    except Exception as e:
        print(f"載入 config.yaml 失敗: {e}")
        return {}

def _generate_status_cards():
    """生成狀態卡片的HTML"""
    selected_stores = _pred_read_selected()
    
    # 預測狀態卡片
    pred_completed = 0
    pred_total = len(selected_stores)
    
    for store_id in selected_stores:
        pred_dir = os.path.join("weighted_models", store_id)
        if os.path.exists(pred_dir):
            pred_files = [f for f in os.listdir(pred_dir) if f.startswith("pred_")]
            if pred_files:
                pred_completed += 1
    
    pred_card = f"""
    <div style='padding:15px;background:#f0f8ff;border-radius:8px;border-left:4px solid #2196F3;'>
        <h4 style='margin:0 0 10px 0;color:#1976D2;'>預測狀態</h4>
        <p style='margin:5px 0;font-size:24px;font-weight:bold;color:#1976D2;'>{pred_completed}/{pred_total}</p>
        <p style='margin:0;color:#666;'>門市預測完成</p>
    </div>
    """
    
    # 風險摘要卡片
    global _priority_data
    device_count = 0
    high_risk_count = 0
    
    if _priority_data and "devices" in _priority_data:
        devices = _priority_data["devices"]
        device_count = len(devices)
        high_risk_count = len([d for d in devices if d.get("Priority_score", 0) > 2.0])
    elif _priority_data and "stores" in _priority_data:
        for store_devices in _priority_data["stores"].values():
            device_count += len(store_devices)
            high_risk_count += len([d for d in store_devices if d.get("Priority_score", 0) > 2.0])
    
    risk_card = f"""
    <div style='padding:15px;background:#fff8f0;border-radius:8px;border-left:4px solid #FF9800;'>
        <h4 style='margin:0 0 10px 0;color:#F57C00;'>風險設備</h4>
        <p style='margin:5px 0;font-size:24px;font-weight:bold;color:#F57C00;'>{high_risk_count}/{device_count}</p>
        <p style='margin:0;color:#666;'>高風險設備</p>
    </div>
    """
    
    # API狀態卡片
    last_push_time = "未推送"
    push_status = "待命"
    
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                last_record = history[0]
                last_push_time = last_record["timestamp"][:16].replace("T", " ")
                push_status = "成功" if last_record["success"] else "失敗"
        except Exception:
            pass
    
    api_card = f"""
    <div style='padding:15px;background:#f0fff0;border-radius:8px;border-left:4px solid #4CAF50;'>
        <h4 style='margin:0 0 10px 0;color:#388E3C;'>API推送</h4>
        <p style='margin:5px 0;font-size:16px;font-weight:bold;color:#388E3C;'>{push_status}</p>
        <p style='margin:0;color:#666;'>{last_push_time}</p>
    </div>
    """
    
    return pred_card, risk_card, api_card

def _infer_device_type(device_id: str) -> str:
    """從設備ID推測設備類型"""
    device_id = str(device_id).upper()
    if device_id.startswith('F'):
        return '冷凍'
    elif device_id.startswith('R'):
        return '冷藏' 
    elif device_id.startswith('O'):
        return '開放櫃'
    else:
        return '未知'

def _calculate_priority_ranking(df: pd.DataFrame, freezer_weight=1.5, fridge_weight=1.0, buffer_min=5):
    """計算設備卸載優先順序"""
    if df.empty:
        return df, "無數據", None
    
    results = []
    
    for _, row in df.iterrows():
        device_type = _infer_device_type(row['device_id'])
        
        # 基礎風險評分 = 1 / TTW_p10 (時間越短風險越高)
        ttw_p10 = max(row.get('TTW_p10', 60), 1)  # 避免除零
        base_risk = 100 / ttw_p10
        
        # 設備類型權重
        type_weight = freezer_weight if '冷凍' in device_type else fridge_weight
        risk_score = base_risk * type_weight
        
        # 可卸載時間 = TTW_p10 - 安全緩衝
        unloadable_time = max(ttw_p10 - buffer_min, 0)
        
        results.append({
            'Store_ID': row['Store_ID'],
            'Device_ID': row['device_id'],
            'Device_Type': device_type,
            'Risk_Score': round(risk_score, 2),
            'Unloadable_Time_Min': int(unloadable_time),
            'TTW_P10': int(ttw_p10),
            'TTW_P50': int(row.get('TTW_p50', ttw_p10)),
            'TTW_P90': int(row.get('TTW_p90', ttw_p10)),
            'Quality': row.get('quality', 'unknown')
        })
    
    # 按風險評分排序
    results.sort(key=lambda x: x['Risk_Score'], reverse=True)
    
    # 添加排名
    for i, item in enumerate(results):
        item['Ranking'] = i + 1
    
    df_result = pd.DataFrame(results)
    
    # 生成 priority.json
    priority_data = _format_priority_output(results)
    
    return df_result, f"已生成 {len(results)} 個設備的優先順序", priority_data

def _format_priority_output(results: pd.DataFrame) -> dict:
    """
    只用「真實功率」估算可卸載能量：
    Unloadable_energy_kWh = (Unloadable_Time_Min / 60) * power
    - 若 power 缺，當 0 處理。
    - Time_since_defrost 直接帶用資料列中的 ingest_ts（或 0）。
    """
    if results.empty:
        return {}

    stores: dict[str, list] = {}
    for _, r in results.iterrows():
        sid = str(r.get("Store_ID") or r.get("store_id") or "")
        if not sid:
            # 沒門市就略過
            continue
        stores.setdefault(sid, [])

        unload_min = int(r.get("Unloadable_Time_Min") or r.get("Unloadable_time_min") or 0)
        # 關鍵：改成用真實功率（kW）
        power_kw = float(pd.to_numeric(pd.Series([r.get("power")])).fillna(0).iloc[0])
        energy_kwh = round(unload_min / 60.0 * power_kw, 3)

        stores[sid].append({
            "Device_ID": str(r.get("Device_ID") or r.get("device_id") or ""),
            "Ranking": int(r.get("Ranking") or 0),
            "Priority_score": float(pd.to_numeric(pd.Series([r.get("Risk_Score")])).fillna(0).iloc[0]),
            "Unloadable_time_min": unload_min,
            "Unloadable_energy_kWh": energy_kwh,
            "Time_since_defrost": float(pd.to_numeric(pd.Series([r.get("Time_since_defrost") or r.get("ingest_ts")])).fillna(0).iloc[0]),
            "Exclude_flag": 1 if str(r.get("Quality","")).lower() == "low" else 0,
        })

    # 保持與單店/多店相容的輸出格式
    if len(stores) == 1:
        sid = next(iter(stores.keys()))
        return {"Store_ID": sid, "devices": stores[sid]}
    return {"timestamp": datetime.now().isoformat(), "stores": stores}


def _generate_risk_priority(freezer_weight, fridge_weight, buffer_min):
    """整合預測結果，計算風險評分並生成優先順序"""
    selected_stores = _pred_read_selected()
    if not selected_stores:
        return pd.DataFrame(), "無選定門市", None
    
    all_predictions = []
    
    # 收集各門市的預測結果
    for store_id in selected_stores:
        try:
            # 1. 讀取最新預測結果
            pred_dir = os.path.join("weighted_models", store_id)
            if not os.path.exists(pred_dir):
                continue
                
            # 找最新的預測檔案（按修改時間）
            pred_files = [f for f in os.listdir(pred_dir) if f.startswith("pred_")]
            if not pred_files:
                continue
                
            latest_pred = max(pred_files, key=lambda x: os.path.getmtime(os.path.join(pred_dir, x)))
            pred_path = os.path.join(pred_dir, latest_pred)
            
            # 讀取預測數據
            df_pred = pd.read_csv(pred_path)
            df_pred['Store_ID'] = store_id
            
            # 2. 讀取實時數據來補充真實資訊
            latest_path = os.path.join("input", "latest", f"{store_id}.json")
            real_device_info = {}
            if os.path.exists(latest_path):
                try:
                    with open(latest_path, "r", encoding="utf-8") as f:
                        latest_data = json.load(f)
                    
                    for device in latest_data.get("devices", []):
                        device_id = device.get("Device_ID")
                        if device_id:
                            real_device_info[device_id] = {
                                "device_type": device.get("Device_type", "未知"),
                                "temp_current": device.get("Temp_current"),
                                "power": device.get("Power", 0),
                                "defrost_status": device.get("Defrost_status", 0),
                                "timestamp": device.get("timestamp")
                            }
                except Exception as e:
                    print(f"讀取 {store_id} 實時數據失敗: {e}")
            
            # 3. 合併預測和實時數據
            for idx, row in df_pred.iterrows():
                device_id = row['device_id']
                real_info = real_device_info.get(device_id, {})
                
                # 使用真實設備類型，若無則推測
                df_pred.at[idx, 'real_device_type'] = real_info.get("device_type", _infer_device_type(device_id))
                df_pred.at[idx, 'real_power'] = real_info.get("power", 0)
                df_pred.at[idx, 'defrost_status'] = real_info.get("defrost_status", 0)
                df_pred.at[idx, 'temp_current'] = real_info.get("temp_current")
                df_pred.at[idx, 'latest_timestamp'] = real_info.get("timestamp")
            
            all_predictions.append(df_pred)
            
        except Exception as e:
            print(f"讀取 {store_id} 預測結果失敗: {e}")
            continue
    
    if not all_predictions:
        return pd.DataFrame(), "無可用預測結果", None
    
    # 合併所有預測結果
    df_all = pd.concat(all_predictions, ignore_index=True)
    
    # 計算風險評分和排序（使用真實數據）
    return _calculate_priority_ranking_with_real_data(df_all, freezer_weight, fridge_weight, buffer_min)

def _calculate_priority_ranking_with_real_data(
    df: pd.DataFrame, freezer_weight=1.5, fridge_weight=1.0, buffer_min=5
):
    """使用真實數據計算設備卸載優先順序"""
    if df is None or df.empty:
        return pd.DataFrame(), "無數據", None

    rows = []

    for _, row in df.iterrows():
        # 1) 設備類型（若無則用推斷）
        device_type = row.get("real_device_type", _infer_device_type(row["device_id"]))

        # 2) 風險基礎：TTW_p10 越短風險越高
        ttw_p10 = max(int(row.get("TTW_p10", 60)), 1)
        base_risk = 100.0 / ttw_p10

        # 3) 類型權重（你也可改成更穩健的 freezer 判斷邏輯）
        type_weight = freezer_weight if ("冷凍" in str(device_type)) else fridge_weight
        risk_score = base_risk * type_weight

        # 4) 可卸載時間（分鐘）
        unloadable_time = max(ttw_p10 - int(buffer_min), 0)

        # 5) 真實功率（kW）→ 後續輸出函式會用 'power' 來算能量
        real_power = float(row.get("real_power", 0.0))

        # 6) 排除條件
        exclude_flag = 0
        exclude_reasons = []
        if int(row.get("defrost_status", 0)) == 1:
            exclude_flag = 1
            exclude_reasons.append("除霜中")
        if str(row.get("quality", "ok")).lower() == "low":
            exclude_flag = 1
            exclude_reasons.append("預測品質低")
        if unloadable_time < 5:
            exclude_flag = 1
            exclude_reasons.append("可用時間太短")

        # 7) 可選：若有 ingest_ts/Time_since_defrost 就帶上（沒有就 0）
        time_since_defrost = float(
            row.get("Time_since_defrost", row.get("ingest_ts", 0)) or 0
        )

        rows.append({
            "Store_ID": row["Store_ID"],
            "Device_ID": row["device_id"],
            "Device_Type": device_type,
            "Risk_Score": round(float(risk_score), 2),
            "Unloadable_Time_Min": int(unloadable_time),
            "TTW_P10": int(ttw_p10),
            "TTW_P50": int(row.get("TTW_p50", ttw_p10)),
            "TTW_P90": int(row.get("TTW_p90", ttw_p10)),
            "Quality": row.get("quality", "unknown"),
            "Current_Temp": row.get("temp_current"),
            # 關鍵：輸出函式會用 'power' 這個鍵名來換算能量
            "power": real_power,
            # 可選：給輸出函式直接用
            "Time_since_defrost": time_since_defrost,
            # 保留你原先算好的能耗（但輸出會以 power×時間重算）
            "Unloadable_Energy_kWh": round(unloadable_time * real_power / 60.0, 3),
            "Exclude_Flag": int(exclude_flag),
            "Exclude_Reasons": ",".join(exclude_reasons),
        })

    # list → DataFrame
    df_result = pd.DataFrame(rows)
    if df_result.empty:
        return df_result, "無數據", None

    # 排序：先未排除、風險高在前
    df_result = df_result.sort_values(
        ["Exclude_Flag", "Risk_Score"], ascending=[True, False]
    ).reset_index(drop=True)

    # 排名：只給未排除者編號，其餘標 999
    mask = df_result["Exclude_Flag"] == 0
    df_result.loc[mask, "Ranking"] = range(1, mask.sum() + 1)
    df_result.loc[~mask, "Ranking"] = 999
    df_result["Ranking"] = df_result["Ranking"].astype(int)

    # 生成 priority.json 的資料（改用新版名稱）
    priority_data = _format_priority_output(df_result)

    excluded_count = int((df_result["Exclude_Flag"] == 1).sum())
    status = f"已生成 {len(df_result)} 個設備的優先順序，其中 {excluded_count} 個被排除"

    # 回傳給表格使用的欄位（可依你 UI 需求調整）
    view_cols = [
        "Store_ID", "Device_ID", "Device_Type",
        "Risk_Score", "Unloadable_Time_Min", "TTW_P10", "Ranking",
    ]
    view_cols = [c for c in view_cols if c in df_result.columns]
    return df_result[view_cols], status, priority_data


def _push_to_api(priority_data: dict) -> tuple[bool, str]:
    """推送優先順序數據到公司API"""
    try:
        # 讀取API配置
        params = _load_api_params_from_yaml()
        if not params.get("base_url"):
            return False, "❌ API配置未設定 (請到「基本資料與閾值設定」分頁配置API)"
        
        # 檢查必要的端點設定
        endpoints = params.get("endpoints", {})
        if not endpoints.get("post_priority"):
            return False, "❌ 缺少 POST 端點設定"
        
        # 執行推送
        success, code, message = post_priority(params, priority_data)
        
        # 記錄推送歷史
        _log_push_history(success, code, message, priority_data)
        
        status_icon = "✅" if success else "❌"
        return success, f"{status_icon} 推送{'成功' if success else '失敗'}: HTTP {code} - {message}"
        
    except Exception as e:
        _log_push_history(False, 0, str(e), priority_data)
        return False, f"❌ 推送異常: {e}"

def _log_push_history(success: bool, code: int, message: str, data: dict):
    """記錄推送歷史"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "response_code": code,
        "message": message,
        "store_count": len(data.get("stores", [data.get("Store_ID", "")])),
        "device_count": sum(len(devices) for devices in 
                          (data.get("stores", {}).values() if "stores" in data 
                           else [data.get("devices", [])]))
    }
    
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        
        history.insert(0, record)  # 最新記錄在前
        history = history[:100]    # 保留最近100筆
        
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"記錄推送歷史失敗: {e}")

def _load_push_history():
    """載入推送歷史記錄"""
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            # 格式化為表格數據
            table_data = []
            for record in history[:20]:  # 顯示最近20筆
                table_data.append([
                    record["timestamp"][:16].replace("T", " "),
                    record["store_count"],
                    record["device_count"],
                    "成功" if record["success"] else "失敗",
                    record["response_code"]
                ])
            
            return pd.DataFrame(table_data, columns=["時間", "門市數", "設備數", "狀態", "回應代碼"])
    except Exception:
        pass
    
    return pd.DataFrame(columns=["時間", "門市數", "設備數", "狀態", "回應代碼"])

def render():
    with gr.Tab("設備卸載排序與API推送"):
        #gr.Markdown("### 設備卸載排序與API推送")

        # ========= API 設定狀態檢查 =========
        def _check_api_config():
            params = _load_api_params_from_yaml()
            if not params.get("base_url"):
                return """
                <div style='padding:10px;background:#fff3cd;border:1px solid #ffeaa7;border-radius:5px;color:#856404;'>
                    ⚠️ <strong>API未設定</strong>: 請先到「基本資料與閾值設定」分頁配置公司API連線
                </div>
                """
            return """
            <div style='padding:10px;background:#d1edff;border:1px solid #74b9ff;border-radius:5px;color:#2d3436;'>
                ✅ <strong>API已設定</strong>: {}</div>
            """.format(params["base_url"])
        
        api_config_status = gr.HTML(_check_api_config())        

        # ========= 快速狀態儀表板 =========
        with gr.Accordion("系統狀態概覽", open=True):
            with gr.Row():
                prediction_status_card = gr.HTML()
                risk_summary_card = gr.HTML()
                api_status_card = gr.HTML()
        
        # ========= 風險分析與排序 =========
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 風險評分設定")
                risk_weight_freezer = gr.Slider(
                    label="冷凍設備風險權重", minimum=0.5, maximum=2.0, 
                    value=1.5, step=0.1, info="冷凍設備相對風險倍數"
                )
                risk_weight_fridge = gr.Slider(
                    label="冷藏設備風險權重", minimum=0.5, maximum=2.0, 
                    value=1.0, step=0.1, info="冷藏設備相對風險倍數"
                )
                safety_buffer_min = gr.Number(
                    label="安全緩衝時間（分鐘）", value=5, precision=0,
                    info="從預測時間中扣除的安全餘量"
                )
                
            with gr.Column():
                generate_priority_btn = gr.Button("生成優先順序", variant="primary")
                priority_status = gr.Textbox(label="生成狀態", interactive=False)
                
        # ========= 優先順序結果 =========
        priority_table = gr.Dataframe(
            headers=["Store_ID", "Device_ID", "Device_Type", "Risk_Score", "Unloadable_Time_Min", "TTW_P10", "Ranking"],
            label="設備卸載優先順序",
            interactive=False
        )
        
        # ========= API 推送設定 =========
        with gr.Accordion("API 推送設定", open=True):
            with gr.Row():
                api_frequency = gr.Number(
                    label="推送頻率（分鐘）", value=15, precision=0, minimum=1,
                    info="不可超過預測範圍(horizon)時間"
                )
                auto_push_enabled = gr.Checkbox(
                    label="自動推送", value=False,
                    info="啟用後將按設定頻率自動推送"
                )
                manual_push_btn = gr.Button("立即推送", variant="secondary")
            
            api_push_status = gr.Textbox(label="推送狀態", interactive=False)
        
        # ========= 推送記錄 =========
        with gr.Accordion("推送記錄", open=False):
            push_history_table = gr.Dataframe(
                headers=["時間", "門市數", "設備數", "狀態", "回應代碼"],
                label="API推送歷史",
                interactive=False,
                value=_load_push_history()
            )
            
        # ========= 定時器與狀態更新 =========
        dashboard_timer = gr.Timer(30, active=True)  # 30秒更新儀表板
        auto_push_timer = gr.Timer(60, active=True)   # 1分鐘檢查自動推送
        
        # ========= 事件綁定 =========
        
        # 生成優先順序
        def on_generate_priority(freezer_w, fridge_w, buffer):
            global _priority_data
            df, status, data = _generate_risk_priority(freezer_w, fridge_w, int(buffer))
            _priority_data = data
            
            # 保存到檔案
            if data:
                os.makedirs("output", exist_ok=True)
                with open(PRIORITY_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            return df, status
        
        generate_priority_btn.click(
            on_generate_priority,
            inputs=[risk_weight_freezer, risk_weight_fridge, safety_buffer_min],
            outputs=[priority_table, priority_status]
        )
        
        # 立即推送
        def on_manual_push(freezer_w, fridge_w, buffer):
            global _priority_data
            # 推送前先重算
            with _push_lock:
                df, status, data = _generate_risk_priority(freezer_w, fridge_w, int(buffer))
                _priority_data = data

                # 也順手落地 priority.json，和「生成優先順序」一致
                if data:
                    os.makedirs("output", exist_ok=True)
                    with open(PRIORITY_PATH, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    return f"❗重算失敗（{status}），未推送。"

                success, message = _push_to_api(_priority_data)
                return message

        # 把風險設定三個元件的值當作 inputs 傳進來
        manual_push_btn.click(
            on_manual_push,
            inputs=[risk_weight_freezer, risk_weight_fridge, safety_buffer_min],
            outputs=[api_push_status]
        )
        
        # 自動推送控制
        def on_auto_push_change(enabled, frequency):
            global _auto_push_state
            _auto_push_state["enabled"] = enabled
            _auto_push_state["frequency"] = max(1, int(frequency))
            _auto_push_state["last_push"] = time.time() if not enabled else _auto_push_state["last_push"]
            
            status = f"自動推送: {'開啟' if enabled else '關閉'}"
            if enabled:
                status += f" (每 {frequency} 分鐘)"
            return status
        
        auto_push_enabled.change(
            on_auto_push_change,
            inputs=[auto_push_enabled, api_frequency],
            outputs=[api_push_status]
        )
        api_frequency.change(
            on_auto_push_change,
            inputs=[auto_push_enabled, api_frequency],
            outputs=[api_push_status]
        )
        
        # 定時更新儀表板
        def update_dashboard():
            return _generate_status_cards()
        
        dashboard_timer.tick(update_dashboard, outputs=[prediction_status_card, risk_summary_card, api_status_card])
        
        # === Auto Push：推送前先重算 ===
        def check_auto_push(freezer_w, fridge_w, buffer):
            """
            自動推送每次觸發：
            - 若未啟用或尚未到頻率，什麼都不做
            - 到頻率時：先重算最新優先順序 → 更新快取/檔案 → 推送 → 更新歷史
            """
            global _auto_push_state, _priority_data

            if not _auto_push_state["enabled"]:
                return gr.update(), gr.update()

            now = time.time()
            elapsed = (now - _auto_push_state["last_push"]) / 60  # 分鐘
            if elapsed < _auto_push_state["frequency"]:
                return gr.update(), gr.update()

            with _push_lock:
                # 1) 重算
                df, status, data = _generate_risk_priority(freezer_w, fridge_w, int(buffer))
                _priority_data = data

                if not data:
                    # 不推送，但回報狀態
                    return gr.update(value=f"自動推送: ❗重算失敗（{status}）"), _load_push_history()

                # 落地 priority.json（保持與手動一致）
                try:
                    os.makedirs("output", exist_ok=True)
                    with open(PRIORITY_PATH, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    # 落地失敗不阻斷推送
                    print(f"寫入 {PRIORITY_PATH} 失敗：{e}")

                # 2) 推送
                success, message = _push_to_api(_priority_data)
                _auto_push_state["last_push"] = now

                # 3) 回寫 UI（狀態 + 歷史表）
                return gr.update(value=f"自動推送: {message}"), _load_push_history()
        
        auto_push_timer.tick(
            check_auto_push,
            inputs=[risk_weight_freezer, risk_weight_fridge, safety_buffer_min],
            outputs=[api_push_status, push_history_table]
        )        
        # 初始化儀表板
        def init_dashboard():
            return _generate_status_cards()
        
        # 把「冷凍權重、冷藏權重、緩衝分鐘」也一併傳進去
        auto_push_timer.tick(
            check_auto_push,
            inputs=[risk_weight_freezer, risk_weight_fridge, safety_buffer_min],
            outputs=[api_push_status, push_history_table]
        )

    return