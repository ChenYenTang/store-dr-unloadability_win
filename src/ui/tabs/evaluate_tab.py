import gradio as gr
import os, json, threading, time, re
import pandas as pd
from datetime import datetime, timezone

# === 路徑常數（可用環境變數覆寫） ===
WEIGHT_DIR = os.getenv("WEIGHT_DIR", "weighted_models")
INDEX_PATH = os.path.join(WEIGHT_DIR, "index.json")

from src.AI_models.lstm_trend import predict_store as predict  # 匯入 LSTM 模型邏輯
from .train_tab import render_panel as render_train_panel  # 嵌入新版訓練面板
from .predict_tab import render_panel as render_predict_panel  # 新版「預測與分析」面板


HISTORY_DIR = "input/history"
HISTORY_PATH = "input/history/history.csv"

_t: "threading.Thread | None" = None  # 全域參考

def start_append(freq_min: float = 1):
    global running, _t
    if not running:
        running = True
        _t = threading.Thread(target=_auto_append_loop, args=(max(1, int(freq_min)),))
        _t.start()
    return "<span style='color:red;'>多店定時寫入已啟動（去重）</span>"

def stop_append():
    global running, _t
    running = False
    if _t and _t.is_alive():
        _t.join(timeout=5)
    return "已停止（背景循環會在當次寫入後結束）"

def _history_day_path() -> str:
    """回傳當日歷史檔：input/history/YYYYMMDD.csv"""
    day = datetime.now().strftime("%Y%m%d")
    return os.path.join(HISTORY_DIR, f"{day}.csv")

# 分店快照：依 Store_ID 讀 input/latest/<Store_ID>.json
CONFIG_JSON = os.path.join("input", "config.json")
LATEST_DIR = os.path.join("input", "latest")

# 固定欄位（僅供 UI 顯示需要）
COLUMNS = ["Time_stamp","Store_ID","Device_ID","Device_type","Temp_current","T_room","Defrost_status","Power","Time_since_defrost"]
HISTORY_COLUMNS = ["ts","store_id","device_id","device_type","temp_current","t_room","defrost_status","power","ingest_ts"]

running = False   # 背景資料追加控制
interval = 180    # 預設 3 分鐘（秒）
_last_status = "尚未開始"  # 顯示背景循環狀態

# === 功率設定全域變數 ===
POWER_SETTINGS = {
    "冷藏": 4.5,
    "冷凍": 5.0,
    "開放櫃": 5.7,
    "空調": 5.0
}

# ---- 自動啟動設定（像 CLI 常駐）----
# EVAL_AUTO_APPEND=1 表示啟動時自動開始追加；0 則不自動。
# EVAL_AUTO_FREQ_MIN 控制頻率（分鐘），預設 1。
_AUTO_ENABLE = os.environ.get("EVAL_AUTO_APPEND", "1") == "1"
try:
    _AUTO_FREQ_MIN = max(1, int(os.environ.get("EVAL_AUTO_FREQ_MIN", "1") or 1))
except Exception:
    _AUTO_FREQ_MIN = 1

def _list_day_files() -> list[str]:
    """列出 input/history 下面的 YYYYMMDD.csv（新→舊排序）"""
    files = []
    if os.path.isdir(HISTORY_DIR):
        for name in os.listdir(HISTORY_DIR):
            if re.match(r"^\d{8}\.csv$", name):
                files.append(name)
    return sorted(files, reverse=True)

def _merge_days_to_history(selected: list[str] | None) -> str:
    """把選取的日檔（YYYYMMDD.csv）合併成 history.csv，並依 (store_id, device_id, ts) 去重。"""
    import pandas as pd
    if not selected:
        return "⚠️ 請先選擇要合併的日檔"
    frames = []
    for name in selected:
        p = os.path.join(HISTORY_DIR, name)
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            # 對齊欄位（缺的補，順序固定）
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df = df[HISTORY_COLUMNS]
            frames.append(df)
        except Exception:
            # 個別檔錯誤跳過，繼續下一個
            continue
    if not frames:
        return "⚠️ 沒有可用的日檔可合併"
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["store_id", "device_id", "ts"])
    os.makedirs(HISTORY_DIR, exist_ok=True)
    out.to_csv(HISTORY_PATH, index=False)
    return f"✅ 已合併 {len(selected)} 份日檔，共 {len(out)} 筆 → {HISTORY_PATH}（去重後）"

def _read_json_with_retry(path: str, retries: int = 3, delay_sec: float = 0.05) -> dict:
    """為了配合原子寫入（.tmp → replace），讀檔時做短暫重試，避免讀到半檔或瞬間不存在。"""
    for _ in range(max(1, retries)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            time.sleep(delay_sec)
    return {}

# -----------------------------
# 功率與 ingest_ts 計算邏輯
# -----------------------------
def _calculate_ingest_ts(device_data_history: list, current_defrost_status: int) -> float:
    """
    計算距離上次除霜的時間（分鐘）
    device_data_history: 該設備的歷史資料（按時間排序）
    current_defrost_status: 當前除霜狀態
    
    邏輯：
    1. 如果當前正在除霜(1)，返回 0
    2. 如果不在除霜(0)，向前搜尋最近的除霜結束點(1→0轉換)
    3. 計算從該點到現在的分鐘數
    """
    if current_defrost_status == 1:
        return 0.0
    
    # 向前搜尋最近的 1→0 轉換點
    for i in range(len(device_data_history) - 1, 0, -1):
        prev_status = device_data_history[i-1].get("defrost_status", 0)
        curr_status = device_data_history[i].get("defrost_status", 0)
        
        if prev_status == 1 and curr_status == 0:
            # 找到除霜結束點，計算時間差
            end_time = datetime.fromisoformat(device_data_history[i]["ts"])
            now = datetime.now()
            diff_minutes = (now - end_time).total_seconds() / 60
            return max(0.0, diff_minutes)
    
    # 如果沒找到除霜記錄，返回一個預設值或 None
    return 0.0

def _get_power_by_device_type(device_type: str) -> float:
    """根據設備類型返回對應功率"""
    return POWER_SETTINGS.get(device_type, 0.0)

def _update_power_settings(freezer_power, refrigerator_power, open_case_power, hvac_power):
    """更新功率設定"""
    global POWER_SETTINGS
    try:
        POWER_SETTINGS["冷凍"] = float(freezer_power or 5.0)
        POWER_SETTINGS["冷藏"] = float(refrigerator_power or 4.5)  
        POWER_SETTINGS["開放櫃"] = float(open_case_power or 5.7)
        POWER_SETTINGS["空調"] = float(hvac_power or 0.0)
        return f"✅ 功率設定已更新：冷凍={POWER_SETTINGS['冷凍']}kW, 冷藏={POWER_SETTINGS['冷藏']}kW, 開放櫃={POWER_SETTINGS['開放櫃']}kW, 空調={POWER_SETTINGS['空調']}kW"
    except Exception as e:
        return f"⚠️ 功率設定更新失敗：{e}"

# -----------------------------
# 背景資料追加：多店定時寫入（去重）
# -----------------------------
def _load_selected_store_ids():
    """優先 Selected_Store_IDs；若為空，再退回 All_Store_IDs。"""
    try:
        with open(CONFIG_JSON, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
        sids = [s for s in (cfg.get("Selected_Store_IDs") or []) if isinstance(s, str)]
        if sids:
            return sids
        # fallback
        return [s for s in (cfg.get("All_Store_IDs") or []) if isinstance(s, str)]
    except Exception:
        return []

def _to_naive_minute_iso(ts_iso: str) -> str | None:
    if not ts_iso:
        return None
    try:
        t = datetime.fromisoformat(ts_iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        # 轉 UTC → 落到整分 → 去 tz
        t = t.astimezone(timezone.utc).replace(second=0, microsecond=0).replace(tzinfo=None)
        return t.isoformat(timespec="seconds")
    except Exception:
        return None

def _load_device_history(device_id: str, days_back: int = 7) -> list:
    """載入設備的歷史除霜資料，用於計算 ingest_ts"""
    history = []
    try:
        # 這裡簡化實現，實際應該從歷史資料庫或檔案中讀取
        # 可以讀取過去幾天的日檔來重建歷史
        for i in range(days_back):
            date = (datetime.now() - pd.Timedelta(days=i)).strftime("%Y%m%d")
            file_path = os.path.join(HISTORY_DIR, f"{date}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                device_data = df[df["device_id"] == device_id].sort_values("ts")
                for _, row in device_data.iterrows():
                    history.append({
                        "ts": row["ts"],
                        "defrost_status": row.get("defrost_status", 0)
                    })
        
        # 按時間排序
        history.sort(key=lambda x: x["ts"])
    except Exception:
        pass
    
    return history

def _append_latest_to_history_multi_once() -> str:
    """執行一次：迭代 Selected_Store_IDs，去重後追加寫入 history.csv。"""
    import csv
    sids = _load_selected_store_ids()
    if not sids:
        return "⚠️ config.json 沒有 Selected_Store_IDs 可用"

    os.makedirs(HISTORY_DIR, exist_ok=True)
    day_path = _history_day_path()
    seen = set()
    if os.path.exists(day_path) and os.path.getsize(day_path) > 0:
        try:
            with open(day_path, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    seen.add((row.get("store_id"), row.get("device_id"), row.get("ts")))
        except Exception:
            pass

    header = HISTORY_COLUMNS
    new_rows = []
    no_latest = []     # 記錄沒有 latest 快照的門市    
    # ingest_ts 可留本地 naive ISO；若想保留 UTC 請自行改回
    now_iso = datetime.now().replace(microsecond=0).isoformat()
    per_store_counts = {}

    for sid in sids:
        path = os.path.join(LATEST_DIR, f"{sid}.json")
        if not os.path.exists(path):
            per_store_counts[sid] = 0
            no_latest.append(sid)
            continue
        snap = _read_json_with_retry(path)
        if not snap:
            per_store_counts[sid] = 0
            continue
        devices = snap.get("devices", []) or []

        added = 0
        for d in devices:
            ts_src = d.get("timestamp")
            ts = _to_naive_minute_iso(ts_src)  # 單一欄位 ts：整分、無時區
            device_id = f"{sid}-{d.get('Device_ID')}"
            device_type = d.get("Device_type")
            defrost_status = d.get("Defrost_status", 0)
            
            # 計算功率
            power = _get_power_by_device_type(device_type)
            
            # 計算 ingest_ts（距離上次除霜的分鐘數）
            ingest_ts = None
            if device_type in ["冷凍", "冷藏", "開放櫃"]:  # 只有這些設備計算除霜時間
                device_history = _load_device_history(device_id)
                ingest_ts = _calculate_ingest_ts(device_history, defrost_status)
            
            row = {
                "ts": ts,
                "store_id": sid,
                "device_id": device_id,
                "device_type": device_type,
                "temp_current": d.get("Temp_current"),
                "t_room": d.get("T_room"),
                "defrost_status": defrost_status,
                "power": power,
                "ingest_ts": ingest_ts,
            }
            key = (row["store_id"], row["device_id"], row["ts"])
            if None in key[:3]:  # ts, store_id, device_id 不能為空
                continue
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(row)
            added += 1
        per_store_counts[sid] = added

    write_header = (not os.path.exists(day_path)) or (os.path.getsize(day_path) == 0)
    if new_rows:
        import csv
        with open(day_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    else:
        # 沒有新資料時，若日檔尚未建立也寫出表頭，讓你看到實體檔案
        if write_header:
            import csv
            with open(day_path, "w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()

    total = sum(per_store_counts.values())
    detail = ", ".join([f"{sid}:{cnt}" for sid, cnt in per_store_counts.items()])

    # 今日總累計：直接統計「當日日檔」列數（排除表頭）
    try:
        import csv
        today_total = 0
        if os.path.exists(day_path) and os.path.getsize(day_path) > 0:
            with open(day_path, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for _ in rdr:
                    today_total += 1
    except Exception:
        today_total = 0

    # 依你需求的訊息格式輸出
    cause = f"；缺少快照門市={no_latest}" if no_latest else ""
    return (
        f"✅ 多店寫入完成：當前資料 {total} 筆，今日總累計 {today_total} 筆"
        f"（各店新增：{detail}）→ {day_path}（去重後）{cause}"
    )

def _auto_append_loop(freq_min: int):
    global running, _last_status, interval
    interval = max(10, int(freq_min) * 60)  # 秒
    while running:
        try:
            _last_status = _append_latest_to_history_multi_once()
        except Exception as e:
            _last_status = f"⚠️ 追加失敗：{e}"
        for _ in range(interval):
            if not running:
                break
            time.sleep(1)
    _last_status = "已停止"

def start_append(freq_min: float = 1):
    global running
    if not running:
        running = True
        _t = threading.Thread(target=_auto_append_loop, args=(max(1, int(freq_min)),))
        _t.start()
    return "<span style='color:red;'>多店定時寫入已啟動（去重）</span>"

def stop_append():
    global running
    running = False
    return "已停止（背景循環會在當次寫入後結束）"

def get_status():
    return _last_status

# ---- 模組載入：若啟用自動模式，開機即自動啟動背景追加（像 CLI）----
if _AUTO_ENABLE:
    try:
        # 防多次載入／重複啟動：由 running 旗標防重
        _ = start_append(_AUTO_FREQ_MIN)
        _last_status = f"自動模式：已啟動（每 {_AUTO_FREQ_MIN} 分）"
    except Exception as e:
        print(f"[evaluate_tab] 自動啟動失敗：{e}")
        _last_status = f"⚠️ 自動啟動失敗：{e}"


# 讀取最新 10 筆 history.csv
def load_latest_history():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    day_path = _history_day_path()
    src = day_path if (os.path.exists(day_path) and os.path.getsize(day_path) > 0) else HISTORY_PATH
    if (not os.path.exists(src)) or (os.path.getsize(src) == 0):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(src)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=COLUMNS)
    
    # 只顯示你要的欄位名稱：ts → Time_stamp，其餘對應大小寫
    rename_map = {
        "ts": "Time_stamp", "store_id": "Store_ID", "device_id": "Device_ID",
        "device_type": "Device_type", "temp_current": "Temp_current",
        "t_room": "T_room", "defrost_status": "Defrost_status", "power": "Power"
    }
    for col in COLUMNS:
        if col not in df.columns and col not in rename_map.values():
            df[col] = None
    df = df.rename(columns=rename_map)
    return df[["Time_stamp","Store_ID","Device_ID","Device_type","Temp_current","T_room","Defrost_status","Power"]].tail(10)

def clear_history():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    # 寫出只有表頭的 CSV（history 欄位）
    pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(HISTORY_PATH, index=False)
    return "已清除歷史資料", pd.DataFrame([{"提示": "⚠️ 尚無歷史資料"}])


# -----------------------------
# Gradio render
# -----------------------------
def render():
    with gr.Tab("設備卸載評估"):

        # ============= 資料儲存 =============
        with gr.Tab("資料儲存"):
            with gr.Row():
                freq_min = gr.Number(label="多店寫入頻率（分鐘）", value=1, precision=0)
                start_btn = gr.Button("開始追加（多店｜去重）")
                stop_btn = gr.Button("停止追加")
                clear_btn = gr.Button("清除歷史資料")
            
            # 新增功率設定區塊
            with gr.Accordion("設備功率設定", open=True):
                gr.Markdown("### 各類型設備功率設定（kW）")
                with gr.Row():
                    freezer_power = gr.Number(label="冷凍設備功率", value=5.0, precision=1, minimum=0)
                    refrigerator_power = gr.Number(label="冷藏設備功率", value=4.5, precision=1, minimum=0)
                    open_case_power = gr.Number(label="開放櫃功率", value=5.7, precision=1, minimum=0)
                    hvac_power = gr.Number(label="空調功率", value=0.0, precision=1, minimum=0)
                
                update_power_btn = gr.Button("更新功率設定", variant="secondary")
                power_status = gr.Markdown(f"目前設定：冷凍={POWER_SETTINGS['冷凍']}kW, 冷藏={POWER_SETTINGS['冷藏']}kW, 開放櫃={POWER_SETTINGS['開放櫃']}kW, 空調={POWER_SETTINGS['空調']}kW")

            status_box = gr.Markdown("尚未開始")
            history_table = gr.DataFrame(interactive=False, wrap=True)

            # 設定頻率並啟動
            def _set_interval_and_start(mins: float):
                try:
                    m = max(1, int(mins))
                except Exception:
                    m = 1
                return start_append(m)

            start_btn.click(_set_interval_and_start, inputs=freq_min, outputs=status_box)
            stop_btn.click(stop_append, outputs=status_box)
            clear_btn.click(clear_history, outputs=[status_box, history_table])
            
            # 功率更新事件
            update_power_btn.click(
                _update_power_settings,
                inputs=[freezer_power, refrigerator_power, open_case_power, hvac_power],
                outputs=power_status
            )

            timer = gr.Timer(10)
            timer.tick(fn=load_latest_history, outputs=history_table)
            # 額外輪詢一下狀態文字
            status_timer = gr.Timer(10)
            status_timer.tick(fn=get_status, outputs=status_box)

            # ============= 合併日檔成 history.csv（多選） =============
            with gr.Accordion("合併日檔 → 匯出 history.csv", open=False):
                refresh_days_btn = gr.Button("重新載入日檔列表")
                day_files = gr.CheckboxGroup(
                    label="選擇日檔（YYYYMMDD.csv，可複選）",
                    choices=_list_day_files()
                )
                merge_days_btn = gr.Button("合併成 history.csv")
                merge_status = gr.Textbox(label="合併狀態", interactive=False)

                def _refresh_day_choices():
                    # 兼容舊版 Gradio：用通用 gr.update(...) 回傳元件更新
                    return gr.update(choices=_list_day_files())

                refresh_days_btn.click(_refresh_day_choices, outputs=day_files)
                merge_days_btn.click(_merge_days_to_history, inputs=day_files, outputs=merge_status)

        # ========= 新「模型訓練」子分頁（嵌入 train_tab 面板） =========
        with gr.Tab("模型訓練"):
            render_train_panel()

         # ========= 預測與分析 子分頁（新版嵌入）=========
        with gr.Tab("預測與分析"):
            render_predict_panel()