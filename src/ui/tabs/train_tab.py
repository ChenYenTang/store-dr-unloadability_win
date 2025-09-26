import os, re
import json
import threading
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pandas as pd
import gradio as gr
import io
import time

_AUTO_THREAD = None
_AUTO_STOP = threading.Event()
_AUTO_LOCK = threading.Lock()

# 修改匯入路徑，使用你的 lstm_trend.py
from src.AI_models.lstm_trend import train_store_lstm

# ==== 修復：取消註解路徑常數 ====
WEIGHT_DIR = "weighted_models"
INDEX_PATH = os.path.join(WEIGHT_DIR, "index.json")

SCHEMA = ["ts","store_id","device_id","device_type","temp_current","t_room","defrost_status","power","ingest_ts"]

def _safe_read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    # 最後一招：二進位讀→強制解碼
    with open(path, "rb") as f:
        raw = f.read()
    txt = raw.decode("cp950", errors="replace")
    return pd.read_csv(io.StringIO(txt), on_bad_lines="skip")

def _load_history_range(start_date: str, end_date: str, selected_store_ids: list[str]) -> pd.DataFrame:
    """從 input/history/ 讀取 [start_date, end_date] 所有日檔、只保留選定門市，回傳合併後 DF。
       若沒有任何可用資料，拋出 ValueError，訊息包含 missing/bad 檔數。"""
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    paths = [os.path.join("input", "history", d.strftime("%Y%m%d") + ".csv") for d in days]
    exist = [p for p in paths if os.path.isfile(p)]
    missing = [p for p in paths if not os.path.isfile(p)]
    
    print(f"DEBUG: Looking for files from {start_date} to {end_date}")
    print(f"DEBUG: Found {len(exist)} files, missing {len(missing)} files")
    
    if not exist:
       raise ValueError(f"No daily files found under input/history for range {start_date} ~ {end_date}")

    frames, bad = [], []
    for p in exist:
        try:
            df = _safe_read_csv(p)
            print(f"DEBUG: Loaded {os.path.basename(p)}, shape: {df.shape}")
            
            # 補齊欄位
            for c in SCHEMA:
                if c not in df.columns:
                    df[c] = pd.NA
            
            # 過濾門市
            if selected_store_ids:
                df = df[df["store_id"].astype(str).isin([str(s) for s in selected_store_ids])]
                print(f"DEBUG: After filtering stores {selected_store_ids}, shape: {df.shape}")
            
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"DEBUG: Failed to load {os.path.basename(p)}: {e}")
            bad.append(f"{os.path.basename(p)}: {e}")

    if not frames:
        detail = []
        if missing: detail.append(f"missing={len(missing)}")
        if bad:     detail.append(f"bad={len(bad)}")
        raise ValueError("No objects to concatenate" + (": " + "; ".join(detail) if detail else ""))

    # 串接並固定欄位/型別
    frames = [f.dropna(axis=1, how="all") for f in frames]
    df_all = pd.concat(frames, ignore_index=True, sort=False).dropna(axis=1, how="all")
    
    print(f"DEBUG: Combined all frames, shape: {df_all.shape}")
    
    for c in SCHEMA:
       if c not in df_all.columns:
           df_all[c] = pd.NA
    df_all["ts"] = pd.to_datetime(df_all["ts"], errors="coerce")
    for c in ("store_id","device_id","device_type"):
       df_all[c] = df_all[c].astype("string")
    for c in ("temp_current","t_room","power","ingest_ts"):
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    df_all["defrost_status"] = pd.to_numeric(df_all["defrost_status"], errors="coerce").astype("Int64")
    
    print(f"DEBUG: Final processed data shape: {df_all.shape}")
    return df_all

# 舊函式改成委派到新實作，避免兩套邏輯分岐
def _load_range_df(start_date: str, end_date: str, store_id: str) -> pd.DataFrame:
    return _load_history_range(start_date, end_date, [store_id])

# ---- 共享狀態（每個進程內） ----
STATUS: Dict[str, Dict[str, Any]] = {}
STOP_EVENT = threading.Event()
STATUS_LOCK = threading.Lock()

# === 逐店訓練（單一背景執行緒）狀態 ===
_SEQ_TRAIN_THREAD: "threading.Thread|None" = None
_SEQ_LOCK = threading.Lock()
_SEQ_STOP = threading.Event()

#  關閉 UI 端補寫，避免與後端同時寫檔造成 index.json 壞檔
def _update_index_after_train(*_args, **_kwargs):
    return None

def _parse_date(s: str):
    """將 YYYY-MM-DD 文字轉成 date；失敗回傳 None。"""
    if not s:
        return None
    s = str(s).strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _read_config_selected() -> List[str]:
    cfg_path = os.path.join("input", "config.json")
    if not os.path.exists(cfg_path):
        return []
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("Selected_Store_IDs", [])

def _load_index() -> Dict[str, Any]:
    """讀 weighted_models/index.json；若毀檔，回傳 {} 並讓 UI 顯示『無可用模型』。"""
    try:
        if not os.path.exists(INDEX_PATH):
            return {}
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"DEBUG: Failed to load index: {e}")
        # 避免整頁炸掉，先讓 UI 起來；細節交給後端重新寫入修復。
        return {}

def _scan_missing_dates(start_date: str, end_date: str) -> List[str]:
    missing = []
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    d = start
    while d <= end:
        p = os.path.join("input", "history", d.strftime("%Y%m%d") + ".csv")
        if not os.path.exists(p):
            missing.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return missing

def _empty_progress_row(store_id: str, idx_latest: Dict[str, Any]):
    latest = None
    if store_id in idx_latest and idx_latest[store_id].get("latest"):
        latest = idx_latest[store_id]["latest"]
    return {
        "Store_ID": store_id,
        "Status": f"待機{'（已存在 ' + latest + '）' if latest else ''}",
        "Progress(%)": 0,
        "Epoch": 0,
        "Loss": "",
        # 修改：調整為 LSTM Survival 模型的 metrics 顯示
        "TTW_MAE": "", "Hit_±3min": "", "P50_Cal": "", 
        "Model": "", "Metrics": "",
    }

def _ensure_status_rows(selected: List[str], idx_latest: Dict[str, Any]):
    with STATUS_LOCK:
        for sid in selected:
            if sid not in STATUS:
                STATUS[sid] = _empty_progress_row(sid, idx_latest)

class _CB:
    def __init__(self, store_id: str, total_epochs: int):
        self.store_id = store_id
        # 保留總 epoch（雖然 on_epoch_end 也會傳進 total_epochs，但存起來更穩）
        self.total_epochs = int(total_epochs)
    def should_stop(self) -> bool:
        return STOP_EVENT.is_set()
    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        if STOP_EVENT.is_set():
            return
        with STATUS_LOCK:
            row = STATUS.get(self.store_id, {})
            row["Status"] = "訓練中"
            row["Epoch"] = epoch
            row["Loss"] = round(float(loss), 6)
            total = int(total_epochs) if int(total_epochs) > 0 else getattr(self, "total_epochs", 0) or 1
            row["Progress(%)"] = int(epoch / max(1, total) * 100)
            STATUS[self.store_id] = row

def _worker(store_id: str, df: pd.DataFrame, params: Dict[str, Any]):
    try:
        print(f"DEBUG: Worker starting for {store_id}, df.shape: {df.shape}")

        if STOP_EVENT.is_set():
            print(f"DEBUG: STOP_EVENT already set for {store_id}")
            return
        cb = _CB(store_id, int(params.get("epochs", 10)))
        
        # 檢查數據有效性
        if df.empty:
            print(f"DEBUG: DataFrame is empty for {store_id}")            
            with STATUS_LOCK:
                STATUS[store_id]["Status"] = "失敗：數據為空"
            return
            
        model_path, metrics_path, ver = train_store_lstm(
            store_id=store_id,
            df=df,
            params=params,
            callbacks=cb,
        )
        
        print(f"DEBUG: Training completed for {store_id}, version: {ver}")
        
        # （條件式保險）若後端尚未把本次版本寫進 models/index.json，才補寫一次
        try:
            idx_now = _load_index()
            ent = (idx_now or {}).get(store_id, {}) or {}
            have = False
            for v in (ent.get("versions") or []):
                if v.get("version") == str(ver):
                    have = True; break
                # 或者以檔名（去副檔名）比對
                if os.path.splitext(os.path.basename(v.get("model","")))[0] == os.path.splitext(os.path.basename(model_path or ""))[0]:
                    have = True; break
            if not have or (ent.get("latest") != str(ver)):
                # 僅在缺少時才補寫，避免與後端重複/競爭寫入
                _update_index_after_train(store_id, model_path, metrics_path, version_hint=str(ver or ""))
        except Exception:
            pass

        # 讀 metrics 填入表格 - 修改：適配 LSTM Survival 模型的輸出格式
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        with STATUS_LOCK:
            row = STATUS.get(store_id, {})
            row.update({
                "Status": f"完成 {ver}",
                "Progress(%)": 100,
                # 修改：使用 LSTM Survival 模型的實際 metrics
                "TTW_MAE": round(float(m.get("mae_ttw", 0)), 3),
                "Hit_±3min": f"{float(m.get('hit_±3min', 0)):.1%}",
                "P50_Cal": f"{float(m.get('p50_cal', 0)):.1%}",
                "Model": model_path, 
                "Metrics": metrics_path,
            })
            STATUS[store_id] = row
    except Exception as e:
        print(f"DEBUG: Worker failed for {store_id}: {e}")
        with STATUS_LOCK:
            row = STATUS.get(store_id, {})
            row["Status"] = f"失敗：{e}"
            STATUS[store_id] = row

def _start_training(selected, start_date, end_date, window, horizon, hidden, layers, dp, bs, ep, lr_, 
                  features, step_val, safe_fridge_val, safe_freezer_val, safe_open_case_val,
                  near_limit_fridge_val, near_limit_freezer_val, near_limit_open_case_val,
                  safety_buffer_freezer_val, safety_buffer_other_val,
                  existing_policy="all_retrain", model_type_val: str = "LSTM_trend"):
    print(f"DEBUG: STOP_EVENT initial state: {STOP_EVENT.is_set()}")
    print(f"DEBUG: _SEQ_STOP initial state: {_SEQ_STOP.is_set()}")
    
    STOP_EVENT.clear()
    print(f"DEBUG: STOP_EVENT after clear: {STOP_EVENT.is_set()}")
    
    # 日期字串解析
    d_start = _parse_date(start_date)
    d_end = _parse_date(end_date)
    if (d_start is None) or (d_end is None):
        return "⚠️ 日期格式錯誤，請輸入 YYYY-MM-DD（例如 2025-09-20）"
    if d_start > d_end:
        return "⚠️ 起日不可晚於訖日"

    print(f"DEBUG: Starting training for stores: {selected}")
    print(f"DEBUG: Date range: {start_date} to {end_date}")

    # 往後邏輯一律使用 d_start / d_end
    STOP_EVENT.clear()
    idx = _load_index()
    _ensure_status_rows(selected, idx)
    warnings = []
    missing = _scan_missing_dates(start_date, end_date)
    if missing:
        warnings.append(f"缺少日檔：{', '.join(missing)}（仍可繼續）")
    if len(selected) > 5:
        warnings.append("同時選擇超過 5 家門市，訓練可能變慢。")

    # 依「既有模型處理策略」
    # existing_policy: "all_load" | "all_retrain" | dict(per_store)
    policy = existing_policy or "all_retrain"
    if isinstance(policy, dict):
        # per-store 動作表：{"S001":"load","S003":"retrain", ...}
        per_store_action = {sid: (policy.get(sid) or "retrain") for sid in selected}
    elif policy == "all_load":
        per_store_action = {sid: "load" for sid in selected}
    elif policy == "all_retrain":
        per_store_action = {sid: "retrain" for sid in selected}
    else:
        # 傳入未知字串時，退回預設 all_retrain
        per_store_action = {sid: "retrain" for sid in selected}

    # === 啟動「逐店」worker：一條 thread 順序執行所有門市 ===
    def _sequential_worker(store_ids: List[str]):
        for sid in store_ids:
            if _SEQ_STOP.is_set():
                print(f"DEBUG: Sequential training stopped for {sid}")
                with STATUS_LOCK:
                    if sid in STATUS:
                        STATUS[sid]["Status"] = "已停止"
                break
            latest = idx.get(sid, {}).get("latest")
            act = None
            if latest:
                if isinstance(policy, str):
                    act = "load" if policy == "all_load" else "retrain"
                else:
                    act = per_store_action.get(sid, "retrain")

            # 載入舊模型就略過訓練（維持原行為）
            if latest and act == "load":
                with STATUS_LOCK:
                    STATUS[sid]["Status"] = f"已載入 {latest}"
                    STATUS[sid]["Progress(%)"] = 100
                    vrec = next((v for v in idx[sid]["versions"] if v["version"] == latest), None)
                    if vrec:
                        STATUS[sid]["Model"] = vrec["model"]
                        STATUS[sid]["Metrics"] = vrec["metrics"]
                continue

            # 讀資料
            try:
                df = _load_history_range(start_date, end_date, [sid])
            except Exception as e:
                print(f"DEBUG: Failed to load data for {sid}: {e}")
                with STATUS_LOCK:
                    STATUS[sid]["Status"] = f"失敗：{e}"
                    STATUS[sid]["Progress(%)"] = 0
                    STATUS[sid]["Epoch"] = 0
                    STATUS[sid]["Loss"] = ""
                continue
            if df.empty:
                with STATUS_LOCK:
                    STATUS[sid]["Status"] = "失敗：此區間無任何可用資料"
                    STATUS[sid]["Progress(%)"] = 0
                    STATUS[sid]["Epoch"] = 0
                    STATUS[sid]["Loss"] = ""
                continue
            with STATUS_LOCK:
                STATUS[sid]["Status"] = "排隊中"
                STATUS[sid]["Progress(%)"] = 0
                STATUS[sid]["Epoch"] = 0
                STATUS[sid]["Loss"] = ""

            # 檢查每台 device 分鐘數
            need = int(window) + int(horizon)
            df_chk = df.copy()
            df_chk["ts"] = pd.to_datetime(df_chk["ts"], errors="coerce")
            bad = []
            print(f"DEBUG: Checking devices for {sid}, need {need} minutes")

            for did, g in df_chk.groupby("device_id", sort=False):
                n_min = g.drop_duplicates(subset=["ts"]).shape[0]
                if n_min < need:
                    bad.append((str(did), n_min))
            if bad:
                msg = "; ".join([f"{d}:{n}/{need}" for d, n in bad])
                with STATUS_LOCK:
                    STATUS[sid]["Status"] = f"跳過：每台需≥{need}分鐘，以下不足 → {msg}"
                    STATUS[sid]["Progress(%)"] = 0
                    STATUS[sid]["Epoch"] = 0
                    STATUS[sid]["Loss"] = ""
                continue

            # 準備參數（避免被後續門市覆蓋）
            params = dict(
                window=int(window), horizon=int(horizon),
                hidden=int(hidden), layers=int(layers),
                dropout=float(dp), batch_size=int(bs), epochs=int(ep), lr=float(lr_),
                _range={"start": start_date, "end": end_date},
                model_type=str(model_type_val),
                features=list(features) if features else ["temp_current","t_room","defrost_status","power"],
                step=int(step_val),
                safe_fridge=float(safe_fridge_val), safe_freezer=float(safe_freezer_val), safe_open_case=float(safe_open_case_val),
                near_limit_fridge=float(near_limit_fridge_val), near_limit_freezer=float(near_limit_freezer_val), near_limit_open_case=float(near_limit_open_case_val),
                safety_buffer_freezer=int(safety_buffer_freezer_val), safety_buffer_other=int(safety_buffer_other_val),
            )
            # 直接呼叫原本的單店 worker（同步執行，跑完才換下一家）
            _worker(sid, df, deepcopy(params))

    with _SEQ_LOCK:
        global _SEQ_TRAIN_THREAD
        if _SEQ_TRAIN_THREAD and _SEQ_TRAIN_THREAD.is_alive():
            return "已有一批訓練在進行，請先停止或等待完成。"
        _SEQ_STOP.clear()
        _SEQ_TRAIN_THREAD = threading.Thread(target=_sequential_worker, args=(selected,))
        _SEQ_TRAIN_THREAD.start()
    return "\n".join(warnings) if warnings else "已開始（逐店訓練）。"

# === 自動訓練：新增 ===
def _auto_stop():
    _AUTO_STOP.set()
    t = globals().get("_AUTO_THREAD")
    if t and t.is_alive():
        t.join(timeout=5)
    with _AUTO_LOCK:
        globals()["_AUTO_THREAD"] = None
    return "自動訓練已關閉。"

def _auto_start(snapshot_args: dict, interval_min: int):
    """snapshot_args：從 UI 讀到的『當下參數』快照；之後每輪用同一套參數自動觸發。"""
    if interval_min is None or int(interval_min) <= 0:
        return "⚠️ 間隔需為正整數分鐘。"
    with _AUTO_LOCK:
        t = globals().get("_AUTO_THREAD")
        if t and t.is_alive():
            return f"自動訓練已啟動中（每 {interval_min} 分鐘）。"
        _AUTO_STOP.clear()

        def _loop():
            while not _AUTO_STOP.is_set():
                try:
                    # 若上一輪訓練還在跑，這輪就略過，避免重疊
                    running = globals().get("_SEQ_TRAIN_THREAD")
                    if running is None or (not running.is_alive()):
                        _start_training(**snapshot_args)
                    else:
                        print("DEBUG: 自動訓練略過本輪（前一批次仍在進行）。")
                except Exception as e:
                    print(f"DEBUG: Auto-train iteration failed: {e}")
                # 以 1 秒為粒度可即時回應停用
                remain = int(interval_min) * 60
                while remain > 0 and (not _AUTO_STOP.is_set()):
                    time.sleep(1)
                    remain -= 1
            print("DEBUG: Auto-train loop exited.")

        th = threading.Thread(target=_loop, daemon=True)
        globals()["_AUTO_THREAD"] = th
        th.start()
        return f"自動訓練已啟動（每 {interval_min} 分鐘）。"


def _stop_all():
    # 停止目前 epoch（由模型 callback 感知）
    STOP_EVENT.set()
    # 停止序列批次的後續門市
    _SEQ_STOP.set()
    t = globals().get("_SEQ_TRAIN_THREAD")
    if t and t.is_alive():
        t.join(timeout=5)
    return "已發出停止指令（逐店模式也已停止）。"

def _refresh_table(selected: List[str]) -> pd.DataFrame:
    idx = _load_index()
    _ensure_status_rows(selected or [], idx)
    with STATUS_LOCK:
        rows = [STATUS[sid] for sid in STATUS.keys() if (not selected) or sid in selected]
    if not rows:
        # 修改表格欄位，移除傳統 ML metrics，加入 LSTM Survival 的 metrics
        return pd.DataFrame(columns=["Store_ID","Status","Progress(%)","Epoch","Loss","TTW_MAE","Hit_±3min","P50_Cal","Model","Metrics"])
    return pd.DataFrame(rows)[["Store_ID","Status","Progress(%)","Epoch","Loss","TTW_MAE","Hit_±3min","P50_Cal","Model","Metrics"]]

# 確保函數名稱正確並加上所有必要的匯出
def render_panel():  
    gr.Markdown("### 模型訓練（LSTM Survival）")
    # 修改：簡化模型選擇，只保留支援的模型
    with gr.Row():
        model_type = gr.Dropdown(
            label="AI 模型",
            choices=["LSTM_trend"],   # 只保留你實作的模型
            value="LSTM_trend"
        )
    # 預設門市與 index 先計算，供元件初值使用
    selected_default = _read_config_selected()
    idx0 = _load_index()
    with gr.Row():
        store_choices = gr.CheckboxGroup(
            label="要訓練的門市（預設來自 input/config.json 的 Selected_Store_IDs）",
            choices=selected_default, value=selected_default
        )
        with gr.Column(scale=0):
            refresh_btn = gr.Button("重新整理", variant="secondary")
            select_all_btn = gr.Button("全選")
            clear_all_btn = gr.Button("全不選")
    with gr.Row():
        start_date = gr.Textbox(
            label="資料期間（起，YYYY-MM-DD）",
            value=(datetime.now().date() - timedelta(days=7)).isoformat(),
            placeholder="例如：2025-09-13"
        )
        end_date = gr.Textbox(
            label="資料期間（訖，YYYY-MM-DD）",
            value=datetime.now().date().isoformat(),
            placeholder="例如：2025-09-20"
        )
        auto_range_btn = gr.Button("使用全部可用期間", variant="secondary")

    def _auto_range():
        # 掃描 input/history 下的 YYYYMMDD.csv 取最早/最新
        base = os.path.join("input", "history")
        try:
            names = [f for f in os.listdir(base) if re.match(r"^\d{8}\.csv$", f)]
            if not names:
                return gr.update(), gr.update()
            days = sorted(int(n[:8]) for n in names)
            dmin = datetime.strptime(str(days[0]), "%Y%m%d").date().isoformat()
            dmax = datetime.strptime(str(days[-1]), "%Y%m%d").date().isoformat()
            return gr.update(value=dmin), gr.update(value=dmax)
        except Exception:
            return gr.update(), gr.update()
    auto_range_btn.click(_auto_range, [], [start_date, end_date])
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("**LSTM Survival 參數**")
            # 修改參數名稱和預設值，配合 LSTM survival 模型
            window = gr.Number(label="觀測窗口 (window)", value=120, precision=0, info="過去N分鐘的溫度歷史")
            horizon = gr.Number(label="預測範圍 (horizon)", value=60, precision=0, info="預測未來N分鐘內越界時間")
            hidden = gr.Number(label="隱藏層大小 (hidden)", value=64, precision=0)
            layers = gr.Number(label="LSTM層數 (layers)", value=2, precision=0)
            dropout = gr.Slider(label="dropout", minimum=0.0, maximum=0.8, value=0.1, step=0.05)
            batch_size = gr.Number(label="batch_size", value=64, precision=0)
            epochs = gr.Number(label="epochs", value=8, precision=0)
            lr = gr.Number(label="學習率 (lr)", value=1e-3)
        with gr.Column():
            gr.Markdown("**溫度安全參數**")
            safe_fridge = gr.Number(label="冷藏安全溫度 (°C)", value=7.0, info="冷藏設備安全上限")
            safe_freezer = gr.Number(label="冷凍安全溫度 (°C)", value=-18.0, info="冷凍設備安全上限")
            safe_open_case = gr.Number(label="開放櫃安全溫度 (°C)", value=7.0, info="開放櫃設備安全上限")
            near_limit_fridge = gr.Number(label="冷藏近臨界值 (°C)", value=6.5, info="冷藏預警溫度")
            near_limit_freezer = gr.Number(label="冷凍近臨界值 (°C)", value=-17.5, info="冷凍預警溫度")
            near_limit_open_case = gr.Number(label="開放櫃近臨界值 (°C)", value=6.5, info="開放櫃預警溫度")
        with gr.Column():
            with gr.Accordion("進階參數", open=False):
                gr.Markdown("**特徵選擇**")
                features_group = gr.CheckboxGroup(
                    label="訓練特徵",
                    choices=["temp_current", "t_room", "defrost_status", "power", "ingest_ts"],
                    value=["temp_current", "t_room", "defrost_status", "power"],
                    info="選擇用於訓練的特徵欄位"
                )
                step = gr.Number(label="窗格步進 (step)", value=1, precision=0, info="滑動窗口的步進間隔（分鐘）")
                
                gr.Markdown("**安全緩衝參數**")
                safety_buffer_freezer = gr.Number(label="冷凍安全緩衝 (分鐘)", value=3, precision=0, info="冷凍設備的安全緩衝時間")
                safety_buffer_other = gr.Number(label="其他設備安全緩衝 (分鐘)", value=2, precision=0, info="冷藏/開放櫃的安全緩衝時間")
                
            with gr.Accordion("既有模型處理", open=False):
                existing_mode = gr.Radio(
                    label="對已存在模型之處理",
                    choices=[("全部載入","all_load"), ("全部重新訓練","all_retrain")],
                    value="all_retrain"
                )
                gr.Markdown("若需個別指定，可在下表「Action」欄選擇（僅對已存在模型的門市有效）。")
    
    # 可編輯的 per-store action 表（建立初值）
    _rows_init = []
    for _sid in (selected_default or []):
        _latest = idx0.get(_sid, {}).get("latest", "")
        _rows_init.append([_sid, _latest or "", "retrain" if _latest else "retrain"])

    action_df = gr.Dataframe(
        headers=["Store_ID", "Latest", "Action"],
        datatype=["str", "str", "category"],
        col_count=(3, "fixed"),
        row_count=(0, "dynamic"),
        interactive=True,
        label="既有模型提示（可個別選擇 載入/重新訓練）",
        value=_rows_init
    )

    with gr.Row():
        start_btn = gr.Button("開始訓練", variant="primary")

        # === 自動訓練：新增（置於兩顆按鈕之間） ===
        with gr.Column(scale=0):
            auto_toggle = gr.Checkbox(label="自動訓練", value=False)
            auto_interval = gr.Number(label="間隔（分鐘）", value=60, precision=0)

        stop_btn = gr.Button("停止全部", variant="stop")
    warn_box = gr.Markdown("")

    # 進度表初值：帶入預設門市
    _df_init = _refresh_table(selected_default)
    # 修改表格標題，配合新的 metrics
    prog_df = gr.Dataframe(
        headers=["Store_ID", "Status", "Progress(%)", "Epoch", "Loss", "TTW_MAE", "Hit_±3min", "P50_Cal", "Model", "Metrics"],
        col_count=(10, "fixed"),  # 調整欄位數量
        row_count=(0, "dynamic"),
        wrap=True,
        interactive=False,
        label="訓練進度（LSTM Survival Metrics）",
        value=_df_init
    )
    timer = gr.Timer(1.5, active=True)

    # 初始化 / 重建 action_df 與進度表（在複選與重新整理時共用）
    def _init_tables(sids):
        idx = _load_index()
        _ensure_status_rows(sids or [], idx)
        rows = []
        for sid in (sids or []):
            latest = idx.get(sid, {}).get("latest", "")
            rows.append([sid, latest or "", "retrain" if latest else "retrain"])
        return gr.update(value=rows), _refresh_table(sids)

    store_choices.change(_init_tables, [store_choices], [action_df, prog_df])

    # 重新整理：從 config.json 讀取、更新複選與下方兩表
    def _on_refresh():
        sids = _read_config_selected() or []
        rows_update, table_update = _init_tables(sids)
        return gr.update(choices=sids, value=sids), rows_update, table_update
    refresh_btn.click(_on_refresh, [], [store_choices, action_df, prog_df])

    # 全選 / 全不選：只改 value，同步刷新兩表
    def _on_select_all():
        sids = _read_config_selected() or []
        rows_update, table_update = _init_tables(sids)
        return gr.update(value=sids), rows_update, table_update
    select_all_btn.click(_on_select_all, [], [store_choices, action_df, prog_df])

    def _on_clear_all():
        rows_update, table_update = _init_tables([])
        return gr.update(value=[]), rows_update, table_update
    clear_all_btn.click(_on_clear_all, [], [store_choices, action_df, prog_df])

    # Timer 刷新
    timer.tick(_refresh_table, [store_choices], [prog_df])

    # 開始訓練 - 修改參數傳遞，加入所有新參數
    def _on_start(sids, sd, ed, win, hor, hid, lay, dp, bs, ep, lr_, 
                  feat_list, step_val, sf_val, sfr_val, soc_val, nlf_val, nlfr_val, nloc_val, 
                  sbf_val, sbo_val, mode, act_rows, model_type_val):
        print(f"DEBUG: UI start training called with stores: {sids}")
        # 組 per-store 行為
        per = {}
        if isinstance(act_rows, list):
            for r in act_rows:
                if len(r) >= 3 and r[1]:
                    per[str(r[0])] = "load" if str(r[2]) == "load" else "retrain"
        existing_policy = mode if mode in ("all_load","all_retrain") else per
        
        try:
            msg = _start_training(
                sids or [], str(sd), str(ed),
                win, hor, hid, lay, dp, bs, ep, lr_,
                feat_list, step_val, sf_val, sfr_val, soc_val, nlf_val, nlfr_val, nloc_val,
                sbf_val, sbo_val, existing_policy, str(model_type_val)
            )
            return f"**狀態：** {msg}"
        except Exception as e:
            print(f"DEBUG: Training failed with exception: {e}")
            return f"**錯誤：** {e}"

    start_btn.click(
        _on_start,
        [store_choices, start_date, end_date, window, horizon, hidden, layers, dropout, batch_size, epochs, lr,
         features_group, step, safe_fridge, safe_freezer, safe_open_case, 
         near_limit_fridge, near_limit_freezer, near_limit_open_case,
         safety_buffer_freezer, safety_buffer_other, existing_mode, action_df, model_type],
        [warn_box]
    )
    
    # === 自動訓練：新增，勾選即啟動/關閉 ===
    def _on_auto_toggle(is_on, sids, sd, ed, win, hor, hid, lay, dp, bs, ep, lr_,
                        feat_list, step_val, sf_val, sfr_val, soc_val, nlf_val, nlfr_val, nloc_val,
                        sbf_val, sbo_val, mode, act_rows, model_type_val, interval_min):
        if not is_on:
            return "**狀態：** " + _auto_stop()

        # 組 per-store 行為（與開始訓練相同邏輯）
        per = {}
        if isinstance(act_rows, list):
            for r in act_rows:
                if len(r) >= 3 and r[1]:
                    per[str(r[0])] = "load" if str(r[2]) == "load" else "retrain"
        existing_policy = mode if mode in ("all_load","all_retrain") else per

        snapshot = dict(
            selected=sids or [],
            start_date=str(sd), end_date=str(ed),
            window=win, horizon=hor, hidden=hid, layers=lay, dp=dp, bs=bs, ep=ep, lr_=lr_,
            features=feat_list, step_val=step_val,
            safe_fridge_val=sf_val, safe_freezer_val=sfr_val, safe_open_case_val=soc_val,
            near_limit_fridge_val=nlf_val, near_limit_freezer_val=nlfr_val, near_limit_open_case_val=nloc_val,
            safety_buffer_freezer_val=sbf_val, safety_buffer_other_val=sbo_val,
            existing_policy=existing_policy, model_type_val=str(model_type_val)
        )
        msg = _auto_start(snapshot, int(interval_min or 60))
        return f"**狀態：** {msg}"

    auto_toggle.change(
        _on_auto_toggle,
        inputs=[auto_toggle,
                store_choices, start_date, end_date, window, horizon, hidden, layers, dropout, batch_size, epochs, lr,
                features_group, step, safe_fridge, safe_freezer, safe_open_case,
                near_limit_fridge, near_limit_freezer, near_limit_open_case,
                safety_buffer_freezer, safety_buffer_other, existing_mode, action_df, model_type, auto_interval],
        outputs=[warn_box]
    )


    def _stop_all_with_auto():
        msg_auto = _auto_stop()
        msg_train = _stop_all()
        return f"{msg_auto}\n{msg_train}"

    stop_btn.click(_stop_all_with_auto, [], [warn_box])


    # 首次載入已由 value 給初值；之後變更靠 change/timer 繫定更新

# 在文件末尾確保所有匯出函數都存在
def build_train_tab():
    """（可選）若需要把訓練頁作為頂層分頁時使用"""
    with gr.Tab("Train"):
        render_panel()
    return

def render():
    """向後相容的函數名稱，直接呼叫 render_panel"""
    return render_panel()

# 確保主要的 render_panel 函數可以被正確匯入
__all__ = ['render_panel', 'render', 'build_train_tab']