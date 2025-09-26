import os, json, threading, inspect, time, re, bisect
import pandas as pd
import gradio as gr
from typing import List, Dict, Any
from src.AI_models.lstm_trend import predict_store  # 統一新路徑

# === 路徑常數（可用環境變數覆寫；固定讀 weighted_models/index.json） ===
WEIGHT_DIR = os.getenv("WEIGHT_DIR", "weighted_models")
INDEX_PATH = os.path.join(WEIGHT_DIR, "index.json")

# 添加版本正則表達式
_SEQ_VER_RE = re.compile(r"^v(\d{8})_(\d+)$")

# === 自動預測，新增全域狀態 ===
_AUTO_PRED_THREAD = None
_AUTO_PRED_STOP = threading.Event()
_AUTO_PRED_LOCK = threading.Lock()

# === 自動預測，偵測是否有預測在跑 ===
def _is_predict_running() -> bool:
    with _PRED_LOCK:
        for v in _PRED_STATUS.values():
            if str(v.get("Status","")).startswith("預測中"):
                return True
    return False

# === 自動預測的啟停函式 ===
def _auto_pred_stop():
    _AUTO_PRED_STOP.set()
    t = globals().get("_AUTO_PRED_THREAD")
    if t and t.is_alive():
        t.join(timeout=5)
    with _AUTO_PRED_LOCK:
        globals()["_AUTO_PRED_THREAD"] = None
    return "自動預測已關閉。"

def _auto_pred_start(snapshot_args: dict, interval_min: int):
    if interval_min is None or int(interval_min) <= 0:
        return "⚠️ 間隔需為正整數分鐘。"
    with _AUTO_PRED_LOCK:
        t = globals().get("_AUTO_PRED_THREAD")
        if t and t.is_alive():
            return f"自動預測已啟動中（每 {interval_min} 分鐘）。"
        _AUTO_PRED_STOP.clear()

        def _loop():
            while not _AUTO_PRED_STOP.is_set():
                try:
                    if not _is_predict_running():
                        # 呼叫既有的預測入口（避免重疊）
                        _pred_start(**snapshot_args)
                    else:
                        print("DEBUG: 自動預測略過本輪（前一輪仍在進行）。")
                except Exception as e:
                    print(f"DEBUG: Auto-predict iteration failed: {e}")
                # 以 1 秒粒度等待，可即時關閉
                remain = int(interval_min) * 60
                while remain > 0 and (not _AUTO_PRED_STOP.is_set()):
                    time.sleep(1)
                    remain -= 1
            print("DEBUG: Auto-predict loop exited.")

        th = threading.Thread(target=_loop, daemon=True)
        globals()["_AUTO_PRED_THREAD"] = th
        th.start()
        return f"自動預測已啟動（每 {interval_min} 分鐘）。"


def _call_predict_store(store_id: str, horizon: int, version: str):
    """
    以 inspect 解析 predict_store 簽名，**自動帶入 params**（如後端需要），
    只走真實 LSTM（不再傳遞/使用模擬器參數），
    最後統一回傳：(pred_file, meta_file, used_version, stats: dict)
    """
    fn = predict_store
    sig = inspect.signature(fn)
    pnames = list(sig.parameters.keys())
    kwargs = {}
    used = []
    # 版本 key
    ver_keys = ["version", "model_version", "version_override", "ver", "v"]
    ver_key = next((k for k in ver_keys if k in pnames), None)
    # horizon key
    hz_keys = ["horizon", "steps"]
    hz_key = next((k for k in hz_keys if k in pnames), None)
    # params-like key
    params_keys = ["params", "param", "options", "option", "cfg", "config", "predict_params", "arguments", "args"]
    params_key = next((k for k in params_keys if k in pnames), None)
    # 先準備 params dict（給需要者）
    pdict = _build_predict_params(store_id, horizon, version)
    # 優先關鍵字呼叫
    if "store_id" in pnames:
        kwargs["store_id"] = store_id; used.append("store_id")
    if ver_key:
        kwargs[ver_key] = version; used.append(ver_key)
    if hz_key:
        kwargs[hz_key] = horizon; used.append(hz_key)
    if params_key:
        kwargs[params_key] = pdict; used.append(params_key)
    try:
        res = fn(store_id=store_id, horizon=horizon, version=version)
        sig_str = "standard_call"
    except Exception as e:
        raise e
    except (TypeError, ValueError) as e_kw:
        # 位置參數嘗試：涵蓋典型簽名
        trials = []
        if params_key:
            # (store_id, params) / (store_id, params, version) / ...
            trials += [
                ("pos:store_id,params",                [store_id, pdict]),
                ("pos:store_id,params,version",        [store_id, pdict, version]),
                ("pos:store_id,version,params",        [store_id, version, pdict]),
                ("pos:store_id,horizon,version,params",[store_id, horizon, version, pdict]),
            ]
        else:
            trials += [
                ("pos:store_id,version",               [store_id, version]),
                ("pos:store_id,horizon,version",       [store_id, horizon, version]),
                ("pos:store_id",                        [store_id]),
            ]
        res = None; sig_str = ""
        last_err = e_kw
        for sig_name, args in trials:
            try:
                res = fn(*args)
                sig_str = sig_name
                break
            except (TypeError, ValueError) as e_pos:
                last_err = e_pos
                continue
        if res is None:
            raise last_err
    # 統一解包
    pred_file = meta_file = ""
    used_version = version or ""
    stats = {}
    if isinstance(res, tuple):
        if len(res) == 4:
            pred_file, meta_file, used_version, stats = res
        elif len(res) == 3:
            pred_file, meta_file, used_version = res
        elif len(res) == 2:
            pred_file, meta_file = res
        elif len(res) == 1:
            pred_file = res[0]
    elif isinstance(res, dict):
        pred_file = res.get("pred_file", "")
        meta_file  = res.get("meta_file", "")
        used_version = res.get("version", used_version)
        stats = res.get("stats", {})
    if isinstance(stats, dict):
        stats.setdefault("_debug_signature", sig_str)
    return pred_file, meta_file, used_version, (stats or {})

def _read_config_selected() -> List[str]:
    """
    從 input/config.json 讀取 Selected_Store_IDs；若沒有，就用 All_Store_IDs；
    若仍沒有，最後以 index.json 的店別鍵名當作候選清單。
    """
    try:
        with open(os.path.join("input", "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
        sids = [s for s in (cfg.get("Selected_Store_IDs") or []) if isinstance(s, str)]
        if sids:
            return sids
        all_ids = [s for s in (cfg.get("All_Store_IDs") or []) if isinstance(s, str)]
        if all_ids:
            return all_ids
    except Exception:
        pass
    # 最後以 index.json 的 keys 當作候選
    try:
        idx = _load_index()
        return sorted([k for k in idx.keys() if isinstance(k, str)])
    except Exception:
        return []

def _load_index() -> Dict[str, Any]:
    """讀取 weighted_models/index.json，若不存在回傳 {}。"""
    try:
        if not os.path.exists(INDEX_PATH):
            return {}
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

    
def _build_predict_params(store_id: str, horizon: int, version: str) -> dict:
    """統一打包一份可傳給後端的 params dict。"""
    return {
        "store_id": store_id,
        "horizon": int(horizon),
        "version": version,
    }

def _version_key(v: str) -> int:
    """僅支援 vYYYYMMDD_N；其餘一律回 -1。"""
    if not v: 
        return -1
    m = _SEQ_VER_RE.match(str(v).strip())
    if not m:
        return -1
    ymd = int(m.group(1)); n = int(m.group(2))
    return ymd * 1000 + n

def _pick_latest_from_index(idx: dict, store_id: str) -> str:
    ent = (idx or {}).get(store_id, {}) or {}
    vers = [v.get("version") for v in (ent.get("versions") or []) if v.get("version")]
    if not vers:
        return ent.get("latest", "") or ""
    best = max(vers, key=_version_key)
    lat = ent.get("latest", "") or ""
    return lat if _version_key(lat) >= _version_key(best) else best

def _extract_vid(s: str) -> str:
    """只從字串中抽出 vYYYYMMDD_N；找不到就回原字串。"""
    if not s:
        return s
    m = re.search(r"v\d{8}_\d{1,3}", str(s))
    return m.group(0) if m else s

def _seq_display_for_version(idx: dict, store_id: str, version: str) -> str:
    """僅對 vYYYYMMDD_N 做顯示（同值直接回傳或照需求 0 起算顯示）。"""
    v = _extract_vid(version or "")
    m = _SEQ_VER_RE.match(v or "")
    if not m:
        return v or ""   # 舊格式 → 原樣回傳或標「不支援」
    # 直接使用 N（若 UI 想 00 起算，可在這裡 N-1 再格式化）
    ymd, n = m.group(1), int(m.group(2))
    return f"v{ymd}_{n:02d}"


def _vid_to_pretty(v: str) -> str:
    """
    將任意版本字串正規化為顯示用的 vYYYYMMDD_HH。
    - 若版本為 vYYYYMMDDHHMMSS → 取 HH → vYYYYMMDD_HH
    - 若版本為 vYYYYMMDD      → vYYYYMMDD_00
    - 若版本已是 vYYYYMMDD_HH → 原樣
    其他情形則原樣回傳。
    """
    s = (v or "").strip()
    m = re.match(r"^v(\d{8})(?:_(\d{2})|(\d{2})(\d{2})(\d{2}))?$", s)
    if not m:
        return s
    ymd = m.group(1)
    if m.group(2):           # vYYYYMMDD_HH
        hh = m.group(2)
    elif m.group(3):         # vYYYYMMDDHHMMSS
        hh = m.group(3)
    else:                    # vYYYYMMDD
        hh = "00"
    return f"v{ymd}_{hh}"

def _list_versions_for_store(idx: dict, store_id: str):
    ent = (idx or {}).get(store_id, {}) or {}
    return [v.get("version") for v in (ent.get("versions") or []) if v.get("version")]

'''def _collect_versions(selected: List[str]) -> List[str]:
    """蒐集目前所選門市的版本聯集，並以新→舊排序。"""
    idx = _load_index()
    pool = set()
    for sid in (selected or []):
        for v in _list_versions_for_store(idx, sid):
            if isinstance(v, str) and v.strip():
                pool.add(v.strip())
    return sorted(pool, key=_version_key, reverse=True) if pool else []'''

_PRED_STATUS = {}
_PRED_LOCK = threading.Lock()
_PRED_STOP = threading.Event()

def _pred_refresh_table(selected: List[str]) -> pd.DataFrame:
    # 讀取 _PRED_STATUS；若尚未有狀態，帶出 index.json 的 latest（顯示為 vYYYYMMDD_XX）
    idx0 = _load_index()
    rows = []
    with _PRED_LOCK:
        for sid in selected or []:
            latest = _pick_latest_from_index(idx0, sid)
            disp = _seq_display_for_version(idx0, sid, latest) if latest else ""
            row = _PRED_STATUS.get(sid, {
                "Store_ID": sid,
                "Model_Version": disp,
                "Status": "待機" if latest else "無可用模型",
                "Samples": "", "Horizon": "", "Progress(%)": 0,            
                "ŷ_mean": "", "ŷ_min": "", "ŷ_max": "",
                "Pred_File": "", "Meta_File": "", "_debug_signature": ""
            })
            rows.append(row)
    cols = ["Store_ID","Model_Version","Status","Samples","Horizon","Progress(%)",
            "ŷ_mean","ŷ_min","ŷ_max","Pred_File","Meta_File","_debug_signature"]
    return pd.DataFrame(rows, columns=cols)

def _pred_worker(store_id, horizon:int, version_override:str):
    try:
        # 版本選擇：優先用 UI 指定；否則讀 models/index.json 的 latest
        idx0 = _load_index()
        latest = _pick_latest_from_index(idx0, store_id)
        version_to_use = _extract_vid((version_override or "").strip() or latest)
        # 確認此版本存在；清單比對也用正規化後的鍵
        ent = (idx0 or {}).get(store_id, {}) or {}
        all_vers = {_extract_vid(v.get("version")) for v in (ent.get("versions") or []) if v.get("version")}
        if version_to_use and version_to_use not in all_vers:
            version_to_use = _extract_vid(_pick_latest_from_index(idx0, store_id))
        if not version_to_use:
            with _PRED_LOCK:
                _PRED_STATUS[store_id] = {
                    "Store_ID": store_id,
                    "Model_Version": _seq_display_for_version(idx0, store_id, version_to_use),
                    "Status": "預測中",
                    "Horizon": horizon, "Progress(%)": 0,
                    "Samples": "", "ŷ_mean": "", "ŷ_min": "", "ŷ_max": "",
                    "Pred_File": "", "Meta_File": "", "_debug_signature": ""
                }
            return
        # 預測開始時，先把 Model_Version 設為「要用版本的檔名」
        with _PRED_LOCK:
            _PRED_STATUS[store_id] = {
                "Store_ID": store_id,
                "Model_Version": _seq_display_for_version(idx0, store_id, version_to_use),
                "Status": "預測中",
                "Horizon": horizon, "Progress(%)": 0,
                "Samples": "", "ŷ_mean": "", "ŷ_min": "", "ŷ_max": "",
                "Pred_File": "", "Meta_File": "", "_debug_signature": ""
            }
        # 子執行緒實際呼叫 predict_store；本執行緒負責進度條
        done_evt = threading.Event()
        result_box = {}
        def _runner():
            try:
                res = _call_predict_store(store_id=store_id, horizon=horizon, version=version_to_use)
                result_box["res"] = res
                result_box["ok"] = True
            except Exception as e:
                result_box["err"] = e
                result_box["ok"] = False
            finally:
                done_evt.set()
        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        # 進度節拍：每 0.5s 增加，最多 95%，完成時設 100%
        progress = 0
        while not done_evt.is_set() and not _PRED_STOP.is_set():
            time.sleep(0.5)
            progress = min(progress + 5, 95)
            with _PRED_LOCK:
                if store_id in _PRED_STATUS:
                    _PRED_STATUS[store_id]["Progress(%)"] = progress
        # 停止被要求
        if _PRED_STOP.is_set() and not done_evt.is_set():
            with _PRED_LOCK:
                _PRED_STATUS[store_id].update({"Status": "已停止"})
            return
        # 等待子執行緒結束
        done_evt.wait()
        if not result_box.get("ok", False):
            raise result_box.get("err") or RuntimeError("predict failed")
        pred_file, meta_file, version, stats = result_box["res"]
        # 後端回傳版本也抽出 VID，再轉成 vYYYYMMDD_XX 顯示
        version = _extract_vid(version or "") or version_to_use
        disp = _seq_display_for_version(idx0, store_id, (version or version_to_use))
        with _PRED_LOCK:
            _PRED_STATUS[store_id].update({
                "Status": f"完成 {disp}",
                "Model_Version": disp,
                "Progress(%)": 100,
                "Samples": stats.get("samples",""),
                "ŷ_mean": stats.get("y_mean",""),
                "ŷ_min": stats.get("y_min",""),
                "ŷ_max": stats.get("y_max",""),
                "Pred_File": pred_file,
                "Meta_File": meta_file,
                "_debug_signature": stats.get("_debug_signature",""),
            })
    except Exception as e:
        with _PRED_LOCK:
            # 失敗時也把已知的版本（若有）寫回，避免 Model_Version 顯示 null
            idx0 = _load_index()
            latest = _pick_latest_from_index(idx0, store_id)
            _PRED_STATUS[store_id] = {
                "Store_ID": store_id,
                "Model_Version": _seq_display_for_version(idx0, store_id, latest) if latest else "",
                "Status": f"失敗：{e}",
                "Horizon": horizon, "Progress(%)": 0
            }
def _pred_start(selected: List[str], horizon: int, lookback: int = 120, batch_mode: bool = False, filter_quality: bool = True):
    _PRED_STOP.clear()
    started = 0
    # 先把 UI 狀態塞進去，讓使用者立即看到「預測中」
    with _PRED_LOCK:
        for sid in selected or []:
            _PRED_STATUS[sid] = {
                "Store_ID": sid, "Model_Version": "", "Status": "預測中",
                "Samples": "", "Horizon": horizon, "Progress(%)": 0,
                "ŷ_mean": "", "ŷ_min": "", "ŷ_max": "", "Pred_File": "", "Meta_File": "", "_debug_signature": ""
            }
    for sid in selected or []:
        # 不指定版本 → 交由 worker 依 index.json 取該店 latest
        t = threading.Thread(target=_pred_worker, args=(sid, horizon, ""), daemon=True)
        t.start(); started += 1
    return "已開始預測。" if started else "未選擇門市。"

def _list_versions_for_store(idx: dict, store_id: str) -> List[str]:
    ent = (idx or {}).get(store_id, {}) or {}
    vs = [v.get("version") for v in (ent.get("versions") or []) if v.get("version")]
    return [v for v in vs if isinstance(v, str) and v.strip()]

'''def _collect_versions(selected: List[str]) -> List[str]:
    """蒐集目前所選門市的版本聯集，並以新→舊排序。"""
    idx = _load_index()
    pool = set()
    for sid in (selected or []):
        for v in _list_versions_for_store(idx, sid):
            pool.add(v)
    if not pool:
        return []
    return sorted(pool, key=_version_key, reverse=True)'''

def _pred_stop_all():
    _PRED_STOP.set()
    return "已要求停止（對正在預測中的門市，下一輪檢查時生效）。"

def render_panel():
    gr.Markdown("### 預測與分析")
    selected_default = _read_config_selected()
    with gr.Row():
        store_choices = gr.CheckboxGroup(
            label="要預測的門市（預設來自 input/config.json 的 Selected_Store_IDs）",
            choices=selected_default, value=selected_default
        )
        with gr.Column(scale=0):
            refresh_btn = gr.Button("重新整理", variant="secondary")
            select_all_btn = gr.Button("全選")
            clear_all_btn = gr.Button("全不選")
    with gr.Row():
        horizon = gr.Slider(minimum=6, maximum=96, step=1, value=12, label="預測範圍（horizon）")
    # 添加新參數
    with gr.Row():
        lookback_window = gr.Number(label="觀測窗口（window）", value=120, precision=0, 
                                info="用於預測的歷史數據長度")
        batch_mode = gr.Checkbox(label="批次預測模式", value=False, 
                                info="同時處理所有設備，提升效率")

    with gr.Accordion("進階預測參數", open=False):
        with gr.Row():
            confidence_level = gr.Slider(label="信心水準", minimum=0.8, maximum=0.99, 
                                    step=0.01, value=0.95, info="預測結果的可信度")
            filter_quality = gr.Checkbox(label="過濾低品質預測", value=True,
                                        info="排除除霜中或數據異常的設備")
        with gr.Row():
            export_format = gr.Dropdown(label="輸出格式", 
                                    choices=["CSV", "JSON", "Excel"], value="CSV")
            include_metadata = gr.Checkbox(label="包含詳細資訊", value=True,
                                        info="輸出檔案包含設備類型、安全閾值等")
    with gr.Row():
        start_btn = gr.Button("開始預測", variant="primary")
            # === 自動預測：新增（置於兩顆按鈕之間） ===
        with gr.Column(scale=0):
            auto_pred_toggle = gr.Checkbox(label="自動預測", value=False)
            auto_pred_interval = gr.Number(label="間隔（分鐘）", value=60, precision=0)

        stop_btn = gr.Button("停止全部", variant="stop")

    warn_box = gr.Markdown("")
    df0 = _pred_refresh_table(selected_default)
    pred_df = gr.Dataframe(
        headers=["Store_ID","Model_Version","Status","Samples","Horizon","Progress(%)","ŷ_mean","ŷ_min","ŷ_max","Pred_File","Meta_File","_debug_signature"],
        col_count=(12,"fixed"), row_count=(0,"dynamic"),
        wrap=True, interactive=False, label="預測狀態",
        value=df0
    )
    with gr.Row():
        export_btn = gr.Button("匯出預測結果", variant="secondary")
        download_file = gr.File(label="下載檔案", visible=False)

    def export_predictions(format_val, include_meta):
        # 收集所有完成的預測結果
        # 根據format_val生成對應格式檔案
        return gr.update(visible=True, value="predictions.csv")

    export_btn.click(
        export_predictions, 
        inputs=[export_format, include_metadata], 
        outputs=[download_file]
    )
    timer = gr.Timer(1.5, active=True)
    timer.tick(fn=_pred_refresh_table, inputs=[store_choices], outputs=[pred_df])

    # 開始/停止
    start_btn.click(fn=_pred_start, inputs=[store_choices, horizon], outputs=[warn_box])
    def _stop_all_with_auto_pred():
        msg_auto = _auto_pred_stop()
        msg_pred = _pred_stop_all()
        return f"{msg_auto}\n{msg_pred}"
    
    stop_btn.click(fn=_stop_all_with_auto_pred, inputs=[], outputs=[warn_box])

    
    # === 快照目前 UI 參數 → 啟停自動預測 ===
    def _on_auto_pred_toggle(is_on, sids, hz, lookback_win, batch, filt, interval_min):
        if not is_on:
            return "**狀態：** " + _auto_pred_stop()

        snapshot = dict(
            selected=sids or [],
            horizon=int(hz),
            lookback=int(lookback_win or 120),
            batch_mode=bool(batch),
            filter_quality=bool(filt),
        )
        msg = _auto_pred_start(snapshot, int(interval_min or 60))
        return f"**狀態：** {msg}"

    auto_pred_toggle.change(
        _on_auto_pred_toggle,
        inputs=[auto_pred_toggle, store_choices, horizon, lookback_window, batch_mode, filter_quality, auto_pred_interval],
        outputs=[warn_box]
    )


    # 重新整理：從 config.json 讀取、更新複選與下方表
    def _on_refresh():
        sids = _read_config_selected() or []
        df = _pred_refresh_table(sids)
        return gr.update(choices=sids, value=sids), df
    refresh_btn.click(_on_refresh, [], [store_choices, pred_df])

    # 全選 / 全不選
    def _on_select_all():
        sids = _read_config_selected() or []
        df = _pred_refresh_table(sids)
        return gr.update(value=sids), df
    select_all_btn.click(_on_select_all, [], [store_choices, pred_df])

    def _on_clear_all():
        df = _pred_refresh_table([])
        return gr.update(value=[]), df
    clear_all_btn.click(_on_clear_all, [], [store_choices, pred_df])