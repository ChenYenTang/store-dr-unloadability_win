import os, json, math
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import numpy as np

# ===== 路徑常數（可用環境變數覆寫） =====
WEIGHT_DIR = os.getenv("WEIGHT_DIR", "weighted_models")
INDEX_PATH = os.path.join(WEIGHT_DIR, "index.json")

# ========= 寫檔工具 =========
def _atomic_write_json(path: str, data: Any) -> None:
    """以 .tmp + os.replace() 原子寫入 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _now_verstamp() -> str:
    """时间戳（檔名尾碼用）：YYYYMMDDHHMMSS"""
    return datetime.now().strftime("%Y%m%d%H%M%S")

def _now_iso() -> str:
    """ISO 字串（不含時區，秒級）。"""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# ========= 讀取 index / 版本 =========
def _load_index(index_path: str = INDEX_PATH) -> Dict[str, Any]:
    if not os.path.exists(index_path):
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_version(idx: Dict[str, Any], store_id: str, version_override: Optional[str]) -> Optional[str]:
    if store_id not in idx:
        return None
    if version_override:
        # 驗證版本存在於該店
        versions = {v["version"] for v in idx[store_id].get("versions", [])}
        return version_override if version_override in versions else None
    return idx[store_id].get("latest")

def _get_metrics_params(idx: Dict[str, Any], store_id: str, version: str) -> Dict[str, Any]:
    """回傳該版本記錄裡的 params（若找不到則空 dict）。"""
    for rec in idx.get(store_id, {}).get("versions", []):
        if rec.get("version") == version:
            # metrics 檔案裡也有 params，但讀 index 內的快；保持一致
            pdict = rec.get("params", {}) or {}
            # ★ 若索引有 model_type，也帶出來（後續載模可用）
            if rec.get("model_type") and "model_type" not in pdict:
                pdict["model_type"] = rec["model_type"]
            return pdict
    return {}

# ========= 讀資料（latest + history 回補） =========
CSV_SCHEMA = ["ts","store_id","device_id","device_type","temp_current","t_room","defrost_status","power","ingest_ts"]

def _read_latest_json(store_id: str) -> pd.DataFrame:
    path = os.path.join("input", "latest", f"{store_id}.json")
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_SCHEMA)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # 允許單物件或 {"data":[...]}
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            data = [data]
    df = pd.DataFrame(data)
    for col in CSV_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    return df[CSV_SCHEMA]

def _read_history_days(dates: List[str]) -> pd.DataFrame:
    frames = []
    for d in dates:
        p = os.path.join("input", "history", f"{d}.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=CSV_SCHEMA)
    df = pd.concat(frames, ignore_index=True)
    for col in CSV_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    return df[CSV_SCHEMA]

def _ensure_ts_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["ts"])
        except Exception:
            # 若轉不動，就當字串排序
            pass
        df = df.sort_values("ts")
    return df

def _infer_freq_seconds(df: pd.DataFrame, default_sec: int = 300) -> int:
    """由 ts 推估頻率（秒）。若不足兩筆或無法判斷，用 default。"""
    try:
        s = pd.to_datetime(df["ts"]).sort_values().reset_index(drop=True)
        if len(s) < 2:
            return default_sec
        deltas = (s.diff().dropna()).dt.total_seconds().astype(int)
        if len(deltas) == 0:
            return default_sec
        # 取眾數
        return int(deltas.value_counts().idxmax())
    except Exception:
        return default_sec

def _assemble_recent(store_id: str, window_hours: int, lookback: int) -> Tuple[pd.DataFrame, int]:
    """
    回傳 (df_recent, freq_sec)。優先 latest，不足則回補 history。
    """
    df_latest = _read_latest_json(store_id)
    df_latest = df_latest[df_latest["store_id"] == store_id] if "store_id" in df_latest else df_latest
    df_latest = _ensure_ts_sorted(df_latest)

    freq_sec = _infer_freq_seconds(df_latest)
    # 估需要的樣本數：lookback + window（保守多抓一些）
    need_rows = max(lookback * 2, int(window_hours * 3600 / max(1, freq_sec)))

    if len(df_latest) >= need_rows:
        df = df_latest.tail(need_rows).copy()
        return df, freq_sec

    # 回補 history：以 latest 最後 ts 為基準往回 N 天（上限 14 天）
    last_ts = None
    if len(df_latest) > 0:
        try:
            last_ts = pd.to_datetime(df_latest["ts"].iloc[-1])
        except Exception:
            last_ts = None
    # 若 latest 為空或沒有 ts，就用今天當基準向前天數
    base_date = (last_ts.date() if last_ts is not None else datetime.now().date())
    days = []
    for k in range(0, 14):  # 最多往回 14 天
        d = (base_date - timedelta(days=k)).strftime("%Y%m%d")
        days.append(d)
    df_hist = _read_history_days(days)
    if "store_id" in df_hist:
        df_hist = df_hist[df_hist["store_id"] == store_id]
    df_hist = _ensure_ts_sorted(df_hist)

    df = pd.concat([df_hist, df_latest], ignore_index=True)
    df = _ensure_ts_sorted(df)
    if len(df) > need_rows:
        df = df.tail(need_rows).copy()
    # 重新推一次頻率更準
    freq_sec = _infer_freq_seconds(df, default_sec=freq_sec or 300)
    return df, freq_sec

# ========= 生成預測（模擬器） =========
def _simulate_predictions(store_id: str, version: str, df_recent: pd.DataFrame,
                        freq_sec: int, horizon: int) -> List[Dict[str, Any]]:
    """
    以近期 power 的移動平均 + 小擾動產生 y_hat，提供 lo/hi band，可重現。
    """
    # 建 seed：店 + 版本 + 最近 ts + horizon
    last_ts_str = str(df_recent["ts"].iloc[-1]) if len(df_recent) else "NA"
    seed = abs(hash(f"{store_id}|{version}|{last_ts_str}|{horizon}")) % (2**32)
    rng = np.random.default_rng(seed)

    # 最近 power 的平滑：取最後 min(lookback, 96) 筆的移動平均做基準
    power = pd.to_numeric(df_recent.get("power", pd.Series(dtype=float)), errors="coerce").fillna(method="ffill")
    if len(power) == 0:
        base = 0.0
    else:
        k = min(max(8, int(len(power) * 0.25)), 96)  # 窗口大小
        base = float(pd.Series(power).tail(k).mean())

    # 生成 horizon 個點
    if len(df_recent) and "ts" in df_recent.columns:
        try:
            last_ts = pd.to_datetime(df_recent["ts"].iloc[-1])
        except Exception:
            last_ts = datetime.now()
    else:
        last_ts = datetime.now()

    preds = []
    step = timedelta(seconds=max(60, int(freq_sec or 300)))
    for i in range(1, horizon + 1):
        ts_i = last_ts + step * i
        # y_hat：基準 + 小幅起伏（±3% 內），附帶一點白噪
        undulation = 0.03 * base * math.sin(2 * math.pi * (i / max(6, horizon)))
        noise = float(rng.normal(0, 0.01 * (abs(base) + 1.0)))
        y_hat = max(0.0, base + undulation + noise)
        band = float(0.1 + 0.05 * rng.random())  # 10~15%
        lo = max(0.0, y_hat * (1 - band))
        hi = y_hat * (1 + band)
        preds.append({
            "ts": ts_i.strftime("%Y-%m-%dT%H:%M:%S"),
            "store_id": store_id,
            "y_hat": round(y_hat, 6),
            "y_hat_lo": round(lo, 6),
            "y_hat_hi": round(hi, 6),
        })
    return preds

# ========= 對外主函式 =========
def predict_store(
    store_id: str,
    model_version: Optional[str],
    df_recent: pd.DataFrame,
    params: Dict[str, Any],
    simulate: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    回傳:
    pred_path: output/predictions/{sid}/pred_{sid}_{version}_{stamp}.json
    meta_path: output/predictions/{sid}/meta_{sid}_{version}_{stamp}.json
    meta:      供 UI 表格摘要用
    """
    idx = _load_index()
    version = _resolve_version(idx, store_id, model_version)
    if not version:
        raise ValueError(f"no model version found for store {store_id}")
    # 推理設定
    horizon = int(params.get("horizon", 12))
    lookback = int(params.get("lookback", 48))  # 允許 UI 傳入，若沒有則從 metrics 補
    if "lookback" not in params:
        mparams = _get_metrics_params(idx, store_id, version)
        lookback = int(mparams.get("lookback", lookback))

    # 準備 recent 資料 & 頻率
    df_recent = _ensure_ts_sorted(df_recent)
    freq_sec = _infer_freq_seconds(df_recent)
    if len(df_recent) < lookback:
        # fallback：往回補
        df_recent2, freq_sec2 = _assemble_recent(store_id, window_hours=24, lookback=lookback)
        if len(df_recent2) > len(df_recent):
            df_recent, freq_sec = df_recent2, freq_sec2

    # 生成預測
    if simulate:
        preds = _simulate_predictions(store_id, version, df_recent, freq_sec, horizon)
    else:
        # TODO: 真實推理（載入 .pt、切窗、前處理、forward）
        preds = _simulate_predictions(store_id, version, df_recent, freq_sec, horizon)

    # 路徑與落檔
    stamp = _now_verstamp()
    out_dir = os.path.join("output", "predictions", store_id)
    pred_path = os.path.join(out_dir, f"pred_{store_id}_{version}_{stamp}.json")
    meta_path = os.path.join(out_dir, f"meta_{store_id}_{version}_{stamp}.json")

    # 摘要
    y = [p["y_hat"] for p in preds]
    meta = {
        "store_id": store_id,
        "model_version": version,
        "generated_at": _now_iso(),
        "horizon": horizon,
        "pred_summary": {
            "mean": round(float(np.mean(y)) if y else 0.0, 6),
            "min": round(float(np.min(y)) if y else 0.0, 6),
            "max": round(float(np.max(y)) if y else 0.0, 6),
        },
    }
    _atomic_write_json(pred_path, preds)
    _atomic_write_json(meta_path, meta)

    # 更新 priority.json
    _update_priority_json(store_id, version, meta, pred_path, meta_path)
    return pred_path, meta_path, meta

# ========= priority.json =========
def _z_to_priority(z: float) -> float:
    """簡單把 z 映射到 0~1；此處先做 sigmoid 近似。"""
    return float(1 / (1 + math.exp(-z)))

def _update_priority_json(store_id: str, version: str, meta: Dict[str, Any],
                        pred_path: str, meta_path: str,
                        priority_path: str = "output/priority.json") -> None:
    os.makedirs(os.path.dirname(priority_path), exist_ok=True)
    if os.path.exists(priority_path):
        with open(priority_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {"generated_at": _now_iso(), "items": []}

    # 以簡單規則算 priority： (max - mean) / (std + 1e-6)
    mean_v = meta.get("pred_summary", {}).get("mean", 0.0)
    max_v = meta.get("pred_summary", {}).get("max", 0.0)
    min_v = meta.get("pred_summary", {}).get("min", 0.0)
    std = max(1e-6, (max(0.0, max_v - min_v)) / 3.0)
    z = (max_v - mean_v) / std
    pr = round(_z_to_priority(z), 6)

    # 先移除舊項（相同 store_id）
    items = [it for it in doc.get("items", []) if it.get("store_id") != store_id]
    items.append({
        "store_id": store_id,
        "model_version": version,
        "horizon": meta.get("horizon"),
        "latest_ts": meta.get("generated_at"),
        "pred_summary": meta.get("pred_summary"),
        "pred_file": pred_path,
        "meta_file": meta_path,
        "priority": pr
    })
    doc["generated_at"] = _now_iso()
    doc["items"] = items
    _atomic_write_json(priority_path, doc)
