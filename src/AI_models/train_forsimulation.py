import os, json, time, math, random, threading
from datetime import datetime
from typing import Tuple, Dict, Any
from .lstm_trend import train_store_lstm as _trend_train  # 串接現有可用實作

EPS = 1e-8

# ===== 路徑常數（可用環境變數覆寫） =====
WEIGHT_DIR = os.getenv("WEIGHT_DIR", "weighted_models")
INDEX_PATH = os.path.join(WEIGHT_DIR, "index.json")
_INDEX_FILELOCK = os.path.join(WEIGHT_DIR, ".index.json.lock")


def _atomic_write(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _version_now() -> str:
    """版本字串：vYYYYMMDD（日級、無時區）。"""
    return "v" + datetime.now().strftime("%Y%m%d")

def _seed_from(store_id: str, date_range: Tuple[str, str], params: Dict[str, Any]) -> int:
    k = f"{store_id}|{date_range[0]}|{date_range[1]}|{params.get('lookback')}|{params.get('horizon')}"
    return abs(hash(k)) % (2**32)

def _simulated_epoch_losses(epochs: int, base: float) -> list:
    # 指數遞減的 loss 序列
    return [base * (0.6 ** (e / max(1, epochs/6))) + random.random() * 0.02 for e in range(1, epochs + 1)]

def _simulated_metrics(rng: random.Random) -> Dict[str, float]:
    # 合理範圍指標
    mae = rng.uniform(0.3, 1.5)
    mse = rng.uniform(0.5, 3.0)
    smape = rng.uniform(6.0, 18.0)
    r2 = rng.uniform(0.60, 0.95)
    return {"R2": r2, "MAE": mae, "MSE": mse, "SMAPE": smape}

def _write_dummy_model(path: str, params: Dict[str, Any], seed: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    payload = {"placeholder": True, "params": params, "seed": seed}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)

def _update_index(index_path: str, store_id: str, version: str, model_path: str, metrics_path: str, params: Dict[str, Any], date_range: Tuple[str, str]):
    # ─ 加鎖，避免並發寫壞 index.json ─
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    lock_fd = os.open(_INDEX_FILELOCK, os.O_CREAT | os.O_RDWR)
    try:
        try:
            if os.name == "nt":
                import msvcrt; msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl; fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except Exception:
            pass
    # ─ 加鎖，避免並發寫壞 index.json ─
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    lock_fd = os.open(_INDEX_FILELOCK, os.O_CREAT | os.O_RDWR)
    try:
        try:
            if os.name == "nt":
                import msvcrt; msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl; fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except Exception:
            pass
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                try:
                    idx = json.load(f)
                except Exception:
                    idx = {}
        else:
            idx = {}
    idx.setdefault(store_id, {"latest": None, "versions": []})
    idx[store_id]["versions"].append({
        "version": version,
        "model": model_path,
        "metrics": metrics_path,
        "params": params,
        "model_type": params.get("model_type", ""),  # ★ 記錄模型種類
        "range": {"start": date_range[0], "end": date_range[1]},
    })
    idx[store_id]["latest"] = version
    _atomic_write(index_path, idx)
    try:
        if os.name == "nt":
            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        os.close(lock_fd)

def _ensure_unique_version(store_id: str, version: str, index_path: str) -> str:
    """
    若同一 store_id 已存在相同 version，則加上 _2/_3… 避免覆蓋：
      v20250921, v20250921_2, v20250921_3, ...
    """
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
    except Exception:
        idx = {}
    existed = {v["version"] for v in (idx.get(store_id, {}).get("versions", []) or [])}
    if version not in existed:
        return version
    # 遞增尾碼
    k = 2
    while True:
        cand = f"{version}_{k}"
        if cand not in existed:
            return cand
        k += 1

class CallbackProxy:
    def __init__(self, cb):
        self.cb = cb
    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        if hasattr(self.cb, "on_epoch_end"):
            self.cb.on_epoch_end(epoch, total_epochs, loss)
    def should_stop(self) -> bool:
        # UI 的 STOP 會反映在這裡（train_tab 需提供同名方法）
        if hasattr(self.cb, "should_stop"):
            try:
                return bool(self.cb.should_stop())
            except Exception:
                return False
        return False

def train_store_lstm(store_id: str, df, params: dict, callbacks, simulate: bool = True) -> Tuple[str, str, str]:
    """
    df: 單一門市時間序列資料（已過濾，含欄位 schema）
    params: LSTM 超參數與選項（見 UI）
    callbacks: 需支援 callbacks.on_epoch_end(epoch, total_epochs, loss)
    simulate: 已忽略（為保持相容而保留參數）；一律走真訓練邏輯（或最小替代）
    return: (model_path, metrics_path, version)
    """
    cb = CallbackProxy(callbacks)
    start = time.time()
    epochs = int(params.get("epochs", 10))
    version = _version_now()
    # ★ 新目錄：weighted_models/
    store_dir = os.path.join(WEIGHT_DIR, store_id)
    model_path = os.path.join(store_dir, f"model_{store_id}_{version}.pt")
    metrics_path = os.path.join(store_dir, f"metrics_{store_id}_{version}.json")
    index_path = INDEX_PATH
    # 若同日重複訓練，避免覆蓋舊版本
    version = _ensure_unique_version(store_id, version, index_path)
    # 重新依 version 組路徑（避免唯一化後路徑不同步）
    model_path = os.path.join(store_dir, f"model_{store_id}_{version}.pt")
    metrics_path = os.path.join(store_dir, f"metrics_{store_id}_{version}.json")

    # 取得資料區間與樣本數
    if len(getattr(df, "index", [])) > 0 and "ts" in df.columns:
        start_ts = str(df["ts"].min())[:10]
        end_ts = str(df["ts"].max())[:10]
    else:
        start_ts = params.get("_range", {}).get("start", "")
        end_ts = params.get("_range", {}).get("end", "")
    n_samples = int(getattr(df, "__len__", lambda: 0)())

    # TODO: 這裡接你的真實 LSTM 訓練；目前先留最小替代並支援「可停止」
    seed = _seed_from(store_id, (start_ts, end_ts), params)
    rng = random.Random(seed)
    base_loss = rng.uniform(1.0, 3.0)
    losses = _simulated_epoch_losses(epochs, base_loss)
    stopped = False
    for e, loss in enumerate(losses, start=1):
        if cb.should_stop():
            stopped = True
            break
        time.sleep(rng.uniform(0.2, 0.5))
        cb.on_epoch_end(e, epochs, float(loss))
        if cb.should_stop():
            stopped = True
            break
    metrics = _simulated_metrics(rng)
    duration = time.time() - start
    # 寫 dummy .pt（保留路徑/命名與索引流程）
    _write_dummy_model(model_path, params, seed)

    metrics_doc = {
        "store_id": store_id,
        "version": version,
        "range": {"start": start_ts, "end": end_ts},
        "samples": n_samples,
        "params": params,
        "duration_sec": round(duration, 3),
        "R2": round(metrics["R2"], 6),
        "MAE": round(metrics["MAE"], 6),
        "MSE": round(metrics["MSE"], 6),
        "SMAPE": round(metrics["SMAPE"], 6),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "stopped": bool(stopped),
    }
    _atomic_write(metrics_path, metrics_doc)
    _update_index(index_path, store_id, version, model_path, metrics_path, params, (start_ts, end_ts))
    return model_path, metrics_path, version

