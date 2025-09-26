from __future__ import annotations

import os
import io
import json
import math
import time
import shutil
import random
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple, Any


import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # 避免匯入失敗中斷（例如僅做靜態檢視）
    torch = None
    nn = object  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore

# =====================================================
# 常數與工具
# =====================================================
SAFE_FRIDGE = 7.0       # 冷藏安全溫度
SAFE_FREEZER = -18.0    # 冷凍安全溫度
SAFE_OPEN_CASE = 7.0    # 開放櫃安全溫度

NEAR_LIMIT_FRIDGE = 6.5
NEAR_LIMIT_FREEZER = -17.5
NEAR_LIMIT_OPEN_CASE = 6.5  # 新增：與冷藏相同

DEFAULT_WINDOW = 120  # 觀測視窗（分鐘）
DEFAULT_STEP = 1      # 窗格步進（分鐘）
DEFAULT_HORIZON = 60  # TTW 目標預測範圍（分鐘）

INDEX_PATH = os.path.join("weighted_models", "index.json")


# --------------------- 小工具 ---------------------
def _next_seq_version(store_id: str) -> str:
    """
    產生「日期_序號」版號：vYYYYMMDD_N
    序號依 index.json 內相同 store、相同日期的既有版本自動 +1。
    若當日未有任何版本，從 1 起算。
    """
    today = datetime.now().strftime("%Y%m%d")
    idx = _read_json(INDEX_PATH, default={})
    node = idx.get(store_id, {})
    seq_max = 0
    for v in node.get("versions", []):
        ver = str(v.get("version", ""))
        m = re.match(rf"^v{today}(?:_(\d+))?$", ver)
        if m:
            # 若剛好有舊式 vYYYYMMDD（無序號），視為 _1
            s = int(m.group(1) or 1)
            if s > seq_max:
                seq_max = s
    return f"v{today}_{seq_max + 1}"


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _read_json(p: str, default: Optional[dict] = None) -> dict:
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def _write_json(p: str, obj: dict):
    _ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _standardize_model_type(s: str) -> str:
    s = (s or "").strip()
    if s.lower() in {"trend", "lstm_trend"}:
        return "LSTM_trend"
    return s or "LSTM_trend"


def _temp_safe_by_type(device_type: str, dynamic_temps: dict = None) -> float:
    """根據設備類型返回安全溫度，支援動態參數覆蓋"""
    safe_fridge = dynamic_temps.get("SAFE_FRIDGE", SAFE_FRIDGE) if dynamic_temps else SAFE_FRIDGE
    safe_freezer = dynamic_temps.get("SAFE_FREEZER", SAFE_FREEZER) if dynamic_temps else SAFE_FREEZER
    safe_open_case = dynamic_temps.get("SAFE_OPEN_CASE", SAFE_OPEN_CASE) if dynamic_temps else SAFE_OPEN_CASE
    
    if device_type is None:
        return safe_open_case
    t = str(device_type)
    if "凍" in t or "freez" in t.lower():
        return safe_freezer
    elif "開放" in t or "open" in t.lower():
        return safe_open_case
    elif "空調" in t:
        return safe_open_case
    else:
        return safe_fridge


def _safety_buffer_by_type(device_type: str, safety_buffers: dict = None) -> int:
    """根據設備類型返回安全緩衝時間，支援動態參數覆蓋"""
    freezer_buffer = safety_buffers.get("freezer", 3) if safety_buffers else 3
    other_buffer = safety_buffers.get("other", 2) if safety_buffers else 2
    
    if device_type is None:
        return other_buffer
    t = str(device_type)
    if "凍" in t or "freez" in t.lower():
        return freezer_buffer
    else:
        return other_buffer


def _near_limit_by_type(device_type: str, dynamic_temps: dict = None) -> float:
    """根據設備類型返回近臨界值，支援動態參數覆蓋"""
    near_freezer = dynamic_temps.get("NEAR_LIMIT_FREEZER", NEAR_LIMIT_FREEZER) if dynamic_temps else NEAR_LIMIT_FREEZER
    near_fridge = dynamic_temps.get("NEAR_LIMIT_FRIDGE", NEAR_LIMIT_FRIDGE) if dynamic_temps else NEAR_LIMIT_FRIDGE
    near_open_case = dynamic_temps.get("NEAR_LIMIT_OPEN_CASE", NEAR_LIMIT_OPEN_CASE) if dynamic_temps else NEAR_LIMIT_OPEN_CASE
    
    if device_type is None:
        return near_open_case
    t = str(device_type)
    if "凍" in t or "freez" in t.lower():
        return near_freezer
    elif "開放" in t or "open" in t.lower():
        return near_open_case
    else:
        return near_fridge


def _is_aircon(device_type: str) -> bool:
    return device_type is not None and ("空調" in str(device_type))


# =====================================================
# 資料前處理與 Dataset
# =====================================================
@dataclass
class InputSpec:
    features: List[str]
    window: int = DEFAULT_WINDOW
    step: int = DEFAULT_STEP


class SeqSurvivalDataset(Dataset):
    """將每分鐘對齊的序列轉為 (X, hazard-label) 訓練樣本。

    - X: shape (seq_len, feat_dim)
    - y_hazard: shape (H,) 以離散時間 hazard（每分鐘一 bin）
    - mask: shape (H,) 0/1，控制截尾情形下的有效 loss 區間

    目標定義：
      對每個 t0，找到最小 Δt (1..H) 使得 Temp(t0+Δt) ≥ T_safe。
      若於 H 分內未越線 → 右截尾（mask 對所有 1..H 有效，且 y_hazard 全 0）。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        device_type_map: Dict[str, str],
        window: int = DEFAULT_WINDOW,
        horizon: int = DEFAULT_HORIZON,
        step: int = DEFAULT_STEP,
        dynamic_temps: dict = None,
        safety_buffers: dict = None,
    ):

        assert "ts" in df.columns and "device_id" in df.columns
        self.features = features
        self.window = window
        self.horizon = horizon
        self.step = step
        self.device_type_map = device_type_map
        self.dynamic_temps = dynamic_temps or {}
        self.safety_buffers = safety_buffers or {}

        # 以每分鐘對齊；假設 df.ts 已為 datetime64[ns]
        self.df = df.sort_values(["device_id", "ts"]).reset_index(drop=True)
        self.samples: List[Tuple[int, int]] = []  # (start_idx, device block end)

        # 建索引（每台設備獨立切窗）
        for dev, g in self.df.groupby("device_id", sort=False):
            idxs = g.index.to_numpy()
            # 可切到 (len- (window + horizon)) 的最後一點當起點
            for i in range(0, len(idxs) - (window + horizon) + 1, step):
                self.samples.append((idxs[i], idxs[i + window + horizon - 1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s, e = self.samples[i]
        rows = self.df.iloc[s : s + self.window + self.horizon]
        xwin = rows.iloc[: self.window].copy()
        future = rows.iloc[self.window : self.window + self.horizon].copy()

        device_id = str(xwin["device_id"].iloc[-1])
        device_type = self.device_type_map.get(device_id, "冷藏")
        t_safe = _temp_safe_by_type(device_type, self.dynamic_temps)

        # ☆ 重要：特徵補值，避免 NaN 傳入 LSTM → BCE 報「input should be between 0 and 1」
        fx = xwin[self.features].astype("float32")
        fx = fx.ffill().bfill().fillna(0.0)
        X = fx.to_numpy(dtype=np.float32)

        # 目標：找到第一個 >= t_safe 的時間（1..H）；若無 → censored
        # 目標使用的溫度也做基本補值，避免 NaN 造成邏輯判斷異常
        ft = future["temp_current"].astype("float32").ffill().bfill().fillna(-1e9)
        fut_temp = ft.to_numpy(dtype=np.float32)
        cross_idx = None
        for k, v in enumerate(fut_temp, start=1):  # 1..H
            if v >= t_safe:
                cross_idx = k
                break

        H = self.horizon
        y = np.zeros(H, dtype=np.float32)     # hazard label
        m = np.zeros(H, dtype=np.float32)     # mask
        if cross_idx is None:
            # 右截尾：對 1..H 皆計入存活對數似然（標記 0，mask=1）
            m[:] = 1.0
        else:
            # 事件於 k：對 1..k-1 計存活、k 計事件
            m[:cross_idx] = 1.0
            y[cross_idx - 1] = 1.0

        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(m),
            device_id,
            device_type,
        )


# =====================================================
# 模型
# =====================================================
class HazardLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=0.0 if layers == 1 else dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, horizon)
        )

    def forward(self, x):  # x: (B, T, D)
        o, _ = self.lstm(x)
        h = o[:, -1, :]  # 使用最後一步隱狀態
        logits = self.head(h)  # (B, H)
        return logits


def _hazard_nll(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """離散時間存活 NLL。
    logits: (B,H) → sigmoid 為每分鐘 hazard p_k。
    y: 事件 one-hot 於第 k 分鐘（若截尾則全 0）
    m: 有效區間 mask（事件於 k → m[0..k-1]=1, m[k]=1；截尾 → m[0..H-1]=1）
    
    NLL =  - [ 事件部份:  y_k * log p_k  + 存活部份: sum_{j<m_k} (1-y_j) * log(1-p_j) ]
    這裡以逐分鐘 BCE 近似計算，僅在 m==1 的 bin 計入 loss。
    """
    p = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy(p, y, reduction="none")
    # 對所有有效 bin：若 y=0 → 相當於鼓勵 (1-p) 存活；若某 bin y=1 → 該 bin 走事件 loss
    loss = (bce * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
    return loss.mean()


def _hazard_to_quantiles(logits: torch.Tensor, qs: List[float]) -> torch.Tensor:
    """將 (B,H) hazard logits 轉為 TTW 分位（分鐘，整數）。
    若在 H 內不越線，分位以 H 回傳（代表保守視為>=H）。
    回傳 shape: (B, len(qs))
    """
    with torch.no_grad():
        p = torch.sigmoid(logits)  # (B,H)
        B, H = p.shape
        one = torch.ones((B,), device=p.device)
        S = []
        s = one
        for k in range(H):
            s = s * (1.0 - p[:, k])  # 累積存活至第 k 分鐘
            S.append(s.clone())
        S = torch.stack(S, dim=1)  # (B,H)
        CDF = 1.0 - S  # (B,H)
        out = []
        for q in qs:
            # 找到最小 k 使 CDF>=q
            mask = (CDF >= q)
            idx = torch.where(mask.any(dim=1), mask.float().argmax(dim=1) + 1, torch.full((B,), H, device=p.device))
            out.append(idx)
        return torch.stack(out, dim=1).float()  # (B, Q)


# =====================================================
# LSTMTrend 類別（共通介面）
# =====================================================
class LSTMTrend:
    def __init__(self, params: Dict[str, Any]):
        self.params = dict(params or {})
        self.params["model_type"] = _standardize_model_type(self.params.get("model_type", "LSTM_trend"))

        # 基本超參數
        self.features: List[str] = self.params.get("features", [
            "temp_current", "t_room", "defrost_status", "power"
        ])
        self.window: int = int(self.params.get("window", DEFAULT_WINDOW))
        self.step: int = int(self.params.get("step", DEFAULT_STEP))
        self.horizon: int = int(self.params.get("horizon", DEFAULT_HORIZON))
        self.hidden: int = int(self.params.get("hidden", 64))
        self.layers: int = int(self.params.get("layers", 2))
        self.dropout: float = float(self.params.get("dropout", 0.1))
        self.epochs: int = int(self.params.get("epochs", 8))
        self.batch_size: int = int(self.params.get("batch_size", 64))
        self.lr: float = float(self.params.get("lr", 1e-3))
        self.device: str = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # 內部模組
        self.model: Optional[HazardLSTM] = None
        self.input_spec = InputSpec(features=self.features, window=self.window, step=self.step)
        self.target_name = "TTW_min_survival"

        # 監控
        self.train_history: List[Dict[str, float]] = []

    # --------------------- 訓練 ---------------------
    def fit(self, train_ds: Dataset, val_ds: Optional[Dataset] = None, callbacks=None) -> Dict[str, float]:
        assert torch is not None, "PyTorch is required for training."
        input_dim = len(self.features)
        H = self.horizon
        self.model = HazardLSTM(input_dim, self.hidden, self.layers, H, self.dropout).to(self.device)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False) if val_ds else None

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val = float("inf")
        best_state = None

        for ep in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            nstep = 0
            for X, y, m, *_ in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                m = m.to(self.device)
                opt.zero_grad()
                logits = self.model(X)
                loss = _hazard_nll(logits, y, m)
                loss.backward()
                opt.step()
                total += loss.item()
                nstep += 1
            train_loss = total / max(nstep, 1)

            # 驗證
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                tot = 0.0
                c = 0
                with torch.no_grad():
                    for X, y, m, *_ in val_loader:
                        X = X.to(self.device); y = y.to(self.device); m = m.to(self.device)
                        logits = self.model(X)
                        loss = _hazard_nll(logits, y, m)
                        tot += loss.item(); c += 1
                val_loss = tot / max(c, 1)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            self.train_history.append({"epoch": ep, "train_loss": train_loss, "val_loss": float(val_loss or 0.0)})
            if callbacks and hasattr(callbacks, "on_epoch_end"):
                try:
                    callbacks.on_epoch_end(ep, self.epochs, float(val_loss or train_loss))
                except Exception:
                    pass

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_loss": self.train_history[-1]["train_loss"],
            "val_loss": self.train_history[-1].get("val_loss", None) or self.train_history[-1]["train_loss"],
        }

    # --------------------- 推論 ---------------------
    def predict(self, x: np.ndarray, horizon: Optional[int] = None) -> Dict[str, float]:
        """輸入單筆或多筆窗口 (B,T,D)。回傳分位數與品質標記。
        若 batch>1，取平均分位後輸出（入口層另做逐設備匯出）。
        """
        assert self.model is not None, "Model not loaded."
        H = int(horizon or self.horizon)
        if H != self.horizon:
            # 臨時縮/放 horizon：以 head 重新線性投影（簡化處理）
            with torch.no_grad():
                last_head = self.model.head[-1]
                new_head = nn.Linear(last_head.in_features, H).to(self.device)
                new_head.weight.data[: min(H, last_head.out_features), :last_head.in_features] = 0
                self.model.head[-1] = new_head
                self.horizon = H

        X = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if X.ndim == 2:
            X = X.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            qs = _hazard_to_quantiles(logits, [0.1, 0.5, 0.9])  # (B,3)
            q = qs.mean(dim=0).cpu().numpy().tolist()
        out = {
            "TTW_p10": float(q[0]),
            "TTW_p50": float(q[1]),
            "TTW_p90": float(q[2]),
            "quality": "ok",
        }
        return out

    # --------------------- 儲存/載入 ---------------------
    def save(self, path: str, extra_meta: dict | None = None) -> None:
        assert self.model is not None, "No model to save."
        _ensure_dir(os.path.dirname(path))
        torch.save({
            "state_dict": self.model.state_dict(),
            "params": self.params,
            "input_spec": {
                "features": self.features,
                "window": self.window,
                "step": self.step,
                "horizon": self.horizon,
            },
            "target": self.target_name,
            "meta": extra_meta or {},
        }, path)

    @classmethod
    def load(cls, path: str) -> "LSTMTrend":
        if not os.path.isfile(path):
            raise Exception(f"[ModelFileMissing] Not found: {path}")
        
        if not torch:
            raise RuntimeError("PyTorch is required to load models")
        
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise Exception(f"[ModelLoadError] Failed to load model file {path}: {e}")
        
        params = obj.get("params", {})
        inst = cls(params)
        
        input_dim = len(inst.features)
        H = int(obj.get("input_spec", {}).get("horizon", inst.horizon))
        inst.horizon = H
        inst.model = HazardLSTM(input_dim, inst.hidden, inst.layers, H, inst.dropout)
        inst.model.load_state_dict(obj["state_dict"])
        inst.model = inst.model.to(inst.device)
        return inst


# =====================================================
# 入口函式：訓練與預測
# =====================================================

def _update_index(store_id: str, version: str, model_path: str, metrics_path: str, model_type: str):
    idx = _read_json(INDEX_PATH, default={})
    node = idx.get(store_id, {"latest": version, "versions": []})
    # 移除舊同版（若重複）
    node["versions"] = [v for v in node.get("versions", []) if v.get("version") != version]
    node["versions"].insert(0, {
        "version": version,
        "model": model_path,
        "metrics": metrics_path,
        "model_type": model_type,
    })
    node["latest"] = version
    idx[store_id] = node
    _write_json(INDEX_PATH, idx)


def _resolve_model_path(store_id: str, version: str) -> Tuple[str, str]:
    mdir = os.path.join("weighted_models", store_id)
    _ensure_dir(mdir)
    model_path = os.path.join(mdir, f"model_{store_id}_{version}.pt")
    metrics_path = os.path.join(mdir, f"metrics_{store_id}_{version}.json")
    return model_path, metrics_path


def _resolve_version_from_index(store_id: str, version: str) -> Tuple[str, dict]:
    idx = _read_json(INDEX_PATH)
    if store_id not in idx:
        raise Exception(f"[IndexMissing] Store '{store_id}' not found in index at {INDEX_PATH}")
    node = idx[store_id]
    if not version:
        if not node.get("latest"):
            raise Exception(f"[VersionMissing] No 'latest' version recorded for store '{store_id}'")
        version = node["latest"]
    found = None
    for v in node.get("versions", []):
        if v.get("version") == version:
            found = v
            break
    if not found:
        raise Exception(f"[VersionMissing] Version '{version}' not found for store '{store_id}'")
    return version, found


# --------------------- 前處理 ---------------------
def _compute_ingest_ts(g: pd.DataFrame) -> pd.Series:
    """由 defrost_status 推估跟上次除霜結束分鐘數（ingest_ts）。
    要則：找到 1→0 的轉折點視為「除霜結束」。之後每分鐘累加。
    若未知或無轉折，回傳 NaN。
    """
    st = g["defrost_status"].fillna(0).astype(int).values
    ts = g["ts"].values
    n = len(g)
    last_end_idx = -1
    ingest = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if i > 0 and st[i - 1] == 1 and st[i] == 0:
            last_end_idx = i
            ingest[i] = 0.0
        elif last_end_idx >= 0:
            # 分鐘差
            dt = (pd.Timestamp(ts[i]) - pd.Timestamp(ts[last_end_idx])) / np.timedelta64(1, "m")
            ingest[i] = float(max(dt, 0.0))
    return pd.Series(ingest, index=g.index, name="ingest_ts")


def _resample_minutely(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    out = []
    for (sid, did), g in df.groupby(["store_id", "device_id"], sort=False):
        g = g.sort_values("ts").set_index("ts")
        # ☆ 關鍵修正：若同一分鐘有多筆，會導致 resample/reindex 報
        #   "cannot reindex on an axis with duplicate labels"
        #   這裡保留最後一筆，先把重複時間戳去掉再做 resample。
        if not g.index.is_monotonic_increasing:
            g = g.sort_index()
        if g.index.has_duplicates:
            g = g[~g.index.duplicated(keep="last")]
        # 1 分鐘對齊，forward fill 部分欄位
        g2 = g.resample("1min").ffill()
        g2["store_id"] = sid
        g2["device_id"] = did
        g2["ts"] = g2.index
        out.append(g2.reset_index(drop=True))
    df = pd.concat(out, ignore_index=True)
    # ingest_ts 若缺→由 defrost_status 推算
    if "ingest_ts" not in df.columns or df["ingest_ts"].isna().all():
        df = df.sort_values(["device_id", "ts"])  # 保證順序
        df["ingest_ts"] = df.groupby("device_id", group_keys=False).apply(_compute_ingest_ts)
    return df


# --------------------- 模型訓練入口 ---------------------
def predict_store(store_id: str, horizon: int = 12, version: str = ""):
    """預測函式：載入模型並生成預測結果
    回傳: (pred_file, meta_file, used_version, stats_dict)
    """
    if not version:
        used_version, node = _resolve_version_from_index(store_id, "")
    else:
        used_version, node = _resolve_version_from_index(store_id, version)
    
    model_path = node.get("model")
    if not os.path.isfile(model_path):
        raise Exception(f"Model file not found: {model_path}")
    
    # 載入模型
    trend = LSTMTrend.load(model_path)
    
    # 載入最新資料進行預測
    ctx_path = os.path.join("weighted_models", store_id, f"context_{store_id}_{used_version}.csv")
    if not os.path.exists(ctx_path):
        raise Exception(f"Context data not found: {ctx_path}")
    
    ctx = pd.read_csv(ctx_path)
    rows = []
    
    for device_id, g in ctx.groupby("device_id", sort=False):
        g = g.sort_values("ts")
        if len(g) < trend.window:
            continue
        
        X = g.iloc[-trend.window:][trend.features].to_numpy(dtype=np.float32)
        pred = trend.predict(X, horizon=horizon)
        
        rows.append({
            "ts": g["ts"].iloc[-1],
            "device_id": device_id,
            "TTW_p10": pred["TTW_p10"],
            "TTW_p50": pred["TTW_p50"],
            "TTW_p90": pred["TTW_p90"],
            "quality": pred["quality"]
        })
    
    # 輸出結果
    df_out = pd.DataFrame(rows)
    out_dir = os.path.join("weighted_models", store_id)
    pred_file = os.path.join(out_dir, f"pred_{store_id}_{used_version}.csv")
    meta_file = os.path.join(out_dir, f"meta_{store_id}_{used_version}.json")
    
    df_out.to_csv(pred_file, index=False)
    
    meta = {
        "store_id": store_id,
        "version": used_version,
        "horizon": horizon,
        "samples": len(df_out)
    }
    _write_json(meta_file, meta)
    
    stats = {
        "samples": len(df_out),
        "y_mean": float(df_out["TTW_p50"].mean()),
        "y_min": float(df_out["TTW_p50"].min()),
        "y_max": float(df_out["TTW_p50"].max())
    }
    
    return pred_file, meta_file, used_version, stats

# --------------------- 預測入口 ---------------------
def predict_store(store_id: str, horizon: int = 12, version: str = ""):
    """預測函式：載入模型並生成預測結果"""
    if not version:
        used_version, node = _resolve_version_from_index(store_id, "")
    else:
        used_version, node = _resolve_version_from_index(store_id, version)
    
    model_path = node.get("model")
    if not os.path.isfile(model_path):
        raise Exception(f"Model file not found: {model_path}")
    
    # 載入模型
    trend = LSTMTrend.load(model_path)
    
    # 載入最新資料進行預測
    ctx_path = os.path.join("weighted_models", store_id, f"context_{store_id}_{used_version}.csv")
    if not os.path.exists(ctx_path):
        raise Exception(f"Context data not found: {ctx_path}")
    
    ctx = pd.read_csv(ctx_path)
    rows = []
    
    for device_id, g in ctx.groupby("device_id", sort=False):
        g = g.sort_values("ts")
        if len(g) < trend.window:
            continue
        
        X = g.iloc[-trend.window:][trend.features].to_numpy(dtype=np.float32)
        pred = trend.predict(X, horizon=horizon)
        
        rows.append({
            "ts": g["ts"].iloc[-1],
            "device_id": device_id,
            "TTW_p10": pred["TTW_p10"],
            "TTW_p50": pred["TTW_p50"],
            "TTW_p90": pred["TTW_p90"],
            "quality": pred["quality"]
        })
    
    # 輸出結果
    df_out = pd.DataFrame(rows)
    out_dir = os.path.join("weighted_models", store_id)
    pred_file = os.path.join(out_dir, f"pred_{store_id}_{used_version}.csv")
    meta_file = os.path.join(out_dir, f"meta_{store_id}_{used_version}.json")
    
    df_out.to_csv(pred_file, index=False)
    
    meta = {
        "store_id": store_id,
        "version": used_version,
        "horizon": horizon,
        "samples": len(df_out)
    }
    _write_json(meta_file, meta)
    
    stats = {
        "samples": len(df_out),
        "y_mean": float(df_out["TTW_p50"].mean()),
        "y_min": float(df_out["TTW_p50"].min()),
        "y_max": float(df_out["TTW_p50"].max())
    }
    
    return pred_file, meta_file, used_version, stats

# =====================================================
# 簡易評估（校準/命中）
# =====================================================

def _eval_metrics(model: LSTMTrend, val_ds: SeqSurvivalDataset) -> Tuple[float, float, Dict[float, float]]:
    """以 validation dataset 粗估：
    - MAE(TTW)：以 p50 與實際 TTW 比（若截尾 -> 以 horizon 計）
    - ±3 分命中率：|p50 - 真值| <=3
    - 分位校準：p10/p50/p90 命中率
    """
    if torch is None or model.model is None:
        return 9.9, 0.0, {0.1: 0.1, 0.5: 0.5, 0.9: 0.9}

    loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    H = model.horizon
    p10hits = p50hits = p90hits = 0
    n = 0
    abs_err = []
    within3 = 0
    with torch.no_grad():
        for X, y, m, *_ in loader:
            X = X.to(model.device)
            logits = model.model(X)
            qs = _hazard_to_quantiles(logits, [0.1, 0.5, 0.9])  # (B,3)
            p10, p50, p90 = [qs[:, i].cpu().numpy() for i in range(3)]

            # 建真值（若 y 有事件，事件 bin 位置即真 TTW；若全 0 則 =H）
            y_np = y.numpy()
            true_ttw = []
            for row in y_np:
                if row.sum() <= 0:
                    true_ttw.append(H)
                else:
                    true_ttw.append(int(np.argmax(row) + 1))
            true_ttw = np.asarray(true_ttw)

            abs_err.extend(np.abs(p50 - true_ttw))
            within3 += int(((np.abs(p50 - true_ttw)) <= 3).sum())

            p10hits += int((true_ttw <= p10).sum())
            p50hits += int((true_ttw <= p50).sum())
            p90hits += int((true_ttw <= p90).sum())
            n += len(true_ttw)

    mae = float(np.mean(abs_err)) if abs_err else 9.9
    hit3 = float(within3 / max(n, 1))
    cal = {
        0.1: float(p10hits / max(n, 1)),
        0.5: float(p50hits / max(n, 1)),
        0.9: float(p90hits / max(n, 1)),
    }
    return mae, hit3, cal

def train_store_lstm(store_id: str, df: pd.DataFrame, params: dict, callbacks=None, simulate: bool = False):
    """訓練 LSTM 模型的統一入口函式
    回傳: (model_path, metrics_path, version)
    """
    print(f"DEBUG: train_store_lstm start: store={store_id}, df.shape={df.shape}, simulate={simulate}")
    
    if simulate:
        raise ValueError("Simulation mode not supported. Use real training data.")

    # 參數標準化
    p = dict(params or {})
    p["model_type"] = _standardize_model_type(p.get("model_type", "LSTM_trend"))
    p["window"]  = int(p.get("window",  p.get("lookback",  DEFAULT_WINDOW)))
    p["horizon"] = int(p.get("horizon", DEFAULT_HORIZON))
    p["hidden"]  = int(p.get("hidden",  p.get("hidden_size", 64)))
    p["layers"]  = int(p.get("layers",  p.get("num_layers", 2)))
    
    # 生成版本號
    version = _next_seq_version(store_id)
    model_path, metrics_path = _resolve_model_path(store_id, version)
    
    # 預處理數據
    df = _resample_minutely(df)
    df = df[df["device_type"].astype(str).str.contains("空調") == False].copy()
    
    if df.empty:
        raise ValueError("No data remaining after preprocessing")
    
    # 檢查數據充足性
    window = int(p.get("window", DEFAULT_WINDOW))
    horizon = int(p.get("horizon", DEFAULT_HORIZON))
    need = window + horizon
    
    mins_per_dev = df.groupby("device_id")["ts"].nunique()
    good_ids = set(mins_per_dev[mins_per_dev >= need].index.astype(str))
    
    if not good_ids:
        detail = "; ".join([f"{str(k)}:{int(v)}/{need}" for k, v in mins_per_dev.items()])
        raise ValueError(f"No training samples (need ≥ {need} mins per device). Detail: {detail}")
    
    df = df[df["device_id"].astype(str).isin(good_ids)].copy()
    
    # 建立設備類型映射
    dev_type = df.groupby("device_id")["device_type"].agg(lambda s: s.iloc[-1]).to_dict()
    
    # 切分訓練/驗證數據
    ts_cut = df["ts"].quantile(0.8)
    df_train = df[df["ts"] <= ts_cut]
    df_val = df[df["ts"] > ts_cut]
    
    # 建立 Dataset
    features = p.get("features", ["temp_current", "t_room", "defrost_status", "power"])
    step = int(p.get("step", DEFAULT_STEP))
    
    train_ds = SeqSurvivalDataset(df_train, features, dev_type, window=window, horizon=horizon, step=step)
    val_ds = SeqSurvivalDataset(df_val, features, dev_type, window=window, horizon=horizon, step=step)
    
    if len(train_ds) == 0:
        raise ValueError("No training samples after windowing")
    
    # 訓練模型
    trend = LSTMTrend(p)
    metrics = trend.fit(train_ds, val_ds=val_ds, callbacks=callbacks)
    
    # 保存模型
    extra_meta = {
        "store_id": store_id,
        "version": version,
        "model_type": p["model_type"],
        "trained_at": datetime.now().isoformat(),
    }
    trend.save(model_path, extra_meta=extra_meta)
    
    # 計算評估指標
    mae_ttw, hit3, pcal = _eval_metrics(trend, val_ds)
    metrics_data = {
        "mae_ttw": float(mae_ttw),
        "hit_±3min": float(hit3),
        "p50_cal": float(pcal.get(0.5, 0.5)),
    }
    _write_json(metrics_path, metrics_data)
    
    # 更新索引
    _update_index(store_id, version, model_path, metrics_path, p["model_type"])
    
    # 保存上下文數據
    ctx_path = os.path.join("weighted_models", store_id, f"context_{store_id}_{version}.csv")
    last_hours = df.groupby("device_id").tail(window + horizon)
    last_hours.to_csv(ctx_path, index=False, encoding="utf-8")
    
    return model_path, metrics_path, version

# =====================================================
# 若以模組形式直接測，給一個簡易 smoke test
# =====================================================
if __name__ == "__main__":
    # 產生假資料做 smoke 測試
    rng = np.random.default_rng(42)
    ts0 = pd.Timestamp("2025-01-01 00:00:00")
    devs = ["F001", "F002", "R001"]
    rows = []
    for d in devs:
        device_type = "冷凍" if d.startswith("F") else "冷藏"
        t = ts0
        temp = -20 if device_type == "冷凍" else 4.0
        for i in range(24*60):
            # 隨機游走 + 偶發除霜
            temp += rng.normal(0, 0.03)
            if device_type == "冷凍":
                temp = max(-25, min(-12, temp))
            else:
                temp = max(1.5, min(8.5, temp))
            defrost = 1 if (i % 360 == 0 and i>0 and rng.uniform()<0.3) else 0
            rows.append({
                "ts": t,
                "store_id": "S001",
                "device_id": d,
                "device_type": device_type,
                "temp_current": temp,
                "t_room": 26 + rng.normal(0, 0.5),
                "defrost_status": defrost,
                "power": 0.6 + 0.2 * (1 if temp < (_temp_safe_by_type(device_type)-2) else 0),
            })
            t += pd.Timedelta(minutes=1)
    df = pd.DataFrame(rows)

    mp, metp, ver = train_store_lstm("S001", df, {"model_type": "trend", "epochs": 1}, simulate=False)
    print("trained:", mp, metp, ver)
    print(predict_store("S001", horizon=60, version=ver))