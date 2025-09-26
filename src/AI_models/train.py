from typing import Tuple
from .lstm_trend import train_store_lstm as _trend_train  # 直接委派到可用實作
try:
    from .lstm_trend import train_store_lstm as _wave_train  # 若未提供則為 None
except Exception:
    _wave_train = None

def train_store_lstm(store_id: str, df, params: dict, callbacks, simulate: bool = False) -> Tuple[str, str, str]:
    """
    Dispatcher：依 params['model_type'] 轉呼叫對應模型訓練。
    目前支援：LSTM_trend；若之後有 LSTM_wave 也可自動掛載。
    """
    mt = str((params or {}).get("model_type", "LSTM_trend")).strip().lower()
    if mt in ("lstm_trend", "trend", ""):
        return _trend_train(store_id, df, params, callbacks, simulate=False)  # 一律真訓練
    if mt in ("lstm_wave", "wave") and _wave_train:
        return _wave_train(store_id, df, params, callbacks, simulate=False)
    raise NotImplementedError(f"Unsupported model_type: {mt}")