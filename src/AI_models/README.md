模型訓練模擬器（README）

放置位置：models/readme.md
目的：說明目前「模型訓練」是如何以模擬器實現、與 UI 的介接契約、輸出檔案與版本規範，以及日後切換為真 LSTM時應遵循的介面與作法。

1. 總覽

目前的「模型訓練」頁（Evaluate → 子分頁「模型訓練」）使用 模擬器 代替真實訓練流程，以便：

先打通 UI/UX 與檔案流程；

穩定輸出模型與指標檔（路徑與格式固定）；

支援並行多店，不阻塞 Gradio 介面；

預留未來切換至真 LSTM 訓練的位置與統一介面。

核心介面（已定版，不建議破壞）：

# src/models/train.py
def train_store_lstm(store_id: str, df, params: dict, callbacks, simulate: bool = True):
    """
    df:     單一門市時間序列資料（已過濾；CSV schema 同下）
    params: LSTM 超參數與選項（見 §3.4）
    callbacks: 需支援 callbacks.on_epoch_end(epoch, total_epochs, loss)
    simulate: True = 跑模擬器；False = 之後接真訓練
    return: (model_path: str, metrics_path: str, version: str)
    """


UI 端（src/ui/tabs/train_tab.py）為每家店啟動一個背景執行緒呼叫此函式，並透過 callback 逐 epoch 回報進度。

2. 目錄與資料流（不可改動的前提）
2.1 資料流（沿用既有規格）
config.json
→ input/latest/<Store_ID>.json
→ input/history/YYYYMMDD.csv（去重）
→ models
→ output/priority.json

2.2 CSV Schema（固定）

所有日檔位於 input/history/YYYYMMDD.csv，欄位：

["ts","store_id","device_id","device_type","temp_current","t_room","defrost_status","power","ingest_ts"]


UI 在讀取資料時會依 ts 排序，最小策略處理缺值（直接丟棄不完整列）。

3. 模擬器行為與參數
3.1 觸發條件

透過 UI，對每個選定的 store_id 啟動一個 worker thread。

預設由環境變數 SIMULATE=1 開啟模擬模式（UI 也會塞 _simulate=True 到 params 以保險）。

3.2 可重現性（seed）

使用下述關鍵資訊生成隨機種子，確保同樣的輸入得到同樣的輸出：

seed = hash(f"{store_id}|{start_date}|{end_date}|{lookback}|{horizon}") % 2**32


其他超參數目前不參與 seed；之後若引入，務必考慮相容性（不要破壞既有版本的可重現性）。

3.3 訓練過程（模擬）

共 epochs 個 epoch。

每個 epoch：

sleep 隨機 0.2–0.5 秒（模擬訓練時間）

回報遞減的 loss（指數遞減 + 少量雜訊）

透過 callbacks.on_epoch_end(epoch, total_epochs, loss) 通知 UI

完成後產生一組合理範圍的指標（隨機，但可重現）：

R² ∈ [0.60, 0.95]

MAE ∈ [0.3, 1.5]

MSE ∈ [0.5, 3.0]

SMAPE ∈ [6, 18]

3.4 參數（來自 UI）

基本：

lookback, horizon, hidden_size, num_layers, dropout, batch_size, epochs, lr


進階（Accordion）：

scaler ∈ {none, zscore, minmax}
loss   ∈ {mse, mae}


這些參數會原樣寫入 metrics_*.json 的 params 欄位。之後接真 LSTM 時，請實作對應行為（例如 scaler、loss 的實際選擇）。

4. 檔案輸出與版本規範（關鍵契約）
4.1 版本字串
vYYYYMMDDHHMMSS    # 秒級、無時區

4.2 檔案路徑（每家店）
models/{Store_ID}/model_{Store_ID}_{version}.pt
models/{Store_ID}/metrics_{Store_ID}_{version}.json


目前 .pt 是小型 JSON 佔位檔（方便之後無痛換成真 PyTorch 權重）。切換為真模型時，路徑與檔名格式不可變。

4.3 原子寫入

所有寫檔採 .tmp → os.replace()，確保不會留下半成品：

先寫到 *.tmp

os.replace(tmp, final_path)

4.4 metrics_*.json Schema
{
  "store_id": "S001",
  "version": "v20250920145712",
  "range": {"start":"2025-09-15","end":"2025-09-16"},
  "samples": 864,
  "params": {
    "lookback":48,"horizon":6,"hidden_size":64,"num_layers":2,"dropout":0.2,
    "batch_size":64,"epochs":8,"lr":0.001,"scaler":"zscore","loss":"mse",
    "_range":{"start":"2025-09-15","end":"2025-09-16"},
    "_simulate": true
  },
  "duration_sec": 14.372,
  "R2": 0.91, "MAE": 0.72, "MSE": 1.38, "SMAPE": 9.84,
  "generated_at": "2025-09-20T14:57:12"
}


時間字串統一 YYYY-MM-DDTHH:MM:SS（無時區）。
SMAPE 計算請加小 eps 避免除以 0（見 §6.2）。

4.5 models/index.json 結構（聚合索引）
{
  "S001": {
    "latest": "v20250920145712",
    "versions": [
      {
        "version": "v20250920145712",
        "model": "models/S001/model_S001_v20250920145712.pt",
        "metrics": "models/S001/metrics_S001_v20250920145712.json",
        "params": { "...": "..." },
        "range": {"start": "2025-09-15", "end": "2025-09-16"}
      }
    ]
  },
  "S003": { "...": "..." }
}


新版本追加進 versions，同時更新 latest。整檔以 .tmp → os.replace() 寫回。

5. UI 介接（Gradio / Evaluate → 子分頁「模型訓練」）
5.1 執行緒與進度刷新

每家店 → 一個 threading.Thread worker。

UI 以 gr.Timer(interval=1.5s) 輪詢共享狀態（STATUS dict）刷新表格。

不阻塞 Gradio queue。

5.2 Callback 契約

UI 端傳入一個 callbacks，需支援：

callbacks.on_epoch_end(epoch: int, total_epochs: int, loss: float)


模擬器/真訓練每個 epoch 結束時呼叫；UI 會更新：

Status, Progress(%), Epoch, Loss

5.3 UI 欄位（每店一列）
Store_ID | Status | Progress(%) | Epoch | Loss | R2 | MAE | MSE | SMAPE | Model | Metrics

5.4 UI 邏輯提示（保持不變）

同時選店數 >5：顯示黃色警告（仍繼續）。

樣本數 <200：顯示黃色警告（仍繼續）。

日期區間有缺日檔：狀態列列出缺少日期（允許繼續）。

若該店已有版本：顯示「已存在 v{latest}」，可 [載入]（不啟動訓練）或 [重新訓練]。

6. 指標定義（供真訓練沿用）

令 y 為真值、ŷ 為預測，N 為樣本數，eps = 1e-8。

MAE：(1/N) * Σ |y - ŷ|

MSE：(1/N) * Σ (y - ŷ)^2

RMSE（可選）：sqrt(MSE)（UI 目前只顯示 MSE）

SMAPE（%）：(1/N) * Σ ( 2*|y-ŷ| / (|y| + |ŷ| + eps) ) * 100

R²：1 - SSE/SST，其中 SSE = Σ (y-ŷ)^2，SST = Σ (y - mean(y))^2

請將這些值寫入 metrics_*.json，小數位建議保留 6 位（目前模擬器也是如此）。

7. 切換為真 LSTM 的實作指南

目標：不改 UI、不改檔名/路徑/版本/指標格式，僅改 simulate=False 分支內容。

7.1 實作位置

src/models/train.py::train_store_lstm(..., simulate=False) 分支：

請保持相同回傳值：(model_path, metrics_path, version)

中繼過程逐 epoch呼叫 callbacks.on_epoch_end(...) 更新進度

7.2 建議步驟（Pseudo）
if simulate:
    ...  # 保留現有模擬器
else:
    t0 = time.time()
    # 1) 資料前處理（依 params.scaler / lookback / horizon）
    X_train, y_train, X_val, y_val = make_windows(df, params)
    scaler_state = fit_scaler_if_any(params, X_train)
    # 2) 建模（hidden_size, num_layers, dropout, loss, lr, ...）
    model = build_lstm(params)
    opt   = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss() or nn.L1Loss()  # 依 params.loss
    # 3) 訓練迴圈
    for epoch in range(1, params["epochs"] + 1):
        loss = train_one_epoch(model, X_train, y_train, loss_fn, opt, batch_size=params["batch_size"])
        # （可選）驗證
        callbacks.on_epoch_end(epoch, params["epochs"], float(loss))
    # 4) 產生預測並計算指標（R2/MAE/MSE/SMAPE）
    y_pred = predict(model, X_val)
    metrics = compute_metrics(y_val, y_pred)
    duration = time.time() - t0
    # 5) 寫出 .pt（真權重）與 metrics.json（含上述欄位）
    torch.save(model.state_dict(), tmp_model_path); os.replace(...)
    atomic_write_json(metrics_path, metrics_doc)
    # 6) 更新 models/index.json
    update_index(...)
    return model_path, metrics_path, version

7.3 注意事項

不要改檔名/路徑/版本與 metrics.json 格式；

每個 epoch 必須回報一次 callbacks.on_epoch_end(...)，否則 UI 進度不會動；

保留 .tmp → os.replace()；

若導入 scaler/標準化，請將其狀態（例如 min/max、mean/std）寫入 .pt 或另檔，但不要改 .pt 路徑格式（可將這些狀態一併包裝在可序列化物件中儲存）；

多執行緒同時更新 models/index.json：目前同一進程內以原子寫入處理，請避免多個 app 實例同時訓練同一組門市。

8. 測試與驗收清單
8.1 功能測試（模擬器）

進 Train 子分頁 → 預設門市自動勾選（來自 input/config.json 的 Selected_Store_IDs）。

選 1–2 家 → epochs=5~10 → 開始訓練 → 觀察每 1–2 秒進度更新（Epoch/Loss/Progress）。

完成後每列顯示 R2/MAE/MSE/SMAPE 與 Model/Metrics 路徑。

確認生成：

models/{sid}/model_{sid}_v*.pt
models/{sid}/metrics_{sid}_v*.json
models/index.json（latest 指向新版本）

8.2 邊界與提醒

勾選 6 家以上 → 有黃色警告但照跑。

選取日期有缺日檔 → 狀態列列出缺少日期（允許繼續）。

樣本 <200 → 黃色警告（允許繼續）。

連跑兩次同店不同參數 → index.json 追加兩個版本、latest 指向新者。

連跑兩次同店相同參數 → 版本字串不同（秒級），兩個版本並存。

9. 已知限制與後續工作

目前 .pt 為占位 JSON（真訓練請換成權重檔，但路徑/命名不變）。

模擬器的 metrics 為範圍內亂數，不可用於真實評估。

尚未設並行上限（UI 只做警告）；真訓練建議未來加 queue 或最多同時 N 店。

資料前處理（例如缺值處理、異常值處理、特徵工程）目前採最小策略；真訓練時請補強。

10. 快速 FAQ（給新同事 / 新 GPT 分頁）

Q：要從哪裡接真訓練？
A：src/models/train.py::train_store_lstm(simulate=False) 分支，維持相同回傳與寫檔介面。

Q：UI 怎麼拿進度？
A：我們會在每個 epoch 結束時呼叫 callbacks.on_epoch_end(...)；UI 1.5 秒輪詢共享狀態繪表。

Q：檔名與路徑能改嗎？
A：不能。請保留 models/{Store_ID}/model_{Store_ID}_{version}.pt 與對應 metrics_*.json、models/index.json 結構。

Q：版本怎麼產生？
A：vYYYYMMDDHHMMSS（秒級）。不要嵌入時區資訊。

Q：模擬器要關掉怎麼做？
A：環境變數 SIMULATE=0，並且在 UI/params _simulate=False（UI 目前預設會依 SIMULATE 設定）。

11. 參考檔案

src/models/train.py（本模擬器與介面定義）

src/ui/tabs/train_tab.py（UI 面板 + 背景執行緒 + callback + 指標顯示與檔案刷新）

models/index.json（訓練完成後會自動更新）

有任何要替換為真模型的需求，務必先確認本 README 的「介面不變」原則（特別是輸出檔名與 JSON 結構），以確保 UI 與後續「預測與分析」分頁能無縫接軌。