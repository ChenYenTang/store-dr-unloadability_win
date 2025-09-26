# Store DR Unloadability Console

門市設備卸載潛力分析與決策支援系統  
（Store Demand Response Unloadability Console）

---

## 📖 專案簡介
本系統模擬連鎖門市的冷藏/冷凍櫃等耗能設備，在需量反應（Demand Response, DR）情境下，估算「可安全卸載時間」並產生卸載優先順序。  
透過 **資料收集 → 預測/估算 → 卸載分析 → 優先排序 → 結果輸出** 的流程，協助門市在保障商品溫控安全的前提下，達到節能減碳。

目前系統提供：
- **資料匯入與測試 API**（`/api/devices/feed`）
- **正式評估 API**（`/api/v1/evaluate`）
- **Gradio UI**（掛載於 `/ui`）
- **最終輸出檔**：`output/priority.json`

---

## 📂 專案結構

STORE-DR-UNLOADABILITY/
|-- README.md                     # 專案說明（建議放入架構/流程/擴充說明）
|-- README_Startup.md             # 快速啟動與 API/保守估算說明
|-- requirements.txt              # 依賴套件（FastAPI、Gradio、Torch、Pydantic v2 等）
|-- changelog.md                  # 版本更新紀錄（如有）
|-- payload.json                  #（範例）devices 格式；供 UI/測試展示

|-- config/
|   |-- config.yaml               # 系統/演算法參數（門檻、權重、除霜懲罰…）
|   `-- README.md                 # 設定檔使用說明（如有）

|-- examples/
|   |-- payload.json              # Evaluate API（cabinets 格式）範例
|   `-- README.md                 # 範例說明（如有）

|-- input/
|   |-- feed.json                 # 測試/模擬輸入（devices 格式；含預排分數）
|   |-- latest.json               # 最新設備快照（devices 格式；偏裝置層欄位）
|   |-- payload.json              # /api/devices/feed 暫存（POST 寫入、GET 讀出）
|   `-- history/
|       `-- history.csv           # 歷史累積資料（獨立預測程式用來訓練/推論）

|-- models/
|   `-- lstm_model.pt             # LSTM 權重（未來獨立預測程式／校正模型可用）

|-- output/
|   |-- priority.json             # ⭐ 最終對外輸出（devices 格式；F=冷凍 R=冷藏 S=開放櫃）
|   `-- train_loss.png            # 訓練曲線（供 UI 顯示用）

|-- src/
|   |-- main.py                   # 入口：掛載 Gradio UI 到 /ui；提供 /api/devices/feed
|   |-- README.md                 # 模組/流程說明（目前空，後續可補）
|   |
|   |-- api/
|   |   |-- routes.py             # 預計放 /api/v1/evaluate /health /version（正式評估）
|   |   `-- README.md             # API 說明（如有）
|   |
|   |-- io/
|   |   |-- schema.py             # EvaluateRequest（cabinets 輸入模型）
|   |   `-- README.md             # Schema/驗證規則說明（如有）
|   |
|   |-- models/
|   |   `-- predict.py            # 預測介面（與 LSTM/灰箱模型銜接的封裝）
|   |
|   |-- policy/
|   |   |-- scoring.py            # 保守估算/規則打分（暫用；可被模型結果取代）
|   |   |-- config_loader.py      # 讀取 config.yaml、注入規則/門檻
|   |   `-- README.md             # 演算法/權重/門檻說明
|   |
|   `-- ui/
|       |-- gradio_app.py         # UI 主程式（分頁組裝）
|       |-- helpers.py            # UI 與後端資料橋接（讀寫 input/output 等）
|       |-- README.md
|       `-- tabs/
|           |-- config_tab.py     # 分頁1：基本資料與閥值設定（config）
|           |-- evaluate_tab.py   # 分頁2：資料儲存/模型訓練/預測分析
|           |-- cabinets_tab.py   # 分頁3：設備卸載分析資訊（表格/卡片）
|           `-- output_tab.py     # 分頁4：輸出與測試（產出與預覽 priority.json）

`-- test_app/
    |-- mock_feed.py              # 產生 latest.json / payload.json 的測試工具
    `-- demo_switch.py            # 測試輔助腳本（情境切換/模擬）


## 🚀 啟動方式

### 1. 安裝環境

```bash
pip install -r requirements.txt

### 2. 啟動 API + UI
# FastAPI + Gradio UI
python -m src.main

# 或直接用 uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000

API server: http://127.0.0.1:8000

Gradio UI: http://127.0.0.1:8000/ui

### 3. 測試工具
python test_app/mock_feed.py
可產生 input/latest.json 與 input/payload.json，模擬不同設備情境。

### 4. API 說明
/api/devices/feed（測試/資料匯入）
POST：寫入一批 devices[] 到 input/payload.json
GET：讀出目前暫存的 devices[]
用途：
作為 前端/設備 → 系統 的資料傳輸通道
也能給 獨立預測程式 監聽，追加至 history.csv，再輸出結果到 priority.json

/api/v1/evaluate（正式評估）
POST：輸入 cabinets[]（依 src/io/schema.py 定義），回傳每個櫃的
unloadable_time_min / unloadable_energy_kWh / priority_score / ranking
GET /health：健康檢查
GET /version：版本資訊
用途：
正式 API 輸入，對外標準化介面
目前內建「保守估算」演算法，未來可改為呼叫獨立預測程式

### 5. 資料格式
1. Evaluate 輸入（cabinets）
檔案：input/payload.json（schema 版）
{
  "store_id": "S001",
  "timestamp": "2025-08-14T16:12:06+00:00",
  "business_hours_flag": 1,
  "cabinets": [
    {
      "cabinet_id": "R-01",
      "type": "refrigerator",
      "air_supply_c": 2.5,
      "air_return_c": 4.2,
      "defrost_status": 0,
      "time_since_defrost_min": 200
    }
  ]
}

2. 匯入/暫存/輸出（devices）
檔案：input/feed.json、input/latest.json、output/priority.json
{
  "devices": [
    {
      "Device_ID": "F001",
      "Priority_score": 0.85,
      "Ranking": 1,
      "Unloadable_time_min": 25,
      "Unloadable_energy_kWh": 1.8,
      "Time_since_defrost": 120,
      "Exclude_flag": 0
    }
  ]
}


Device_ID 規則

Fxxx: 冷凍櫃
Rxxx: 冷藏櫃
Sxxx: 開放櫃
Hxxx: 空調（不參與卸載，因此不會出現在 priority.json）

🧮 演算法現況與擴充
保守估算（目前暫用）

### 6. 設計說明

- **演算法**：此版本提供「保守估計版」：

用 `threshold - current_value` 的剩餘溫差 ÷ `rise_c_per_min_max` 推估關機下到達門檻的時間，
再依 `defrost.grace_min / penalty_factor` 進行縮減，僅做範例。
實戰會替換為灰箱熱模型 + LSTM 斜率校正。
優先分：依 `weights` 將時間與能量標準化計分，加上風險與除霜懲罰，標準化加總為 Priority_score。


- **未來規劃**

獨立預測程式：

監看 input/ 與 history/history.csv
使用 LSTM 或灰箱熱模型預測升溫斜率
輸出 output/priority.json 作為最終結果
API 與 UI 不需大改，只要替換底層演算法即可
轉換層：可視需求建立 devices ⇄ cabinets 欄位對照，讓傳輸與正式評估保持一致

📌 版本

最新版本：v0.5.0

更新紀錄請參考 CHANGELOG.md