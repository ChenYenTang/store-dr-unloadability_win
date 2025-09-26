📌 更新紀錄 (Changelog)
### v0.7.0
- Change: 內部資料快照改為「分店儲存」：`input/latest/<Store_ID>.json`（原 `input/latest.json` 已廢止）
- Add: cabinets 分頁改為「表格檢視」、可切換 Store_ID、支援自動更新（每 60 秒本地重讀）
- Add: evaluate 分頁提供「從快照寫入歷史（去重）」；以 (Store_ID, Device_ID, minute(timestamp)) 去重，輸出 `input/history/history.csv`
- API（規格預告）：`GET /devices/{device_id}/realtime` 將改為 **`GET /stores/{store_id}/devices/{device_id}/realtime`**（因 Device_ID 非全域唯一）。舊路徑遇到重名將回 409，請客戶端更新。


### v0.6.0
新增 src/io/company_api.py（封裝 3 GET / 1 POST 與欄位映射）

src/ui/config_tab.py：新增「公司 API 設定」Accordion（自動抓取預設 1 分鐘、可測試/更新門市/儲存/抓取一次/自動抓取）

src/ui/output_tab.py：新增「送出一次」與「自動送出」（預設 20 分鐘）

小修：

src/ui/helpers.py：讀 latest.json 改成 .get("devices", [])

src/ui/evaluate_tab.py：HISTORY_PATH 改為 input/history/history.csv

config/config.yaml：新增 api 預設區塊，auth.type 預設 api_key（IP/Port 可在 UI 改）



### v0.5.0 - 2025-09-17
稍微整理

交給新的GPT專案重讀，修改原本的README.md



[v0.4.0] - 2025-08-23
新增

evaluate_tab

增加「開始追加 latest.json 到 history.csv」功能，支援自動連接 API 並持續更新歷史資料。

新增狀態提示：在最新 10 筆資料表格上方紅字顯示「讀取最新設備資料中……」。

資料自動刷新，讓使用者可即時觀察追加進度。

models/predict.py

新增 LSTM 訓練與推論模組，支援 time steps、hidden units、epochs、batch size 等參數設定。

提供訓練 Loss 曲線輸出，並維護模型檔案狀態（已載入 / 未訓練）。

output_tab

增加自動輸出頻率設定（3、6、9 分鐘）。

新增 priority.json 預覽視窗。

調整輸出格式：Store_ID 放在 devices 外層，Device 內不再包含 Store_ID。

修正

latest.json 與 history.csv 欄位數不一致造成 ParserError 的問題。

evaluate_tab 匯入/追加歷史資料時，會自動補齊缺少欄位，避免模型訓練錯誤。

修正 cabinets_tab 卡片模式無法排序的問題，並支援「優先順序、優先分數、可卸載時間、可節能量、除霜後時間、排除標註」等排序。

調整

main.py

config_tab 改名為 基本資料與閥值設定。

output_tab 改名為 輸出與測試。

evaluate_tab 改名為 設備卸載評估，並拆成子分頁（資料儲存、模型訓練、預測與分析、紀錄與下載）。

命名統一

feed.json → payload.json

輸出結果固定為 priority.json。

[v0.3.0] - 2025-08-21
新增

cabinets_tab

新增表格 / 卡片雙模式顯示，支援即時刷新。

顯示「目前共 N 台設備分析中」統計。

表格欄位改為中文，排序依「優先順序」。

卡片新增下半部資訊（優先順序、分數、可卸載時間、可節能量、除霜後時間、排除標註）。

調整

API /api/devices/feed 增加存取 payload.json，取代原本的 feed.json。

mock_feed.py 改為同時輸出 latest.json 與 payload.json。

[v0.2.0] - 2025-08-20
新增

main.py

整合 FastAPI 與 Gradio UI，提供 /ui 網頁前端。

API /api/devices/feed 可接收 POST/GET，並存取 feed.json。

cabinets_tab

初版表格模式與卡片模式切換。

mock_feed.py

模擬設備資料，定時更新 latest.json 與 feed.json。

[v0.1.0] - 2025-08-18
新增

專案初始化。

基本 Gradio UI：分頁包括 config_tab、evaluate_tab、cabinets_tab、output_tab。

API 架構雛形，支援設備資料收集與顯示。