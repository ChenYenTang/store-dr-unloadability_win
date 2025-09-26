test_app

這個資料夾放的是本地測試工具與模擬伺服器，用來在不依賴公司正式 API 的情況下，驗證 UI 與資料流程。

內容物

mock_company_api.py
本地 Mock 伺服器（FastAPI）。提供 3 個 GET 與 1 個 POST 端點：

GET /stores：列出固定 50 間門市（S001 ~ S050）

GET /stores/{store_id}/devices：該門市設備清單（{"devices":[{"id":"F001","type":"freezer"}, ...]}）

GET /stores/{store_id}/devices/{device_id}/realtime：設備即時資料（中文欄位，見下方範例）

POST /output/priority.json：接收並回覆一個統計結果（僅做回傳，不驗證）

api_smoke.py
CLI 健檢工具。可打 Mock 或 正式公司 API，做連線/資料流檢查（列門市、列設備、抓即時、送 priority）。

需求

Python 3.9+

FastAPI / Uvicorn（只給 Mock 用）

httpx / pyyaml（api_smoke.py 使用）

安裝（專案根目錄執行）：

pip install fastapi uvicorn httpx pyyaml

一、啟動 Mock 伺服器
uvicorn test_app.mock_company_api:app --host 0.0.0.0 --port 9000

Mock 行為說明

門市固定：S001 ~ S050

資料每「分鐘」重生（同一分鐘請求回同一份）

15% 機率該店本輪「沒資料」（/stores/{id}/devices 會回空陣列）

5% 機率單台設備本輪 204（/realtime 無內容）

Device_ID 僅三碼（Fxxx/Rxxx/Sxxx/Hxxx），不含店號

realtime 走路徑版：/stores/{store_id}/devices/{device_id}/realtime

可調整（環境變數）：

# 亂數種子（可重現）
set MOCK_SEED=42
# 店級無資料比例（0~1）
set NO_DATA_RATE=0.15
# 單台設備 204 比例（0~1）
set MISS_RATE=0.05

端點回覆範例

GET /stores

[
  {"id":"S001","name":"店001"},
  {"id":"S002","name":"店002"},
  ...
]


GET /stores/S010/devices

{
  "devices": [
    {"id":"F001","type":"freezer"},
    {"id":"R001","type":"refrigerator"},
    {"id":"S001","type":"open_shelf"},
    {"id":"H001","type":"hvac"}
  ]
}


GET /stores/S010/devices/F001/realtime（櫃類）

{
  "timestamp":"2025-09-17T14:03:00+08:00",
  "Device_ID":"F001",
  "Device_type":"冷凍",
  "Temp_current":-18.7,
  "Defrost_status":0
}


GET /stores/S010/devices/H001/realtime（空調）

{
  "timestamp":"2025-09-17T14:03:00+08:00",
  "Device_ID":"H001",
  "Device_type":"空調",
  "T_room":24.3
}


POST /output/priority.json

{
  "ok": true,
  "received_store": "S012",
  "device_count": 18,
  "received_at": "2025-09-17T06:03:00+00:00",
  "note": null
}

二、使用 api_smoke.py 做 API 健檢

全域參數請放在子指令前面。
--base-url 也可改用環境變數 COMPANY_API_BASE_URL。

常用指令（打本機 Mock：:9000）
# 列門市
python -m test_app.api_smoke --base-url http://127.0.0.1:9000 stores

# 列指定門市設備
python -m test_app.api_smoke --base-url http://127.0.0.1:9000 devices --store S010

# 抓指定門市所有設備即時資料（逐台）
python -m test_app.api_smoke --base-url http://127.0.0.1:9000 realtime --store S010

# 送出 priority（讀 output/priority.json；若無 Store_ID 會從 input/config.json 補）
python -m test_app.api_smoke --base-url http://127.0.0.1:9000 post


若使用 Token / API Key：

# Bearer
python -m test_app.api_smoke --base-url http://127.0.0.1:9000 --auth-type bearer --token <TOKEN> stores

# API Key（預設 Header: X-API-Key，可用 --api-key-header 自訂）
python -m test_app.api_smoke --base-url http://127.0.0.1:9000 --auth-type api_key --api-key <KEY> stores

預期輸出（節錄）

stores

[
  {"id":"S001","name":"店001"},
  {"id":"S002","name":"店002"},
  {"id":"S003","name":"店003"}
]


devices --store S010

{
  "devices": [
    {"id":"F001","type":"freezer"},
    {"id":"R001","type":"refrigerator"},
    {"id":"S001","type":"open_shelf"},
    {"id":"H001","type":"hvac"}
  ]
}


realtime --store S010（成功會逐台印出 OK F001…最後輸出一個陣列）

[
  {
    "device_id": "F001",
    "data": {
      "timestamp": "2025-09-17T14:03:00+08:00",
      "Device_ID": "F001",
      "Device_type": "冷凍",
      "Temp_current": -18.7,
      "Defrost_status": 0
    }
  },
  {
    "device_id": "H001",
    "data": {
      "timestamp": "2025-09-17T14:03:00+08:00",
      "Device_ID": "H001",
      "Device_type": "空調",
      "T_room": 24.3
    }
  }
]


post

POST /output/priority.json -> 200
{"ok":true,"received_store":"S012","device_count":18,...}

三、與 UI 的搭配

在 UI 的「公司 API 設定」把 Base URL 設為 http://127.0.0.1:9000，端點路徑請使用路徑版：

/stores

/stores/{store_id}/devices

/stores/{store_id}/devices/{device_id}/realtime

POST /output/priority.json

「抓取一次（測試）」成功後，會在 input/latest/<Store_ID>.json 生成快照；cabinets 分頁可直接讀檔顯示表格。

四、常見問題（Troubleshooting）

realtime 打到舊路徑 404
請確認路徑是否為：/stores/{store_id}/devices/{device_id}/realtime（非 /devices/{device_id}/realtime）。

api_smoke.py: error: unrecognized arguments
請把 --base-url 等全域參數放在子指令前面：
python -m test_app.api_smoke --base-url ... stores

Mock 有時回空或 204？
屬於設計行為：店級 15% 機率本輪沒資料、設備級 5% 機率 204（可用環境變數調整）。

版本對照（重要）

Mock 與 api_smoke.py 皆已支援 路徑版 realtime。

之後公司正式 API 若切換到路徑版，請在 config/config.yaml 的 api.endpoints.realtime_of_device 同步更新為：

/stores/{store_id}/devices/{device_id}/realtime