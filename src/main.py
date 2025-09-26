import os
import json
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, Response

# 引入各個分頁
from src.ui.tabs import config_tab, evaluate_tab, cabinets_tab, output_tab

app = FastAPI()


# -------------------------
# API - 模擬設備評估結果
# -------------------------
@app.post("/api/devices/feed")
def update_devices_feed(payload: dict):
    devices = payload.get("devices", [])
    # 存到 payload.json
    with open("input/payload.json", "w", encoding="utf-8") as f:
        json.dump({"devices": devices}, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "count": len(devices)}

@app.get("/api/devices/feed")
def get_devices_feed():
    if os.path.exists("input/payload.json"):
        with open("input/payload.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {"devices": []}

# ---- v0.6.0: 靜態資源預設（避免 /manifest.json 與 /favicon.ico 404）----
@app.get("/manifest.json", include_in_schema=False)
def web_manifest():
    return JSONResponse({
        "name": "Store DR Unloadability",
        "short_name": "SDRU",
        "start_url": "/",
        "display": "standalone",
        "icons": [
            {"src": "/favicon.ico", "sizes": "32x32", "type": "image/x-icon"}
        ],
        "theme_color": "#0f172a",
        "background_color": "#0f172a"
    })

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # 若你有真實小圖示，放在 static/favicon.ico（見下方可選方案）
    icon_path = os.path.join("static", "favicon.ico")
    if os.path.exists(icon_path):
        return FileResponse(icon_path, media_type="image/x-icon")
    # 沒有檔就回 204，避免 404 汙染 log
    return Response(status_code=204)

# -------------------------
# Gradio UI
# -------------------------
def build_demo():
    with gr.Blocks(title="Store DR Unloadability Console") as demo:
        gr.Markdown("# 門市卸載評估控制台")

        with gr.Tabs():
            # 基本資料與閥值設定
            config_tab.render() 
            # 設備卸載評估
            evaluate_tab.render()
            # 設備即時資訊
            cabinets_tab.render()
            # 輸出與測試
            output_tab.render()

    return demo


# -------------------------
# 啟動
# -------------------------
demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
