import gradio as gr
import re
import threading, time
from src.ui.helpers import DEFAULT_CONFIG_YAML, load_yaml_from_file, validate_config_yaml, save_config_yaml
import os, json, yaml
from src.io.company_api import test_connection, fetch_stores, fetch_devices, fetch_realtime, save_api_settings
from datetime import datetime, timezone
from typing import List
import threading, time

# ---- 背景抓取（不依賴前端 Timer）----
_BG_FETCH_RUNNING = False
_BG_FETCH_THREAD: threading.Thread | None = None
_BG_FETCH_ARGS = {}  # 保存最近一次勾選時的 table_rows 與 API/Mapping 參數
# 自動啟動（載入就跑），預設開；CFG_AUTO_FREQ_MIN 預設 1 分鐘
_CFG_AUTO_ENABLE = os.environ.get("CFG_AUTO_FETCH", "1") == "1"
try:
    _CFG_AUTO_FREQ_MIN = max(1, int(os.environ.get("CFG_AUTO_FREQ_MIN", "1") or 1))
except Exception:
    _CFG_AUTO_FREQ_MIN = 1


def render():
    with gr.Tab("基本資料與閥值設定"):
        # -----------------------------
        # v0.5.0 基本設定（config.yaml）    
        # -----------------------------
        with gr.Accordion("基本設定（config.yaml）", open=False):
            with gr.Row():
                cfg_file = gr.File(label="上傳 config.yaml", file_types=[".yaml", ".yml"])
                cfg_text = gr.Textbox(label="config.yaml（可編輯）", value=DEFAULT_CONFIG_YAML, lines=22)
            with gr.Row():
                btn_load = gr.Button("從檔案載入→文字框")
                btn_validate = gr.Button("檢核 YAML")
                btn_save = gr.Button("儲存到伺服器 ./config.yaml")
            cfg_result = gr.JSON(label="結果（驗證 / 測試用）")
            cfg_save_status = gr.Textbox(label="儲存狀態", interactive=False)
            btn_load.click(load_yaml_from_file, inputs=cfg_file, outputs=cfg_text)
            btn_validate.click(validate_config_yaml, inputs=cfg_text, outputs=cfg_result)
            btn_save.click(save_config_yaml, inputs=cfg_text, outputs=cfg_save_status)

        # -----------------------------
        # v0.6.0 公司 API 設定（Accordion）
        # 目的：設定公司 API → 取門市/設備 → 抓即時資料 → 產生 input/latest/<Store_ID>.json
        # -----------------------------
        with gr.Accordion("公司 API 設定", open=True):
            # === 連線與認證 ===
            with gr.Row():
                base_url = gr.Textbox(label="Base URL（例：http://192.168.0.10:8080）",
                                    value=os.environ.get("COMPANY_API_BASE_URL","http://127.0.0.1:9000"))
                auth_type = gr.Dropdown(["bearer","api_key"], value="api_key", label="Auth Type")
                token = gr.Textbox(label="Bearer Token（選填；可用環境變數）", type="password")
            with gr.Row():
                api_key_header = gr.Textbox(label="API Key Header（api_key 模式）", value="X-API-Key")
                api_key = gr.Textbox(label="API Key（api_key 模式）", type="password")
                timeout_sec = gr.Number(label="Timeout (sec)", value=10, precision=0)
                verify_ssl = gr.Checkbox(label="Verify SSL", value=True)

            # === 端點路徑（可依公司 API 調整）===
            gr.Markdown("**端點路徑**（可依實際 API 調整）")
            with gr.Row():
                ep_stores = gr.Textbox(label="/stores", value="/stores")
                ep_devices = gr.Textbox(label="/stores/{store_id}/devices", value="/stores/{store_id}/devices")
                ep_realtime = gr.Textbox(label="/stores/{store_id}/devices/{device_id}/realtime", value="/stores/{store_id}/devices/{device_id}/realtime")
                ep_post = gr.Textbox(label="POST /output/priority.json", value="/output/priority.json")

            # === 欄位映射（DotPath → 系統 devices 欄位）===
            gr.Markdown("**欄位映射（DotPath）→ 系統 devices 欄位**")
            with gr.Row():
                map_device_id = gr.Textbox(label="device_id", value="Device_ID")
                map_device_type = gr.Textbox(label="device_type", value="Device_type")
                map_temp = gr.Textbox(label="temp_current", value="Temp_current")
                map_troom = gr.Textbox(label="t_room", value="T_room")
                map_power = gr.Textbox(label="power", value="Power")
                map_defrost = gr.Textbox(label="defrost_status", value="Defrost_status")
                map_ts = gr.Textbox(label="timestamp", value="timestamp")

            with gr.Row():
                btn_test = gr.Button("測試連線")
                btn_save_api = gr.Button("保存 API 設定", variant="secondary")
                btn_load_api = gr.Button("載入 API 設定", variant="secondary")

            api_save_status = gr.Textbox(label="API 設定狀態", interactive=False)

            # === 門市 / 設備 / 即時資料 操作 ===
            with gr.Row():
                btn_list_stores = gr.Button("更新門市列表")
                btn_select_all = gr.Button("全選")
                btn_unselect_all = gr.Button("全不選")
            stores_table = gr.Dataframe(
                headers=["選取", "Store_ID", "名稱"],
                datatype=["bool", "str", "str"],
                interactive=True,
                row_count=(1, "dynamic"),
                col_count=(3, "fixed"),
                label="門市列表（可複選）",
                type="array"
            )
            save_stores_btn = gr.Button("儲存門市（寫入 input/config.json）")
            save_stores_status = gr.Textbox(label="儲存狀態", interactive=False)

            with gr.Row():
                btn_list_devices = gr.Button("取得設備清單")
                devices_json = gr.JSON(label="設備清單（原樣）")
            with gr.Row():
                btn_fetch_once = gr.Button("抓取一次（測試）")
                auto_fetch_enable = gr.Checkbox(label="自動抓取即時資料", value=True)
                auto_fetch_freq_min = gr.Number(label="抓取頻率（分鐘）", value=1, precision=0)  # 預設 1 分鐘
                latest_status = gr.Textbox(label="狀態", interactive=False)

            # -------- 後端背景抓取：不依賴前端 Timer，跨分頁也會持續 --------
            def _bg_fetch_loop():
                """依 freq_min 週期執行 on_fetch_selected_once。"""
                global _BG_FETCH_RUNNING, _BG_FETCH_ARGS
                while _BG_FETCH_RUNNING:
                    try:
                        args = dict(_BG_FETCH_ARGS or {})
                        table_rows = args.get("table_rows")
                        vals = args.get("vals", [])
                        _ = on_fetch_selected_once(table_rows, *vals)
                        freq_min = max(1, int(args.get("freq_min", 1)))
                        for _ in range(freq_min * 60):
                            if not _BG_FETCH_RUNNING:
                                break
                            time.sleep(1)
                    except Exception:
                        time.sleep(1)

            def _bg_fetch_start(freq_min, table_rows, *vals):
                global _BG_FETCH_RUNNING, _BG_FETCH_THREAD, _BG_FETCH_ARGS
                _BG_FETCH_ARGS = {"freq_min": max(1, int(freq_min or 1)), "table_rows": table_rows, "vals": vals}
                if not _BG_FETCH_RUNNING:
                    _BG_FETCH_RUNNING = True
                    _BG_FETCH_THREAD = threading.Thread(target=_bg_fetch_loop, daemon=True)
                    _BG_FETCH_THREAD.start()

            def _bg_fetch_stop():
                global _BG_FETCH_RUNNING
                _BG_FETCH_RUNNING = False

            # ---- Render 結束前：若啟用自動模式，載入就啟動背景抓取 ----
            if _CFG_AUTO_ENABLE:
                try:
                    # 以 config.json 內的 Selected_Store_IDs 為準，不強制更新門市表
                    # 讓 on_fetch_selected_once 走既有選店
                    dummy_table = []  # 不需要真的表格資料
                    _bg_fetch_start(_CFG_AUTO_FREQ_MIN, dummy_table)
                except Exception as e:
                    print(f"[config_tab] 背景抓取自啟動失敗: {e}")

            # 內部狀態與計時器：每 60 秒 tick，達到指定分鐘才觸發
            auto_fetch_state = gr.State({"enabled": True, "freq_min": 1, "last_ts": 0.0})
            auto_fetch_timer = gr.Timer(60)  #原程式 gr.Timer(60)

            def on_auto_fetch_tick(state, table_rows, *vals):
                """Timer 每次觸發時檢查頻率；達到設定分鐘就把 Selected_Store_IDs 全部抓一輪"""
                import time
                st = dict(state or {})
                now = time.time()
                if not st.get("enabled"):
                    return st, gr.update()
                # last_ts==0 代表剛啟用 → 這個 tick 立刻跑一次
                if st.get("last_ts", 0) != 0 and (now - st["last_ts"] < (st.get("freq_min", 1) * 60)):
                    return st, gr.update()
                st["last_ts"] = now
                # 直接重用多店一次抓取
                msg = on_fetch_selected_once(table_rows, *vals)
                return st, msg

            # -------- 後端背景抓取：不依賴前端 Timer，跨分頁也會持續 --------
            def _bg_fetch_loop():
                """背景常駐迴圈：依 freq_min 週期執行 on_fetch_selected_once"""
                global _BG_FETCH_RUNNING, _BG_FETCH_ARGS
                while _BG_FETCH_RUNNING:
                    try:
                        args = dict(_BG_FETCH_ARGS or {})
                        table_rows = args.get("table_rows")
                        vals = args.get("vals", [])
                        # 抓一次（多店）
                        _ = on_fetch_selected_once(table_rows, *vals)
                        # 休眠 freq_min 分鐘（秒）
                        freq_min = max(1, int(args.get("freq_min", 1)))
                        for _ in range(freq_min * 60):
                            if not _BG_FETCH_RUNNING:
                                break
                            time.sleep(1)
                    except Exception:
                        # 靜默重試，避免偶發 I/O 使執行緒整體中止
                        time.sleep(1)

            def _bg_fetch_start(freq_min, table_rows, *vals):
                """啟動背景迴圈；保存最新參數供背景使用"""
                global _BG_FETCH_RUNNING, _BG_FETCH_THREAD, _BG_FETCH_ARGS
                _BG_FETCH_ARGS = {"freq_min": max(1, int(freq_min or 1)), "table_rows": table_rows, "vals": vals}
                if not _BG_FETCH_RUNNING:
                    _BG_FETCH_RUNNING = True
                    _BG_FETCH_THREAD = threading.Thread(target=_bg_fetch_loop, daemon=True)
                    _BG_FETCH_THREAD.start()

            def _bg_fetch_stop():
                """停止背景迴圈"""
                global _BG_FETCH_RUNNING
                _BG_FETCH_RUNNING = False



            # -------- 後端：彙整 UI 參數 --------
            def _collect_params(
                base_url_v, auth_type_v, token_v,
                api_key_header_v, api_key_v, timeout_sec_v, verify_ssl_v,
                ep_stores_v, ep_devices_v, ep_realtime_v, ep_post_v,
                map_device_id_v, map_device_type_v, map_temp_v, map_troom_v, map_power_v, map_defrost_v, map_ts_v
            ):
                return {
                    "base_url": base_url_v,
                    "auth_type": auth_type_v,
                    "token": token_v,
                    "api_key_header": api_key_header_v,
                    "api_key": api_key_v,
                    "timeout_sec": int(timeout_sec_v or 10),
                    "verify_ssl": bool(verify_ssl_v),
                    "endpoints": {
                        "list_stores": ep_stores_v,
                        "list_devices_of_store": ep_devices_v,
                        "realtime_of_device": ep_realtime_v,
                        "post_priority": ep_post_v,
                    },
                    "mapping": {
                        "device_id": map_device_id_v,
                        "device_type": map_device_type_v,
                        "temp_current": map_temp_v,
                        "t_room": map_troom_v,
                        "power": map_power_v,
                        "defrost_status": map_defrost_v,
                        "timestamp": map_ts_v,
                    },
                }

            # -------- 動作：測試 / 列門市 / 儲存門市 / 列設備 / 抓一次 / 自動抓取 --------
            def on_test(*vals):
                ok, info = test_connection(_collect_params(*vals))
                return {"api_test": {"ok": ok, "info": info}}

            def on_toggle_auto_fetch(enabled, freq_min, state, table_rows, *vals):
                """更新自動抓取開關與頻率（分鐘）；啟用時先跑一次「取得設備清單」＋「抓取一次」"""
                import time
                st = dict(state or {})
                st["enabled"] = bool(enabled)
                try:
                    st["freq_min"] = max(1, int(freq_min or 1))
                except Exception:
                    st["freq_min"] = 1
                # 啟用：先跑一次「取得設備清單」+ 立刻抓一次即時
                if st["enabled"]:
                    _ = on_list_devices_from_ui(table_rows, *vals)  # 先更新設備清單
                    msg = on_fetch_selected_once(table_rows, *vals)  # 立刻抓一次
                    # 啟動背景迴圈（不依賴前端 Timer）
                    _bg_fetch_start(st["freq_min"], table_rows, *vals)
                    st["last_ts"] = time.time()  # 保持原欄位，不影響舊 UI 顯示
                    return st, f"⏱️ 自動抓取：開啟（{st['freq_min']} 分）｜已執行一次；背景常駐中｜{msg}"
                else:
                    _bg_fetch_stop()
                    st["last_ts"] = time.time()
                    return st, f"⏱️ 自動抓取：關閉（已停止背景常駐）"

            # ---- v0.6.3: 門市多選（表格） ----
            def _make_store_table(items, selected_ids: List[str] | None = None):
                """把 /stores 回傳轉成 DataFrame 用的 list-of-lists：[[選取, Store_ID, 名稱], ...]"""
                selected = set(selected_ids or [])
                rows = []
                for it in (items or []):
                    sid = it.get("id") or it.get("Store_ID")
                    name = it.get("name") or f"店{sid[1:]}" if sid else ""
                    rows.append([sid in selected, sid, name])
                return rows

            def _toggle_all(table_rows, value: bool):
                """將第一欄選取設成同一值"""
                return [[bool(value), r[1], r[2]] for r in (table_rows or [])]
            
            #--- v 0.6.3 --
            CONFIG_JSON = os.path.join("input", "config.json")

            def _load_cfg() -> dict:
                try:
                    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
                        return json.load(f) or {}
                except Exception:
                    return {}

            def _save_cfg(cfg: dict):
                """寫入 input/config.json：
                - 其他欄位：indent=2 美化
                - 陣列欄位（All_Store_IDs, Selected_Store_IDs）：單行顯示
                """
                os.makedirs("input", exist_ok=True)
                s = json.dumps(cfg, ensure_ascii=False, indent=2)
                s = _inline_array_field(s, "All_Store_IDs", cfg.get("All_Store_IDs"))
                s = _inline_array_field(s, "Selected_Store_IDs", cfg.get("Selected_Store_IDs"))
                with open(CONFIG_JSON, "w", encoding="utf-8") as f:
                    f.write(s)

            def _inline_array_field(s: str, key: str, arr) -> str:
                """把 JSON 字串 s 中的某個陣列欄位壓成單行（保留其他縮排）"""
                inline = json.dumps(arr or [], ensure_ascii=False, separators=(",", ":"))
                pattern = rf'("{re.escape(key)}"\s*:\s*)\[(?:.|\n)*?\]'
                return re.sub(pattern, rf'\1{inline}', s, flags=re.DOTALL)

            def _get_selected_ids_from_table(table_rows) -> List[str]:
                """把 Dataframe 的 rows 轉成選中的 store id 陣列"""
                try:
                    import pandas as pd
                    if isinstance(table_rows, pd.DataFrame):
                        table_rows = table_rows.values.tolist()
                except Exception:
                    pass
                return [r[1] for r in (table_rows or []) if r and r[0] is True and r[1]]

            def _load_selected_store_ids() -> List[str]:
                cfg = _load_cfg()
                return list(cfg.get("Selected_Store_IDs") or [])

#            def _load_all_store_ids() -> List[str]:
#                cfg = _load_cfg()
#                return list(cfg.get("All_Store_IDs") or [])
            
           
            def on_list_devices_from_ui(table_rows, *vals):
                selected = _get_selected_ids_from_table(table_rows)
                if not selected:
                    selected = _load_selected_store_ids()
                if not selected:
                    return {"error": "尚未選擇任何門市（Selected_Store_IDs 為空）"}
                params = _collect_params(*vals)
                out = {}
                for sid in selected:
                    try:
                        out[sid] = fetch_devices(params, sid)
                    except Exception as e:
                        out[sid] = {"error": str(e)}
                return {"stores": out}

            def _save_selected_stores(table_rows):
                """將勾選的門市寫入 input/config.json：Selected_Store_IDs"""
                try:
                    import pandas as pd
                    if isinstance(table_rows, pd.DataFrame): table_rows = table_rows.values.tolist()
                except Exception: pass
                os.makedirs("input", exist_ok=True)
                selected = [r[1] for r in (table_rows or []) if r and r[0] is True and r[1]]
                if not selected:
                    return "⚠️ 請先在表格勾選至少 1 間門市"

                cfg = _load_cfg()
                cfg["Selected_Store_IDs"] = selected
                _save_cfg(cfg)
                return f"✅ 已儲存 {len(selected)} 間門市（Selected_Store_IDs）"
            
            def _save_api_settings(table_rows, *vals):
                """保存 API 設定到 config.yaml"""
                try:
                    params = _collect_params(*vals)
                    
                    # 構建要保存的配置
                    config_data = {
                        "api": {
                            "base_url": params["base_url"],
                            "timeout_sec": params["timeout_sec"],
                            "verify_ssl": params["verify_ssl"],
                            "auth": {
                                "type": params["auth_type"],
                                "api_key_header": params["api_key_header"]
                            },
                            "endpoints": params["endpoints"]
                        },
                        "mapping": params["mapping"]
                    }
                    
                    # 保存到 config/config.yaml
                    os.makedirs("config", exist_ok=True)
                    with open("config/config.yaml", "w", encoding="utf-8") as f:
                        yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)
                    
                    return "✅ API 設定已保存到 config/config.yaml"
                except Exception as e:
                    return f"❌ 保存失敗：{e}"

            def _load_api_settings():
                """從 config.yaml 載入 API 設定"""
                try:
                    params = _load_api_params_from_yaml()
                    if not params:
                        return {}
                    
                    return {
                        "base_url": params.get("base_url", "http://127.0.0.1:9000"),
                        "auth_type": params.get("auth_type", "api_key"),
                        "api_key_header": params.get("api_key_header", "X-API-Key"),
                        "timeout_sec": params.get("timeout_sec", 10),
                        "verify_ssl": params.get("verify_ssl", True),
                        "endpoints": params.get("endpoints", {}),
                        "mapping": params.get("mapping", {})
                    }
                except Exception:
                    return {}

            # 綁定：更新 → 表格、全選/全不選、儲存
            def on_list_stores(*vals):
                items = fetch_stores(_collect_params(*vals))  # 打 /stores
                all_ids = [it.get("id") or it.get("Store_ID") for it in (items or []) if (it.get("id") or it.get("Store_ID"))]
                # 覆寫 All_Store_IDs，Selected_Store_IDs 取交集保留舊選擇
                cfg = _load_cfg()
                old_selected = set(cfg.get("Selected_Store_IDs") or [])
                cfg["All_Store_IDs"] = all_ids
                cfg["Selected_Store_IDs"] = [sid for sid in all_ids if sid in old_selected]
                _save_cfg(cfg)
                # 回表格（第一欄勾選）
                return _make_store_table(items, cfg["Selected_Store_IDs"])

            # ---- v 0.6.5 ----
            def _to_naive_seconds_iso(ts: str | None) -> str | None:
                """把任何 ISO（可含/不含 tz）→ 去 tz、去微秒，保留到秒：YYYY-MM-DDTHH:MM:SS"""
                if not ts:
                    return None
                try:
                    t = datetime.fromisoformat(ts)
                    if t.tzinfo is not None:
                        # 轉成 UTC 再拿掉 tz，避免 +08:00 / +00:00 尾巴
                        t = t.astimezone(timezone.utc).replace(tzinfo=None)
                    return t.replace(microsecond=0).isoformat(timespec="seconds")
                except Exception:
                    # 非標準格式就原樣回傳（避免整筆丟失）
                    return ts

            def _fetch_realtime_for_store(params, store_id_val: str) -> tuple[int, int]:
                """抓某店→寫快照，回傳 (設備清單數, 即時資料數)"""
                devs = fetch_devices(params, store_id_val)
                device_ids = [d.get("id") or d.get("Device_ID") for d in (devs or [])]
                # 可能遇到 mock 店級無資料 → 空清單
                devices = fetch_realtime(params, store_id_val, device_ids, params["mapping"]) if device_ids else []
                # 正規化每台設備的 timestamp：去 tz、到秒
                for d in (devices or []):
                    if isinstance(d, dict) and "timestamp" in d:
                        d["timestamp"] = _to_naive_seconds_iso(d.get("timestamp"))
                os.makedirs(os.path.join("input","latest"), exist_ok=True)
                path = os.path.join("input","latest", f"{store_id_val}.json")
                tmp_path = path + ".tmp"
                generated_at = datetime.now().replace(microsecond=0).isoformat(timespec="seconds")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"Store_ID": store_id_val, "generated_at": generated_at, "devices": devices},
                        f, ensure_ascii=False, indent=2
                    )
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                os.replace(tmp_path, path) # 原子替換，讀者不會看到半檔
                return (len(device_ids or []), len(devices or []))

            def on_fetch_selected_once(table_rows, *vals):
                selected = _get_selected_ids_from_table(table_rows)
                if not selected:
                    selected = _load_selected_store_ids()
                if not selected:
                    return "⚠️ 尚未選擇任何門市（Selected_Store_IDs 為空）"
                params = _collect_params(*vals)
                total_rt, total_list, detail = 0, 0, []
                for sid in selected:
                    n_list, n_rt = _fetch_realtime_for_store(params, sid)
                    total_list += n_list
                    total_rt += n_rt
                    detail.append(f"{sid}:{n_list}/{n_rt}")
                return (
                    "✅ 已更新 {stores} 店；各店（設備清單/即時）＝ {detail} ／"
                    "合計（清單/即時）＝ {tot_list}/{tot_rt}"
                ).format(stores=len(selected), detail=", ".join(detail), tot_list=total_list, tot_rt=total_rt)

            btn_select_all.click(lambda rows: _toggle_all(rows, True), inputs=stores_table, outputs=stores_table)
            btn_unselect_all.click(lambda rows: _toggle_all(rows, False), inputs=stores_table, outputs=stores_table)
            save_stores_btn.click(_save_selected_stores, inputs=stores_table, outputs=save_stores_status)

            # ---- 綁定事件 ----
            # 把所有參數元件都當成 inputs 傳入，確保使用者當下在 UI 的值會生效
            _param_inputs = [
                base_url, auth_type, token, api_key_header, api_key, timeout_sec, verify_ssl,
                ep_stores, ep_devices, ep_realtime, ep_post,
                map_device_id, map_device_type, map_temp, map_troom, map_power, map_defrost, map_ts
            ]
            btn_test.click(on_test, inputs=_param_inputs, outputs=cfg_result)
            btn_list_stores.click(on_list_stores, inputs=_param_inputs, outputs=stores_table)
            btn_list_devices.click(on_list_devices_from_ui, inputs=[stores_table, *_param_inputs], outputs=devices_json)
            btn_fetch_once.click(on_fetch_selected_once, inputs=[stores_table, *_param_inputs], outputs=latest_status)
            auto_fetch_timer.tick(on_auto_fetch_tick, inputs=[auto_fetch_state, stores_table, *_param_inputs], outputs=[auto_fetch_state, latest_status])
            auto_fetch_enable.change(on_toggle_auto_fetch,
                                    inputs=[auto_fetch_enable, auto_fetch_freq_min, auto_fetch_state, stores_table, *_param_inputs],
                                    outputs=[auto_fetch_state, latest_status])
            auto_fetch_freq_min.change(on_toggle_auto_fetch,
                                    inputs=[auto_fetch_enable, auto_fetch_freq_min, auto_fetch_state, stores_table, *_param_inputs],
                                    outputs=[auto_fetch_state, latest_status])
            
            btn_save_api.click(
                _save_api_settings,
                inputs=[stores_table, *_param_inputs],
                outputs=[api_save_status]
            )

            def on_load_api_settings():
                settings = _load_api_settings()
                if not settings:
                    return [gr.update() for _ in _param_inputs] + ["無可載入的設定"]
                
                endpoints = settings.get("endpoints", {})
                mapping = settings.get("mapping", {})
                
                return [
                    gr.update(value=settings.get("base_url", "")),
                    gr.update(value=settings.get("auth_type", "api_key")),
                    gr.update(),  # token 保持空白
                    gr.update(value=settings.get("api_key_header", "X-API-Key")),
                    gr.update(),  # api_key 保持空白
                    gr.update(value=settings.get("timeout_sec", 10)),
                    gr.update(value=settings.get("verify_ssl", True)),
                    gr.update(value=endpoints.get("list_stores", "/stores")),
                    gr.update(value=endpoints.get("list_devices_of_store", "/stores/{store_id}/devices")),
                    gr.update(value=endpoints.get("realtime_of_device", "/stores/{store_id}/devices/{device_id}/realtime")),
                    gr.update(value=endpoints.get("post_priority", "/output/priority.json")),
                    gr.update(value=mapping.get("device_id", "Device_ID")),
                    gr.update(value=mapping.get("device_type", "Device_type")),
                    gr.update(value=mapping.get("temp_current", "Temp_current")),
                    gr.update(value=mapping.get("t_room", "T_room")),
                    gr.update(value=mapping.get("power", "Power")),
                    gr.update(value=mapping.get("defrost_status", "Defrost_status")),
                    gr.update(value=mapping.get("timestamp", "timestamp")),
                    "✅ API 設定已載入"
                ]

            btn_load_api.click(
                on_load_api_settings,
                outputs=[*_param_inputs, api_save_status]
            )
