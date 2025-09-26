import gradio as gr
import pandas as pd
import json
import os
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any

# -----------------------------
# Paths (v0.6.0 multi-store)
# -----------------------------
CONFIG_PATH = os.path.join("input", "config.json")
LATEST_DIR = os.path.join("input", "latest")


# -----------------------------
# Data helpers
# -----------------------------
def _load_selected_store_ids() -> List[str]:
    """Read Selected_Store_IDs from config.json. Return [] if missing."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
        sids = cfg.get("Selected_Store_IDs")
        if isinstance(sids, list) and all(isinstance(s, str) for s in sids):
            return sids
        # fallback: if only All_Store_IDs exists, preselect all
        all_sids = cfg.get("All_Store_IDs")
        if isinstance(all_sids, list) and all(isinstance(s, str) for s in all_sids):
            return all_sids
    except Exception:
        pass
    return []


def _read_latest_snapshot(store_id: str) -> Tuple[str | None, list]:
    """Read input/latest/<sid>.json and return (generated_at, devices)."""
    if not store_id:
        return None, []
    path = os.path.join(LATEST_DIR, f"{store_id}.json")
    if not os.path.exists(path):
        return None, []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data.get("generated_at"), data.get("devices", [])
    except Exception:
        return None, []


def _devices_to_rows(devs: list, store_id: str) -> list[dict[str, Any]]:
    """Convert devices list to rows with normalized columns."""
    rows = []
    for d in devs or []:
        dtype = d.get("Device_type")
        temp = d.get("Temp_current") if dtype in ("冷凍", "冷藏", "開放櫃") else d.get("T_room")
        defrost = d.get("Defrost_status") if dtype in ("冷凍", "冷藏") else None
        rows.append({
            "Store_ID": store_id,
            "Device_ID": d.get("Device_ID"),
            "類型": dtype,
            "溫度(°C)": temp,
            "除霜": ("是" if defrost == 1 else ("否" if defrost in (0, None) else defrost)),
            "時間戳": d.get("timestamp"),
        })
    return rows


def _load_all_selected() -> Tuple[pd.DataFrame, Dict[str, str | None]]:
    """Load all selected stores' snapshots → merged DataFrame and per-store generated_at dict."""
    sids = _load_selected_store_ids()
    all_rows: list[dict[str, Any]] = []
    gen_map: Dict[str, str | None] = {}
    for sid in sids:
        gen_at, devs = _read_latest_snapshot(sid)
        gen_map[sid] = gen_at
        all_rows.extend(_devices_to_rows(devs, sid))
    df = pd.DataFrame(all_rows, columns=["Store_ID", "Device_ID", "類型", "溫度(°C)", "除霜", "時間戳"]) \
             .sort_values(by=["Store_ID", "溫度(°C)"], ascending=[True, False], na_position="last")
    return df, gen_map


def _last_update_text(gen_map: Dict[str, str | None]) -> str:
    times = [g for g in gen_map.values() if g]
    if not gen_map:
        return "尚未讀取（沒有選擇任何門市）"
    if not times:
        missing = [sid for sid, g in gen_map.items() if not g]
        return f"🗂️ 找不到快照：{', '.join(missing)}"

    # Parse to timezone-aware datetimes (normalize to UTC if naive)
    def _to_aware(dt: datetime) -> datetime:
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    dt_list = []
    for s in times:
        try:
            dt = datetime.fromisoformat(s)
            dt_list.append(_to_aware(dt).astimezone(timezone.utc))
        except Exception:
            pass
    if not dt_list:
        return f"最後更新（文字）：{min(times)} ~ {max(times)}"

    # Compare everything in UTC to avoid naive/aware mismatch
    tmin, tmax = min(dt_list), max(dt_list)
    now = datetime.now(timezone.utc)
    warn = " ⚠️（部份資料超過 3 分鐘）" if any((now - t) >= timedelta(minutes=3) for t in dt_list) else ""

    def _fmt(dt: datetime) -> str:
        # Show as ISO in UTC
        return dt.astimezone(timezone.utc).isoformat()

    return f"最後更新：{_fmt(tmin)} ~ {_fmt(tmax)}{warn}"


def _filter_df(df: pd.DataFrame, store_ids: List[str] | None, types: List[str] | None) -> pd.DataFrame:
    res = df
    if store_ids:
        res = res[res["Store_ID"].isin(store_ids)]
    if types:
        res = res[res["類型"].isin(types)]
    return res.sort_values(by=["Store_ID", "溫度(°C)"], ascending=[True, False], na_position="last")


# -----------------------------
# Gradio render
# -----------------------------
def render():
    with gr.Tab("設備現況分析資訊"):
        #gr.Markdown("### 門市設備即時監測 + 卸載分析結果（多店彙整）")

        # Load initial data
        df0, gen0 = _load_all_selected()
        all_store_choices = sorted(df0["Store_ID"].unique().tolist()) if not df0.empty else _load_selected_store_ids()
        type_choices = sorted([x for x in df0["類型"].dropna().unique().tolist()]) if not df0.empty else []

        with gr.Row():
            store_multi = gr.Dropdown(
                label="Store 篩選（多選）",
                choices=all_store_choices,
                value=all_store_choices,  # default: select all
                multiselect=True,
                interactive=True,
                allow_custom_value=False,
            )
            type_multi = gr.Dropdown(
                label="類型篩選（多選）",
                choices=type_choices,
                value=type_choices,  # default: select all types
                multiselect=True,
                interactive=True,
                allow_custom_value=False,
            )
            refresh_btn = gr.Button("重新整理（重讀檔案）")
            auto_update = gr.Checkbox(label="自動更新（每 60 秒）", value=False)

        last_update_md = gr.Markdown(_last_update_text(gen0) if gen0 else "尚未讀取")

        # Show table
        table = gr.Dataframe(
            headers=["Store_ID", "Device_ID", "類型", "溫度(°C)", "除霜", "時間戳"],
            value=_filter_df(df0, store_multi.value, type_multi.value) if not df0.empty else pd.DataFrame(columns=["Store_ID","Device_ID","類型","溫度(°C)","除霜","時間戳"]),
            interactive=False,
            row_count=(0 if df0.empty else (len(df0))),
            col_count=(6),
        )

        # Small stats line
        def _stats_text(df: pd.DataFrame) -> str:
            n_devices = len(df)
            n_stores = df["Store_ID"].nunique() if not df.empty else 0
            return f"共 {n_stores} 門市、{n_devices} 台設備"

        stats_md = gr.Markdown(_stats_text(_filter_df(df0, store_multi.value, type_multi.value) if not df0.empty else df0))

        # Handlers
        def _reload_and_filter(store_sel: List[str], type_sel: List[str]):
            df, gen_map = _load_all_selected()
            # Refresh dropdown choices in case config changed
            new_store_choices = sorted(df["Store_ID"].unique().tolist()) if not df.empty else _load_selected_store_ids()
            new_type_choices = sorted([x for x in df["類型"].dropna().unique().tolist()]) if not df.empty else []

            # If previous selections are empty, default to all; otherwise keep intersection
            store_sel_set = set(store_sel or new_store_choices)
            type_sel_set = set(type_sel or new_type_choices)
            store_val = sorted([s for s in new_store_choices if s in store_sel_set]) or new_store_choices
            type_val = sorted([t for t in new_type_choices if t in type_sel_set]) or new_type_choices

            df_show = _filter_df(df, store_val, type_val)

            return (
                gr.update(choices=new_store_choices, value=store_val),
                gr.update(choices=new_type_choices, value=type_val),
                _last_update_text(gen_map),
                df_show,
                _stats_text(df_show),
            )

        def _only_filter(store_sel: List[str], type_sel: List[str]):
            # Only use cached file data by reloading, in case files changed; keeps things simple & consistent
            df, gen_map = _load_all_selected()
            # Keep current choices; just enforce intersection to avoid invalid selections
            all_store_choices_local = sorted(df["Store_ID"].unique().tolist()) if not df.empty else []
            all_type_choices_local = sorted([x for x in df["類型"].dropna().unique().tolist()]) if not df.empty else []
            store_val = sorted([s for s in (store_sel or []) if s in all_store_choices_local]) or all_store_choices_local
            type_val = sorted([t for t in (type_sel or []) if t in all_type_choices_local]) or all_type_choices_local
            df_show = _filter_df(df, store_val, type_val)
            return (
                gr.update(value=store_val, choices=all_store_choices_local),
                gr.update(value=type_val, choices=all_type_choices_local),
                _last_update_text(gen_map),
                df_show,
                _stats_text(df_show),
            )

        # Bind events
        refresh_btn.click(
            _reload_and_filter,
            inputs=[store_multi, type_multi],
            outputs=[store_multi, type_multi, last_update_md, table, stats_md],
        )

        # Auto update timer
        timer = gr.Timer(60)
        def on_tick(auto: bool, store_sel: List[str], type_sel: List[str]):
            if not auto:
                return [gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]
            return _only_filter(store_sel, type_sel)

        timer.tick(
            on_tick,
            inputs=[auto_update, store_multi, type_multi],
            outputs=[store_multi, type_multi, last_update_md, table, stats_md],
        )

        # Also re-filter when user changes dropdowns (no file reload needed)
        store_multi.change(_only_filter, inputs=[store_multi, type_multi], outputs=[store_multi, type_multi, last_update_md, table, stats_md])
        type_multi.change(_only_filter, inputs=[store_multi, type_multi], outputs=[store_multi, type_multi, last_update_md, table, stats_md])

    return table
