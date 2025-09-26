
from typing import Dict, List, Tuple
from math import isfinite

def _safe(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def estimate_unload_time_and_energy(cab: dict, cfg: dict) -> Tuple[float, float, Dict[str, str]]:
    ttype = cab.get("type")
    th = cfg["thresholds"][ttype]
    defrost = cfg["defrost"]

    # 殘餘溫差（愈小愈危險）
    gaps = []
    reasons = []

    air_ret = cab.get("air_return_c")
    if air_ret is not None:
        gap_air = th["air_return_c_max"] - air_ret if ttype == "refrigerator" else abs(th["air_return_c_max"] - air_ret)
        gaps.append(gap_air)
        reasons.append(f"air_gap={gap_air:.2f}C")

    if ttype == "refrigerator":
        if cab.get("prod_t_milk_c") is not None:
            gap_milk = th["milk_surface_c_max"] - cab["prod_t_milk_c"]
            gaps.append(gap_milk)
            reasons.append(f"milk_gap={gap_milk:.2f}C")
        if cab.get("prod_t_mw_chill_c") is not None:
            gap_chill = th["chill_mw_surface_c_max"] - cab["prod_t_mw_chill_c"]
            gaps.append(gap_chill)
            reasons.append(f"chill_gap={gap_chill:.2f}C")
    else:
        if cab.get("prod_t_mw_freeze_c") is not None:
            # 對冷凍，門檻 -12，比現在溫度（如 -16）接近 0 表示餘裕大，差值用 (prod - limit) 的絕對值估幅度
            gap_fz = abs(th["mw_freeze_surface_c_max"] - cab["prod_t_mw_freeze_c"])
            gaps.append(gap_fz)
            reasons.append(f"fz_gap={gap_fz:.2f}C")

    # 取最保守的 gap（最小者）
    if not gaps:
        # 沒資料就假設零餘裕
        min_gap = 0.0
    else:
        min_gap = min(gaps)

    # 升溫速率（保守取 rise_c_per_min_max）
    rise_rate = th["rise_c_per_min_max"]
    unload_time_min = max(0.0, min_gap / rise_rate) if rise_rate > 0 else 0.0

    # 除霜懲罰（剛除霜仍可納入，但縮減時間）
    if int(cab.get("defrost_status") or 0) == 1 and int(cab.get("time_since_defrost_min") or 0) <= int(defrost["grace_min"]):
        unload_time_min *= float(defrost["penalty_factor"])
        reasons.append("DefrostGraceApplied")

    # 粗估能量（示意：冷凍 > 冷藏）
    p_eq = 1.2 if ttype == "freezer" else 1.0
    unload_energy_kwh = p_eq * (unload_time_min / 60.0)
    return unload_time_min, unload_energy_kwh, {"reason": ";".join(reasons)}

def risk_level_and_score(cab: dict, cfg: dict) -> Tuple[str, float]:
    ttype = cab.get("type")
    th = cfg["thresholds"][ttype]
    # 粗略風險：距離任一商品門檻 < 1C 視為高；冷凍較低
    high = False
    if ttype == "refrigerator":
        if cab.get("prod_t_milk_c") is not None:
            high |= (th["milk_surface_c_max"] - cab["prod_t_milk_c"]) < 1.0
        if cab.get("prod_t_mw_chill_c") is not None:
            high |= (th["chill_mw_surface_c_max"] - cab["prod_t_mw_chill_c"]) < 1.0
    else:
        # 冷凍距離 -12 小於 1C 視為中
        if cab.get("prod_t_mw_freeze_c") is not None:
            high |= abs(th["mw_freeze_surface_c_max"] - cab["prod_t_mw_freeze_c"]) < 1.0

    if ttype == "freezer":
        # baseline 低風險
        return ("low_freezer", 0.2 if not high else 0.5)
    else:
        return ("high_milk" if high else "medium_chill", 1.0 if high else 0.5)

def prioritize(cabinets: List[dict], business_flag: int, cfg: dict) -> List[dict]:
    # 先計算每台的時間與能量
    results = []
    times, energies = [], []
    for cab in cabinets:
        t, e, info = estimate_unload_time_and_energy(cab, cfg)
        risk_name, risk_score = risk_level_and_score(cab, cfg)
        res = {
            "cabinet_id": cab["cabinet_id"],
            "type": cab["type"],
            "unloadable_time_min": round(t, 2),
            "unloadable_energy_kWh": round(e, 3),
            "risk_level": risk_name,
            "risk_score": risk_score,
            "defrost_penalty": 1.0 if (int(cab.get("defrost_status") or 0) == 1 and int(cab.get("time_since_defrost_min") or 0) <= int(cfg["defrost"]["grace_min"])) else 0.0,
            "reason_codes": []
        }
        if info.get("reason"):
            res["reason_codes"] = info["reason"].split(";")
        results.append(res)
        times.append(t); energies.append(e)

    # 標準化
    t_max = max(times) if times else 1.0
    e_max = max(energies) if energies else 1.0

    w = cfg["weights"]
    for r in results:
        t_norm = (r["unloadable_time_min"] / t_max) if t_max > 0 else 0.0
        e_norm = (r["unloadable_energy_kWh"] / e_max) if e_max > 0 else 0.0
        risk_norm = min(1.0, max(0.0, r["risk_score"]))  # 0~1
        open_flag = 1.0 if int(business_flag) == 1 else 0.0
        dload = 0.0  # 未估
        defrost_pen = 1.0 if r["defrost_penalty"] > 0 else 0.0

        score = (
            w["w_time"] * t_norm
            + w["w_energy"] * e_norm
            - w["w_risk"] * risk_norm
            + w["w_open"] * open_flag
            + w["w_dload"] * dload
            - w["w_defrost"] * defrost_pen
        )
        r["priority_score"] = round(float(score), 4)

    # 排序
    results_sorted = sorted(results, key=lambda x: x["priority_score"], reverse=True)
    return results_sorted
