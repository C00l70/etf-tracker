#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一次性历史回填脚本：为「沪深300 ETF 多周期追踪系统」生成约 2 年（500 交易日）的 data.json 冷启动数据。

说明：
- 指数：akshare 拉取 sh000300 最近 N 条真实日线收盘价与日期。
- 8 只 ETF 总规模：仅「最后一条（最近交易日）」使用你设定的真实值；更早日为模拟。
  模拟逻辑：规模变动与当日指数涨跌幅适度正相关，再叠加随机申赎噪声；全序列不低于「真实规模×保底比例」。
- 总份额（亿份）：末日为你设定的真实 8 只合计；历史由两部分合成：
  1) 骨架：份额 ≈ 规模 / 有效净值，且有效净值随指数点位同向变动（与 scraper 中「规模/份额≈净值」一致）；
  2) 逆向申赎：从末日倒推，指数大跌日倾向净申购（份额较前一日上升），快速上涨日倾向净赎回；
     再与骨架按权重混合，避免规模与份额隐含净值偏离过大。

使用前请确认已安装：pip install akshare pandas
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

# -----------------------------------------------------------------------------
# 请在运行前修改：最近交易日 8 只 ETF 总规模、总份额（与当前真实数据一致）
# -----------------------------------------------------------------------------
REAL_TODAY_ETF_TOTAL: float = 5345.50
# 8 只沪深300 ETF 总份额合计，单位「亿份」（请按你手动核对或 scraper 最新输出填写）
REAL_TODAY_ALL8_SHARES_YI: float = 1250.75

# 回填交易日数量（约 2 年；A 股约 250 日/年 × 2）
N_TRADING_DAYS: int = 500

# 模拟规模下限 = REAL_TODAY_ETF_TOTAL × 该比例（防止早期数据跌穿合理区间）
SCALE_FLOOR_RATIO: float = 0.30

# 模拟份额下限 = REAL_TODAY_ALL8_SHARES_YI × 该比例
SHARE_FLOOR_RATIO: float = 0.35

# 指数联动：从「第 i 日→第 i+1 日」的规模变化（亿元）≈ BETA × 指数日收益率 × S[i+1](亿) + 噪声
INDEX_TO_SCALE_BETA: float = 0.38

# 随机申赎噪声（亿元），闭区间
FLOW_NOISE_MIN_YI: float = -10.0
FLOW_NOISE_MAX_YI: float = 10.0

# 逆向申赎：份额日变动对指数日收益率 r 的敏感度（正向 r 往往伴随赎回倾向，份额增速受抑）
# 正向模拟：shares[i+1] ≈ shares[i] * (1 - CONTRA_BETA*r) + 噪声；由末日倒推 shares[i]
CONTRA_BETA: float = 0.85

# 份额日噪声（亿份）：绝对值 + 相对末日份额的比例噪声
SHARE_NOISE_ABS_YI: float = 1.8
SHARE_NOISE_REL_FRAC: float = 0.0025

# 骨架与逆向模拟的混合权重（0=纯逆向模拟，1=纯指数挂钩骨架）；推荐 0.35~0.55
SHARE_BACKBONE_BLEND: float = 0.45

# 指数日涨跌幅截断（防止数据源异常单日导致跳变失真）
MAX_ABS_DAILY_RETURN: float = 0.12

# 输出文件（与脚本同目录，即项目根）
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(_BASE, "data.json")

TOP_TWO_CODES = ("510300", "510310")
ETF_CODES_ORDER = [
    "510300",
    "510310",
    "510330",
    "159919",
    "515330",
    "159925",
    "510360",
    "510350",
]

_ETF_WEIGHTS: Dict[str, float] = {
    "510300": 0.3726,
    "510310": 0.2550,
    "510330": 0.1683,
    "159919": 0.1724,
    "515330": 0.0156,
    "159925": 0.0047,
    "510360": 0.0054,
    "510350": 0.0060,
}


def _normalize_weights() -> Dict[str, float]:
    """将权重归一化，保证与 8 只代码一一对应且和为 1。"""
    s = sum(_ETF_WEIGHTS[c] for c in ETF_CODES_ORDER)
    return {c: _ETF_WEIGHTS[c] / s for c in ETF_CODES_ORDER}


def _fetch_hs300_last_n_days(n: int) -> Tuple[List[str], List[float]]:
    """
    使用 akshare 获取沪深300指数 sh000300 最近 n 个交易日的日期与收盘价。

    返回：
        dates:  YYYY-MM-DD，时间正序（从旧到新）
        closes: 与 dates 对齐的收盘价（float）

    异常时抛出，由 main 统一捕获打印。
    """
    import akshare as ak
    import pandas as pd

    df = ak.stock_zh_index_daily(symbol="sh000300")
    if df is None or df.empty:
        raise RuntimeError("akshare 返回的指数日线为空")

    if len(df) < n:
        raise ValueError(
            f"指数历史不足 {n} 个交易日，当前仅有 {len(df)} 条，请减小 N_TRADING_DAYS"
        )

    tail = df.iloc[-n:].copy()
    tail["date"] = pd.to_datetime(tail["date"], errors="coerce")
    tail = tail.dropna(subset=["date"])
    tail["close"] = pd.to_numeric(tail["close"], errors="coerce")
    tail = tail.dropna(subset=["close"])
    if len(tail) < n:
        raise ValueError(f"清洗后有效行数 {len(tail)} 小于 {n}")

    dates = tail["date"].dt.strftime("%Y-%m-%d").tolist()
    closes = [float(x) for x in tail["close"].tolist()]
    return dates, closes


def _daily_index_return(closes: List[float], i: int) -> float:
    """
    计算从交易日 i 到 i+1 的指数简单收益率 (close[i+1]/close[i] - 1)。
    对极端值做截断，避免脏数据拉爆模拟规模。
    """
    c0, c1 = closes[i], closes[i + 1]
    if c0 is None or c0 <= 0:
        return 0.0
    r = (c1 / c0) - 1.0
    if r > MAX_ABS_DAILY_RETURN:
        return MAX_ABS_DAILY_RETURN
    if r < -MAX_ABS_DAILY_RETURN:
        return -MAX_ABS_DAILY_RETURN
    return float(r)


def _simulate_etf_totals_yi_correlated(
    real_last: float,
    closes: List[float],
) -> List[float]:
    """
    从最近一日真实规模出发，按时间倒推生成与指数日涨跌「适度正相关」的规模序列。

    时间正序下标 0 为最旧，n-1 为最新；保证 totals[-1] == real_last。
    """
    n = len(closes)
    if n < 2:
        return [float(real_last)] if n == 1 else []

    floor_yi = max(float(real_last) * SCALE_FLOOR_RATIO, 1.0)
    out = [0.0] * n
    out[-1] = float(real_last)

    for i in range(n - 2, -1, -1):
        r = _daily_index_return(closes, i)
        s_next = out[i + 1]
        delta_yi = INDEX_TO_SCALE_BETA * r * s_next + random.uniform(
            FLOW_NOISE_MIN_YI, FLOW_NOISE_MAX_YI
        )
        s_i = s_next - delta_yi
        if s_i < floor_yi:
            s_i = floor_yi
        out[i] = s_i

    return out


def _backbone_shares_yi(
    scale_yi: List[float],
    closes: List[float],
    real_scale: float,
    real_shares: float,
) -> List[float]:
    """
    骨架份额（亿份）：隐含有效净值与指数同向变动，且强制末日与 (real_scale, real_shares) 一致。

    满足 shares_i = scale_i * real_shares * I_end / (real_scale * I_i)，
    即 shares_i / scale_i ∝ 1/I_i，对应「净值≈k×指数」时份额与规模在剔除净值变动后的关系。
    """
    n = len(closes)
    if n == 0:
        return []
    i_end = closes[-1]
    if i_end <= 0:
        raise ValueError("指数末日收盘无效")
    rs = float(real_scale)
    rv = float(real_shares)
    if rs <= 0 or rv <= 0:
        raise ValueError("真实规模或份额须为正")

    out: List[float] = []
    for i in range(n):
        ci = closes[i]
        if ci <= 0:
            out.append(max(rv * SHARE_FLOOR_RATIO, 1.0))
            continue
        sh = scale_yi[i] * rv * i_end / (rs * ci)
        out.append(float(sh))
    return out


def _simulate_shares_yi_contrarian_backward(
    real_last_shares: float,
    closes: List[float],
) -> List[float]:
    """
    从末日真实份额倒推：与指数涨跌反向相关的申赎冲击。

    正向关系（i→i+1）：shares[i+1] ≈ shares[i] * (1 - CONTRA_BETA*r) + noise
    - r<0 大跌：因子 >1，份额上升（抄底申购）
    - r>0 大涨：因子 <1，份额易降（获利赎回）

    倒推：shares[i] = (shares[i+1] - noise) / (1 - CONTRA_BETA*r)，并对分母做下限保护。
    """
    n = len(closes)
    if n < 1:
        return []
    if n == 1:
        return [float(real_last_shares)]

    floor_sh = max(float(real_last_shares) * SHARE_FLOOR_RATIO, 1.0)
    out = [0.0] * n
    out[-1] = float(real_last_shares)

    for i in range(n - 2, -1, -1):
        r = _daily_index_return(closes, i)
        sh_next = out[i + 1]
        noise_fwd = random.uniform(-SHARE_NOISE_ABS_YI, SHARE_NOISE_ABS_YI)
        noise_fwd += random.uniform(-SHARE_NOISE_REL_FRAC, SHARE_NOISE_REL_FRAC) * sh_next

        denom = 1.0 - CONTRA_BETA * r
        if denom < 0.2:
            denom = 0.2

        sh_i = (sh_next - noise_fwd) / denom
        if sh_i < floor_sh:
            sh_i = floor_sh
        out[i] = sh_i

    return out


def _blend_shares_series(
    sim: List[float],
    backbone: List[float],
    lam: float,
) -> List[float]:
    """凸组合混合；sim[-1] 与 backbone[-1] 均等于真实末日份额时，末日严格不变。"""
    if not sim or not backbone or len(sim) != len(backbone):
        raise ValueError("sim/backbone 长度不一致或为空")
    l = max(0.0, min(1.0, float(lam)))
    return [(1.0 - l) * sim[i] + l * backbone[i] for i in range(len(sim))]


def _build_per_code_yi(all8: float, weights: Dict[str, float]) -> Dict[str, float]:
    """按固定权重将总规模拆到 8 只代码（亿元，保留 6 位小数）。"""
    return {c: round(all8 * weights[c], 6) for c in ETF_CODES_ORDER}


def _build_per_code_shares_yi(all8_sh: float, weights: Dict[str, float]) -> Dict[str, float]:
    """按相同权重将总份额拆到 8 只代码（亿份）。"""
    return {c: round(all8_sh * weights[c], 6) for c in ETF_CODES_ORDER}


def _build_history_records(
    dates: List[str],
    closes: List[float],
    etf_totals_yi: List[float],
    etf_shares_yi: List[float],
) -> List[Dict[str, Any]]:
    """
    合并指数真实值 + 模拟规模 + 模拟份额，生成 history 列表（时间正序）。

    字段与 scraper 对齐：含 all8_shares_yi、top2_shares_yi、other6_shares_yi、etf_per_code_shares_yi。
    """
    n = len(dates)
    if len(closes) != n or len(etf_totals_yi) != n or len(etf_shares_yi) != n:
        raise ValueError("dates / closes / etf_totals_yi / etf_shares_yi 长度不一致")

    weights = _normalize_weights()
    records: List[Dict[str, Any]] = []

    for i in range(n):
        all8 = etf_totals_yi[i]
        all8_sh = etf_shares_yi[i]
        per = _build_per_code_yi(all8, weights)
        per_sh = _build_per_code_shares_yi(all8_sh, weights)
        top2 = sum(per[c] for c in TOP_TWO_CODES)
        other6 = round(all8 - top2, 6)
        top2_sh = sum(per_sh[c] for c in TOP_TWO_CODES)
        other6_sh = round(all8_sh - top2_sh, 6)

        if i < n - 1:
            warn = "历史模拟数据"
        else:
            warn = "✅ 系统初始化完成"

        records.append(
            {
                "date": dates[i],
                "hs300_close": round(float(closes[i]), 4),
                "all8_scale_yi": round(float(all8), 6),
                "top2_scale_yi": round(float(top2), 6),
                "other6_scale_yi": other6,
                "all8_shares_yi": round(float(all8_sh), 6),
                "top2_shares_yi": round(float(top2_sh), 6),
                "other6_shares_yi": other6_sh,
                "etf_per_code_yi": per,
                "etf_per_code_shares_yi": per_sh,
                "periods": {},
                "warning": warn,
            }
        )

    return records


def _build_payload(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """与现有 scraper 输出结构兼容的顶层 JSON。"""
    return {
        "meta": {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "periods_days": [1, 5, 13, 30, 60],
            "trend_epsilon_pct": 0.02,
            "backfill": True,
            "backfill_note": (
                f"最近 {N_TRADING_DAYS} 交易日：指数为 akshare 真实行情；"
                "ETF 总规模除末日为设定值外均为模拟（与指数日涨跌适度正相关 + 亿元级噪声，"
                f"下限为真实规模×{SCALE_FLOOR_RATIO:.0%}）。"
                f"总份额除末日为设定值（{REAL_TODAY_ALL8_SHARES_YI} 亿份）外，由指数挂钩骨架与逆向申赎模拟混合生成"
                f"（混合权重骨架={SHARE_BACKBONE_BLEND:.2f}），分项按固定权重拆分。"
            ),
        },
        "history": history,
    }


def main() -> int:
    print("=" * 60)
    print("data.json 冷启动回填（约 2 年 · 指数真实 + 规模/份额模拟）")
    print(f"输出路径: {DATA_JSON_PATH}")
    print(f"末日 8 只总规模 REAL_TODAY_ETF_TOTAL = {REAL_TODAY_ETF_TOTAL} 亿元")
    print(f"末日 8 只总份额 REAL_TODAY_ALL8_SHARES_YI = {REAL_TODAY_ALL8_SHARES_YI} 亿份")
    print(f"规模保底 ≈ {REAL_TODAY_ETF_TOTAL * SCALE_FLOOR_RATIO:.2f} 亿元")
    print(f"份额保底 ≈ {REAL_TODAY_ALL8_SHARES_YI * SHARE_FLOOR_RATIO:.2f} 亿份")
    print("=" * 60)

    if REAL_TODAY_ETF_TOTAL <= 0 or REAL_TODAY_ALL8_SHARES_YI <= 0:
        print("[错误] REAL_TODAY_ETF_TOTAL 与 REAL_TODAY_ALL8_SHARES_YI 必须为正。", file=sys.stderr)
        return 1

    try:
        dates, closes = _fetch_hs300_last_n_days(N_TRADING_DAYS)
    except ImportError as exc:
        print(f"[错误] 未安装 akshare 或 pandas: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 获取指数历史失败（网络或 akshare 接口异常）: {exc}", file=sys.stderr)
        return 1

    try:
        etf_yi = _simulate_etf_totals_yi_correlated(REAL_TODAY_ETF_TOTAL, closes)
        backbone_sh = _backbone_shares_yi(
            etf_yi, closes, REAL_TODAY_ETF_TOTAL, REAL_TODAY_ALL8_SHARES_YI
        )
        sim_sh = _simulate_shares_yi_contrarian_backward(
            REAL_TODAY_ALL8_SHARES_YI, closes
        )
        etf_sh_yi = _blend_shares_series(
            sim_sh, backbone_sh, SHARE_BACKBONE_BLEND
        )
        # 数值稳定：末日强制等于设定值（混合后应已一致，防浮点误差）
        etf_sh_yi[-1] = float(REAL_TODAY_ALL8_SHARES_YI)

        history = _build_history_records(dates, closes, etf_yi, etf_sh_yi)
        payload = _build_payload(history)
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 组装数据失败: {exc}", file=sys.stderr)
        return 1

    try:
        with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[错误] 写入文件失败: {exc}", file=sys.stderr)
        return 1

    last = history[-1]
    print(f"已写入 {len(history)} 条记录（日期范围 {history[0]['date']} ~ {history[-1]['date']}）")
    print(
        f"校验末日: all8_shares_yi={last['all8_shares_yi']} （应等于 {REAL_TODAY_ALL8_SHARES_YI}）"
    )
    print("建议随后执行 python scraper.py，刷新末日真实分项与多周期 periods。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
