#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
沪深300 ETF 规模追踪与多周期背离预警 — 数据采集与计算

数据来源（多路容错）：
1. akshare / 东方财富 ETF 行情（总市值、最新价、最新份额）
2. akshare 上交所、深交所 ETF 份额披露（仅补充行情表中缺失的标的）
3. 沪深300指数 sh000300 最新收盘价（akshare 日线，失败时尝试前值）

输出：
- etf_scale_history.csv：分项与合计（亿元）
- data.json：供 GitHub Pages 可视化；含规模(亿)、总份额(亿份)、各周期「份额 vs 指数」背离（保留约 2 年）

若某路接口失败，不中断程序；规模/份额缺失时优先沿用上日 JSON 分项，再无则份额可按规模估算或置 0。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# 配置：8 只主流沪深300 ETF（代码与公开交易信息一致）
# -----------------------------------------------------------------------------
@dataclass
class EtfTarget:
    """单只目标 ETF 的静态信息"""

    code: str  # 6 位交易代码
    name: str  # 展示名称（全称）
    exchange: str  # "SH" 上交所 或 "SZ" 深交所
    name_cn: str  # 表头用中文简称（一般为管理人/品牌简称）


TOP_TWO_CODES = ("510300", "510310")

TARGET_ETFS: List[EtfTarget] = [
    EtfTarget("510300", "华泰柏瑞沪深300ETF", "SH", "华泰柏瑞"),
    EtfTarget("510310", "易方达沪深300ETF", "SH", "易方达"),
    EtfTarget("510330", "华夏沪深300ETF", "SH", "华夏"),
    EtfTarget("159919", "嘉实沪深300ETF", "SZ", "嘉实"),
    EtfTarget("515330", "天弘沪深300ETF", "SH", "天弘"),
    EtfTarget("159925", "南方沪深300ETF", "SZ", "南方"),
    EtfTarget("510360", "广发沪深300ETF", "SH", "广发"),
    EtfTarget("510350", "工银沪深300ETF", "SH", "工银"),
]

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(_BASE_DIR, "etf_scale_history.csv")
DATA_JSON = os.path.join(_BASE_DIR, "data.json")

# 多周期背离：日历日回溯（非交易日则取不晚于目标日的最近一条历史）
PERIODS_DAYS: Tuple[int, ...] = (1, 5, 13, 30, 60)
# 涨跌幅绝对值低于该阈值视为「横盘」，避免噪声误判
TREND_EPS_PCT = 0.02
# data.json 最多保留交易日/记录条数（约 2 年按自然日上限）
MAX_HISTORY_RECORDS = 800
HS300_SYMBOL = "sh000300"

# CSV 引导时无份额列：用「规模(亿)/近似净值」估算亿份，仅用于历史 periods 连贯性
NAV_PROXY_YUAN_FOR_SHARES_ESTIMATE: float = 4.28

# 历史 CSV 曾为三列（无分项）；检测到旧表头时自动迁移为「8 只分项 + 合计」宽表
OLD_CSV_HEADER = ["日期", "前两只规模合计", "八只规模合计"]

# 宽表曾仅有 6 位代码、无中文简称列名；列顺序与当前一致时可仅升级表头并保留数据
LEGACY_WIDE_HEADER_CODES_ONLY: List[str] = (
    ["日期"] + [t.code for t in TARGET_ETFS] + ["前两只规模合计", "八只规模合计"]
)


def _csv_header() -> List[str]:
    """表头：日期 + 8 列「代码_中文简称」(亿元) + 前两只合计 + 八只合计"""
    return (
        ["日期"]
        + [f"{t.code}_{t.name_cn}" for t in TARGET_ETFS]
        + ["前两只规模合计", "八只规模合计"]
    )


def _csv_data_row(
    run_date: str,
    scale_yuan: Dict[str, float],
    top2_yi: float,
    all8_yi: float,
) -> List[Any]:
    """单行数据：分项缺失时留空字符串，合计仍写入（可能不含缺失项之和）"""
    row: List[Any] = [run_date]
    for t in TARGET_ETFS:
        y = scale_yuan.get(t.code)
        row.append(round(_yuan_to_yi(y), 6) if y is not None and y > 0 else "")
    row.append(top2_yi)
    row.append(all8_yi)
    return row


@dataclass
class FetchState:
    """记录告警与缺失，便于汇总打印"""

    warnings: List[str] = field(default_factory=list)
    missing_codes: List[str] = field(default_factory=list)

    def add_warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[警告] {msg}", file=sys.stderr)

    def mark_missing(self, code: str, reason: str) -> None:
        if code not in self.missing_codes:
            self.missing_codes.append(code)
        self.add_warn(f"{code}: {reason}")


def _norm_code(raw: str) -> str:
    """统一为 6 位字符串代码"""
    s = str(raw).strip()
    if s.isdigit():
        return s.zfill(6)
    return s


def _safe_call(label: str, fn: Callable[[], Any], state: FetchState) -> Optional[Any]:
    """包装调用，失败返回 None 并记录"""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — 需求要求接口失败不中断
        state.add_warn(f"{label} 调用失败: {exc}")
        return None


def _fetch_eastmoney_etf_spot_df(state: FetchState) -> Optional[pd.DataFrame]:
    """
    东方财富 ETF 全市场行情表。
    优先使用 akshare.fund_etf_spot_em；失败时用 requests 分页请求同源接口。
    """
    try:
        import akshare as ak

        df = _safe_call("akshare.fund_etf_spot_em", ak.fund_etf_spot_em, state)
        if df is not None and not df.empty:
            return df
    except Exception as exc:  # noqa: BLE001
        state.add_warn(f"akshare 导入或 fund_etf_spot_em 不可用: {exc}")

    url = "https://88.push2.eastmoney.com/api/qt/clist/get"
    base_params = {
        "pz": "200",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "wbp2u": "|0|0|0|web",
        "fid": "f12",
        "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827",
        "fields": "f2,f12,f14,f20,f38,f297",
    }

    def _pull() -> pd.DataFrame:
        rows: List[dict] = []
        page = 1
        while True:
            params = {**base_params, "pn": str(page)}
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            diff = data.get("data", {}).get("diff") or []
            if not diff:
                break
            for item in diff:
                rows.append(
                    {
                        "代码": _norm_code(item.get("f12", "")),
                        "名称": item.get("f14", ""),
                        "最新价": item.get("f2"),
                        "总市值": item.get("f20"),
                        "最新份额": item.get("f38"),
                        "数据日期": item.get("f297"),
                    }
                )
            total = int(data.get("data", {}).get("total", 0))
            if page * 200 >= total:
                break
            page += 1
        return pd.DataFrame(rows)

    return _safe_call("东方财富 ETF clist API 分页抓取", _pull, state)


def _row_to_float(row: pd.Series, key: str) -> float:
    v = pd.to_numeric(row.get(key), errors="coerce")
    if v is None or pd.isna(v):
        return float("nan")
    return float(v)


def _scale_yuan_from_spot_row(
    row: pd.Series, state: FetchState, code: str
) -> Tuple[float, str]:
    """
    由行情行得到统一规模（元）。
    优先总市值；否则 最新价×最新份额。
    """
    mv = _row_to_float(row, "总市值")
    if mv > 0:
        return mv, "总市值(元)"

    px = _row_to_float(row, "最新价")
    sh = _row_to_float(row, "最新份额")
    if px > 0 and sh > 0:
        state.add_warn(f"{code}: 总市值缺失，使用 最新价×最新份额 估算规模")
        return px * sh, "最新价×份额(元)"

    return float("nan"), ""


def _shares_fen_from_spot_row(row: pd.Series, state: FetchState, code: str) -> float:
    """
    东方财富「最新份额」字段（f38）：一般为基金总份额，单位「份」。
    异常或缺失时返回 nan，由上层决定是否用交易所补数。
    """
    sh = _row_to_float(row, "最新份额")
    if sh > 0:
        return sh
    if not pd.isna(sh) and sh <= 0:
        state.add_warn(f"{code}: 行情表最新份额为非正数，已忽略")
    return float("nan")


def _prepare_spot_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "代码" not in df.columns:
        return df
    for col in ("最新价", "总市值", "最新份额"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_code"] = df["代码"].map(_norm_code)
    return df


def _fetch_sse_shares_for_date(date_str: str, state: FetchState) -> Optional[pd.DataFrame]:
    def _fn() -> pd.DataFrame:
        import akshare as ak

        return ak.fund_etf_scale_sse(date=date_str)

    return _safe_call(f"上交所 ETF 份额({date_str})", _fn, state)


def _fetch_szse_shares(state: FetchState) -> Optional[pd.DataFrame]:
    def _fn() -> pd.DataFrame:
        import akshare as ak

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ak.fund_etf_scale_szse()

    return _safe_call("深交所 ETF 份额", _fn, state)


def _fill_exchange_scale_and_shares(
    spot_df: Optional[pd.DataFrame],
    state: FetchState,
    need_scale: List[str],
    need_shares: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    从深交所/上交所取「基金份额」（份），可选换算为规模（元）。
    返回:
      scale_yuan: 仅对 need_scale 中且能拿到净值×份额 的代码填充
      shares_fen: 对 need_shares（及规模侧需要时同一 vol）填充 份
    """
    scale_out: Dict[str, float] = {}
    shares_out: Dict[str, float] = {}
    if spot_df is None or "代码" not in spot_df.columns:
        return scale_out, shares_out

    sdf = _prepare_spot_df(spot_df)
    px_map: Dict[str, float] = {}
    for _, row in sdf.iterrows():
        c = row["_code"]
        p = _row_to_float(row, "最新价")
        if p > 0:
            px_map[c] = p

    need_sz = [c for c in need_scale + need_shares if c in ("159919", "159925")]
    need_sz = list(dict.fromkeys(need_sz))
    sh_targets = [t.code for t in TARGET_ETFS if t.exchange == "SH"]
    need_sh = [c for c in need_scale + need_shares if c in sh_targets]
    need_sh = list(dict.fromkeys(need_sh))

    sz_df = _fetch_szse_shares(state)
    if sz_df is not None and not sz_df.empty and "基金代码" in sz_df.columns:
        tdf = sz_df.copy()
        tdf["_c"] = tdf["基金代码"].map(_norm_code)
        for code in need_sz:
            sub = tdf[tdf["_c"] == code]
            if sub.empty:
                continue
            try:
                vol = float(pd.to_numeric(sub.iloc[0].get("基金份额"), errors="coerce"))
            except (TypeError, ValueError):
                continue
            if vol <= 0 or pd.isna(vol):
                continue
            if code in need_shares:
                shares_out[code] = vol
            if code in need_scale:
                px = px_map.get(code, float("nan"))
                if px > 0:
                    scale_out[code] = vol * px
                else:
                    state.add_warn(f"{code}: 有深交所份额但缺少行情最新价，无法折规模(元)")

    today = datetime.now().date()
    for delta in range(0, 15):
        d = today - timedelta(days=delta)
        ds = d.strftime("%Y%m%d")
        sse_df = _fetch_sse_shares_for_date(ds, state)
        if sse_df is None or sse_df.empty or "基金代码" not in sse_df.columns:
            continue
        tdf = sse_df.copy()
        tdf["_c"] = tdf["基金代码"].map(_norm_code)
        for code in need_sh:
            need_s = code in need_shares and code not in shares_out
            need_sc = code in need_scale and code not in scale_out
            if not need_s and not need_sc:
                continue
            sub = tdf[tdf["_c"] == code]
            if sub.empty:
                continue
            try:
                vol = float(pd.to_numeric(sub.iloc[0].get("基金份额"), errors="coerce"))
            except (TypeError, ValueError):
                continue
            if vol <= 0 or pd.isna(vol):
                continue
            if need_s:
                shares_out[code] = vol
            if need_sc:
                px = px_map.get(code, float("nan"))
                if px > 0:
                    scale_out[code] = vol * px
                    state.add_warn(
                        f"{code}: 使用上交所披露份额×最新价 估算规模 (统计日 {ds})"
                    )
        still_sh = any(
            (c in need_shares and c not in shares_out)
            or (c in need_scale and c not in scale_out)
            for c in need_sh
        )
        if not still_sh:
            break

    return scale_out, shares_out


def _collect_scales_and_shares(
    state: FetchState,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float]]:
    """
    返回:
      scale_yuan: 基金代码 -> 规模(元)
      detail: 基金代码 -> 口径说明（打印用）
      shares_fen: 基金代码 -> 总份额(份)；缺失则为该代码无键
    """
    scale_yuan: Dict[str, float] = {}
    detail: Dict[str, str] = {}
    shares_fen: Dict[str, float] = {}

    raw_spot = _fetch_eastmoney_etf_spot_df(state)
    spot_df = _prepare_spot_df(raw_spot) if raw_spot is not None else None

    if spot_df is not None and not spot_df.empty and "代码" in spot_df.columns:
        for t in TARGET_ETFS:
            sub = spot_df[spot_df["_code"] == t.code]
            if sub.empty:
                continue
            row = sub.iloc[0]
            yuan, desc = _scale_yuan_from_spot_row(row, state, t.code)
            if not pd.isna(yuan) and yuan > 0:
                scale_yuan[t.code] = yuan
                detail[t.code] = desc
            sf = _shares_fen_from_spot_row(row, state, t.code)
            if not pd.isna(sf) and sf > 0:
                shares_fen[t.code] = sf
    else:
        state.add_warn("东方财富 ETF 行情表不可用或为空")

    missing_scale = [t.code for t in TARGET_ETFS if t.code not in scale_yuan]
    missing_shares = [t.code for t in TARGET_ETFS if t.code not in shares_fen]
    if missing_scale or missing_shares:
        s_add, f_add = _fill_exchange_scale_and_shares(
            raw_spot, state, missing_scale, missing_shares
        )
        for c, v in s_add.items():
            if v > 0 and c not in scale_yuan:
                scale_yuan[c] = v
                detail[c] = detail.get(c, "交易所份额×最新价(元)")
        for c, v in f_add.items():
            if v > 0 and c not in shares_fen:
                shares_fen[c] = v

    return scale_yuan, detail, shares_fen


def _parse_iso_date(s: str) -> datetime.date:
    return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()


def _load_data_json() -> Dict[str, Any]:
    """读取 data.json；不存在或损坏时返回空壳结构。"""
    empty: Dict[str, Any] = {"meta": {}, "history": []}
    if not os.path.isfile(DATA_JSON):
        return empty
    try:
        with open(DATA_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return empty
        raw.setdefault("meta", {})
        h = raw.get("history")
        if not isinstance(h, list):
            raw["history"] = []
        return raw
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 读取 data.json 失败，将重建: {exc}", file=sys.stderr)
        return empty


def _save_data_json(payload: Dict[str, Any]) -> None:
    payload["meta"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["meta"]["periods_days"] = list(PERIODS_DAYS)
    payload["meta"]["trend_epsilon_pct"] = TREND_EPS_PCT
    with open(DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _trim_history(payload: Dict[str, Any]) -> None:
    """按日期排序后截断为最近 MAX_HISTORY_RECORDS 条。"""
    hist = payload.get("history") or []
    if not hist:
        return

    def _key(rec: Dict[str, Any]) -> str:
        return str(rec.get("date", ""))

    hist.sort(key=_key)
    if len(hist) > MAX_HISTORY_RECORDS:
        payload["history"] = hist[-MAX_HISTORY_RECORDS:]


def _fetch_hs300_latest_close(state: FetchState) -> Optional[float]:
    """沪深300指数 sh000300 最新收盘。"""

    def _ak() -> float:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=HS300_SYMBOL)
        if df is None or df.empty:
            raise ValueError("empty index daily")
        last = df.iloc[-1]
        return float(pd.to_numeric(last["close"], errors="coerce"))

    return _safe_call("akshare.stock_zh_index_daily(sh000300)", _ak, state)


def _fetch_hs300_daily_history(state: FetchState) -> Optional[pd.DataFrame]:
    """拉取指数日线，用于 CSV 引导或校验。"""

    def _ak() -> pd.DataFrame:
        import akshare as ak

        return ak.stock_zh_index_daily(symbol=HS300_SYMBOL)

    return _safe_call("指数日线全量", _ak, state)


def _forward_fill_scales_yuan(
    scale_yuan: Dict[str, float],
    prev_per_code_yi: Optional[Dict[str, Any]],
    state: FetchState,
) -> Dict[str, float]:
    """缺失 ETF 用上一日 JSON 中的亿元规模折算为元；再无则置 0。"""
    out = dict(scale_yuan)
    prev = prev_per_code_yi or {}
    for t in TARGET_ETFS:
        if t.code in out and out[t.code] > 0:
            continue
        py = prev.get(t.code)
        if py is not None and str(py).strip() != "":
            try:
                v = float(py)
                if v > 0:
                    out[t.code] = v * 1e8
                    state.add_warn(f"{t.code}: 本日抓取失败，沿用上日 JSON 规模 {v:.4f} 亿元")
                    continue
            except (TypeError, ValueError):
                pass
        out[t.code] = 0.0
        state.add_warn(f"{t.code}: 无有效规模，按 0 计入汇总")
    return out


def _forward_fill_shares_fen(
    shares_fen: Dict[str, float],
    prev_per_code_shares_yi: Optional[Dict[str, Any]],
    scale_yuan: Dict[str, float],
    state: FetchState,
) -> Dict[str, float]:
    """缺失份额优先沿用上日 JSON 亿份折算为份；再无则按规模÷近似净值估算；最后置 0。"""
    out = dict(shares_fen)
    prev = prev_per_code_shares_yi or {}
    nav = float(NAV_PROXY_YUAN_FOR_SHARES_ESTIMATE)
    if nav <= 0:
        nav = 4.28
    for t in TARGET_ETFS:
        c = t.code
        if c in out and out[c] > 0:
            continue
        py = prev.get(c)
        if py is not None and str(py).strip() != "":
            try:
                v = float(py)
                if v > 0:
                    out[c] = v * 1e8
                    state.add_warn(f"{c}: 本日份额抓取失败，沿用上日 JSON 份额 {v:.4f} 亿份")
                    continue
            except (TypeError, ValueError):
                pass
        sc = float(scale_yuan.get(c, 0.0) or 0.0)
        if sc > 0:
            out[c] = sc / nav
            state.add_warn(
                f"{c}: 无有效份额，按规模÷近似净值({nav}) 估算份额（份）"
            )
            continue
        out[c] = 0.0
        state.add_warn(f"{c}: 无有效份额，按 0 计入汇总")
    return out


def _all8_shares_yi_from_record(rec: Dict[str, Any]) -> Optional[float]:
    """从一条 history 记录解析 8 只总份额（亿份）；兼容仅有规模的老数据。"""
    raw = rec.get("all8_shares_yi")
    try:
        if raw is not None and str(raw).strip() != "":
            v = float(raw)
            if v > 0:
                return v
    except (TypeError, ValueError):
        pass
    d = rec.get("etf_per_code_shares_yi")
    if isinstance(d, dict) and d:
        s = 0.0
        for t in TARGET_ETFS:
            try:
                s += float(d.get(t.code, 0) or 0)
            except (TypeError, ValueError):
                continue
        if s > 0:
            return s
    nav = float(NAV_PROXY_YUAN_FOR_SHARES_ESTIMATE)
    if nav <= 0:
        nav = 4.28
    try:
        sc = rec.get("all8_scale_yi")
        if sc is not None and str(sc).strip() != "":
            v = float(sc)
            if v > 0:
                return v / nav
    except (TypeError, ValueError):
        pass
    return None


def _find_past_record(
    history: List[Dict[str, Any]], target_idx: int, ref_date: datetime.date
) -> Optional[Dict[str, Any]]:
    """在 target_idx 及之前，找日期不晚于 ref_date 的最近一条。"""
    for j in range(target_idx, -1, -1):
        try:
            d = _parse_iso_date(str(history[j]["date"]))
        except (ValueError, KeyError, TypeError):
            continue
        if d <= ref_date:
            return history[j]
    return None


def _trend_from_pct(pct: float) -> str:
    if pct > TREND_EPS_PCT:
        return "up"
    if pct < -TREND_EPS_PCT:
        return "down"
    return "flat"


def _compute_periods_for_index(
    history: List[Dict[str, Any]], target_idx: int
) -> Dict[str, Any]:
    """
    以 history[target_idx] 为「当前日」，对各周期计算 8 只 ETF 总份额 vs 沪深300 的涨跌幅与背离。
    """
    out: Dict[str, Any] = {}
    if target_idx < 0 or target_idx >= len(history):
        return out
    cur = history[target_idx]
    try:
        cur_date = _parse_iso_date(str(cur["date"]))
    except (ValueError, KeyError, TypeError):
        return out

    cur_etf_f = _all8_shares_yi_from_record(cur)
    cur_idx = cur.get("hs300_close")
    try:
        cur_idx_f = float(cur_idx) if cur_idx is not None else float("nan")
    except (TypeError, ValueError):
        cur_idx_f = float("nan")

    for n in PERIODS_DAYS:
        key = str(n)
        ref_date = cur_date - timedelta(days=int(n))
        past = _find_past_record(history, target_idx - 1, ref_date)
        if past is None:
            out[key] = {
                "sufficient": False,
                "days": n,
                "note": "历史长度不足或缺少对比日",
            }
            continue
        p_etf = _all8_shares_yi_from_record(past)
        try:
            p_idx = float(past.get("hs300_close"))
        except (TypeError, ValueError):
            out[key] = {"sufficient": False, "days": n, "note": "对比日数值无效"}
            continue
        if p_etf is None or cur_etf_f is None or p_etf <= 0 or cur_etf_f <= 0:
            out[key] = {"sufficient": False, "days": n, "note": "ETF总份额无效或缺失"}
            continue
        if p_idx <= 0 or cur_idx_f <= 0 or (cur_idx_f != cur_idx_f):
            out[key] = {"sufficient": False, "days": n, "note": "指数收盘价无效"}
            continue

        etf_pct = (cur_etf_f - p_etf) / p_etf * 100.0
        idx_pct = (cur_idx_f - p_idx) / p_idx * 100.0
        etf_tr = _trend_from_pct(etf_pct)
        idx_tr = _trend_from_pct(idx_pct)
        div = (etf_tr == "up" and idx_tr == "down") or (etf_tr == "down" and idx_tr == "up")
        if etf_tr == "up" and idx_tr == "down":
            div_type = "share_up_index_down"
            label = "份额↑ 指数↓（申赎/资金与指数背离）"
        elif etf_tr == "down" and idx_tr == "up":
            div_type = "share_down_index_up"
            label = "份额↓ 指数↑（申赎/资金与指数背离）"
        else:
            div_type = "none"
            label = "无背离"

        out[key] = {
            "sufficient": True,
            "days": n,
            "etf_change_pct": round(etf_pct, 4),
            "index_change_pct": round(idx_pct, 4),
            "etf_trend": etf_tr,
            "index_trend": idx_tr,
            "divergence": div,
            "divergence_type": div_type,
            "label": label,
            "compare_date": str(past.get("date")),
        }
    return out


def _bootstrap_history_from_csv(
    state: FetchState,
) -> List[Dict[str, Any]]:
    """
    若 data.json 无历史，尝试用 etf_scale_history.csv + 指数日线对齐生成初始序列。
    """
    if not os.path.isfile(HISTORY_CSV):
        return []

    idx_df = _fetch_hs300_daily_history(state)
    if idx_df is None or idx_df.empty:
        state.add_warn("引导失败：无法获取沪深300历史日线，跳过 CSV 导入")
        return []

    idx_df = idx_df.copy()
    idx_df["date"] = pd.to_datetime(idx_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    idx_df["close"] = pd.to_numeric(idx_df["close"], errors="coerce")
    px_map: Dict[str, float] = {}
    for _, row in idx_df.iterrows():
        ds = row["date"]
        if isinstance(ds, str) and ds and not pd.isna(row["close"]):
            px_map[ds] = float(row["close"])

    out_list: List[Dict[str, Any]] = []

    try:
        with open(HISTORY_CSV, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_s = str(row.get("日期", "")).strip()
                if not date_s:
                    continue
                per: Dict[str, float] = {}
                for t in TARGET_ETFS:
                    col = f"{t.code}_{t.name_cn}"
                    raw = row.get(col, "")
                    if raw is not None and str(raw).strip() != "":
                        try:
                            per[t.code] = float(raw)
                        except ValueError:
                            per[t.code] = 0.0
                    else:
                        per[t.code] = 0.0
                try:
                    top2 = float(row.get("前两只规模合计", 0) or 0)
                    all8 = float(row.get("八只规模合计", 0) or 0)
                except (TypeError, ValueError):
                    top2, all8 = 0.0, 0.0
                other6 = all8 - top2
                ic = px_map.get(date_s[:10])
                if ic is None:
                    ic = None
                nav = float(NAV_PROXY_YUAN_FOR_SHARES_ESTIMATE)
                if nav <= 0:
                    nav = 4.28
                per_sh = {
                    k: round((v / nav) if v and v > 0 else 0.0, 6)
                    for k, v in per.items()
                }
                top2_sh = sum(per_sh[c] for c in TOP_TWO_CODES)
                all8_sh = sum(per_sh[t.code] for t in TARGET_ETFS)
                other6_sh = all8_sh - top2_sh
                out_list.append(
                    {
                        "date": date_s[:10],
                        "hs300_close": ic,
                        "top2_scale_yi": round(top2, 6),
                        "all8_scale_yi": round(all8, 6),
                        "other6_scale_yi": round(other6, 6),
                        "all8_shares_yi": round(all8_sh, 6),
                        "top2_shares_yi": round(top2_sh, 6),
                        "other6_shares_yi": round(other6_sh, 6),
                        "etf_per_code_yi": {k: round(v, 6) for k, v in per.items()},
                        "etf_per_code_shares_yi": per_sh,
                        "periods": {},
                    }
                )
    except Exception as exc:  # noqa: BLE001
        state.add_warn(f"读取 CSV 引导失败: {exc}")
        return []

    out_list.sort(key=lambda r: r["date"])
    # 逐条补全 periods
    for i in range(len(out_list)):
        out_list[i]["periods"] = _compute_periods_for_index(out_list, i)
    if out_list:
        print(f"[提示] 已从 CSV + 指数日线引导 {len(out_list)} 条历史至 data.json", flush=True)
    return out_list


def _build_today_record(
    run_date: str,
    scale_yuan: Dict[str, float],
    shares_fen: Dict[str, float],
    hs300_close: Optional[float],
    state: FetchState,
) -> Dict[str, Any]:
    """组装当日写入 JSON 的基础字段（不含 periods）。"""
    per_yi = {t.code: round(_yuan_to_yi(scale_yuan.get(t.code, 0.0)), 6) for t in TARGET_ETFS}
    per_shares_yi = {
        t.code: round(float(shares_fen.get(t.code, 0.0) or 0.0) / 1e8, 6)
        for t in TARGET_ETFS
    }
    top2_yi = sum(per_yi[c] for c in TOP_TWO_CODES)
    all8_yi = sum(per_yi[t.code] for t in TARGET_ETFS)
    other6_yi = all8_yi - top2_yi
    top2_sh = sum(per_shares_yi[c] for c in TOP_TWO_CODES)
    all8_sh = sum(per_shares_yi[t.code] for t in TARGET_ETFS)
    other6_sh = all8_sh - top2_sh
    return {
        "date": run_date,
        "hs300_close": round(float(hs300_close), 4) if hs300_close is not None else None,
        "top2_scale_yi": round(top2_yi, 6),
        "all8_scale_yi": round(all8_yi, 6),
        "other6_scale_yi": round(other6_yi, 6),
        "all8_shares_yi": round(all8_sh, 6),
        "top2_shares_yi": round(top2_sh, 6),
        "other6_shares_yi": round(other6_sh, 6),
        "etf_per_code_yi": per_yi,
        "etf_per_code_shares_yi": per_shares_yi,
        "periods": {},
    }


def _merge_history_record(payload: Dict[str, Any], record: Dict[str, Any]) -> None:
    """按日期去重合并：同日覆盖，最后按日期排序。"""
    hist: List[Dict[str, Any]] = list(payload.get("history") or [])
    d = record["date"]
    hist = [x for x in hist if str(x.get("date")) != str(d)]
    hist.append(record)
    hist.sort(key=lambda x: str(x.get("date", "")))
    payload["history"] = hist


def _yuan_to_yi(val: float) -> float:
    return val / 1e8


def _persist_history_csv(
    run_date: str,
    scale_yuan: Dict[str, float],
    top2_yi: float,
    all8_yi: float,
    state: FetchState,
) -> None:
    """
    写入 etf_scale_history.csv：
    - 新文件：写宽表头 + 当日行；
    - 表头已匹配：写入当日行；若已有同一「日期」则覆盖该行，否则追加；
    - 旧版三列：整表迁移为宽表（分项列为空）后写入当日行；
    - 仅代码宽表：升级表头为「代码_中文简称」并保留数据行；
    - 其它表头：备份原文件后重写为当日数据。
    """
    header = _csv_header()
    new_row = _csv_data_row(run_date, scale_yuan, top2_yi, all8_yi)

    def _write_all(data_rows: List[List[Any]]) -> None:
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data_rows)

    if not os.path.isfile(HISTORY_CSV):
        _write_all([new_row])
        return

    with open(HISTORY_CSV, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        _write_all([new_row])
        return

    existing_header = rows[0]

    if existing_header == header:
        # 同一天多次运行：覆盖当日行，避免重复日期
        kept = [r for r in rows[1:] if r and str(r[0]).strip() != run_date]
        kept.append(new_row)
        _write_all(kept)
        return

    if existing_header == LEGACY_WIDE_HEADER_CODES_ONLY:
        data_rows = [
            r
            for r in rows[1:]
            if len(r) >= len(header) and r[0] != run_date
        ]
        data_rows.append(new_row)
        _write_all(data_rows)
        print(
            "\n[提示] 表头已由「仅代码」更新为「代码_中文简称」，历史数据列已对齐保留。",
            flush=True,
        )
        return

    if existing_header == OLD_CSV_HEADER:
        migrated: List[List[Any]] = []
        for data in rows[1:]:
            if len(data) < 3:
                continue
            date_s, t2, a8 = data[0], data[1], data[2]
            # 与本次采集同一天的历史旧行由 new_row 覆盖，避免重复日期
            if date_s == run_date:
                continue
            migrated.append(
                [date_s, "", "", "", "", "", "", "", "", t2, a8]
            )
        migrated.append(new_row)
        _write_all(migrated)
        print(
            "\n[提示] 历史 CSV 已由旧版（仅合计列）迁移为「8 只分项 + 合计」宽表；"
            "旧行分项列为空，新采集起写入分项。",
            flush=True,
        )
        return

    bak = HISTORY_CSV.replace(".csv", "_backup_mismatched_header.csv")
    shutil.copy2(HISTORY_CSV, bak)
    state.add_warn(
        f"CSV 表头与预期不符，已备份至 {os.path.basename(bak)}，"
        "已用新表头仅保留今日一行。"
    )
    _write_all([new_row])


def run() -> None:
    state = FetchState()
    print("=" * 60)
    print("沪深300 ETF 规模追踪与多周期背离预警 — 数据采集")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    payload = _load_data_json()
    if not payload.get("history"):
        payload["history"] = _bootstrap_history_from_csv(state)

    raw_scale_yuan, detail, raw_shares_fen = _collect_scales_and_shares(state)
    last = payload["history"][-1] if payload["history"] else None
    prev_yi = (last or {}).get("etf_per_code_yi") or {}
    prev_shares_yi = (last or {}).get("etf_per_code_shares_yi") or {}
    scale_yuan = _forward_fill_scales_yuan(raw_scale_yuan, prev_yi, state)
    shares_fen = _forward_fill_shares_fen(
        raw_shares_fen, prev_shares_yi, scale_yuan, state
    )

    hs300_close = _fetch_hs300_latest_close(state)
    if hs300_close is None and last is not None:
        try:
            pv = last.get("hs300_close")
            if pv is not None:
                hs300_close = float(pv)
                state.add_warn("沪深300指数本日抓取失败，沿用上一日收盘价")
        except (TypeError, ValueError):
            pass

    run_date = datetime.now().strftime("%Y-%m-%d")
    today_rec = _build_today_record(
        run_date, scale_yuan, shares_fen, hs300_close, state
    )
    _merge_history_record(payload, today_rec)
    hi = len(payload["history"]) - 1
    payload["history"][hi]["periods"] = _compute_periods_for_index(payload["history"], hi)
    _trim_history(payload)
    _save_data_json(payload)

    row = payload["history"][hi]

    # 控制台：先逐项金额，再其余 6 只小计，最后为前两只合计与八只合计
    print("\n【各基金明细】（单位：亿元人民币）")
    for t in TARGET_ETFS:
        yi = _yuan_to_yi(scale_yuan.get(t.code, 0.0))
        if t.code in raw_scale_yuan:
            print(f"  {t.code} {t.name}: {yi:.4f} 亿元 ({detail.get(t.code, '')})")
        elif yi > 0:
            print(f"  {t.code} {t.name}: {yi:.4f} 亿元 (沿用上一日 JSON)")
        else:
            print(f"  {t.code} {t.name}: {yi:.4f} 亿元 (无数据，按0)")

    print("\n【各基金份额】（单位：亿份）")
    for t in TARGET_ETFS:
        sy = shares_fen.get(t.code, 0.0) / 1e8
        if t.code in raw_shares_fen:
            print(f"  {t.code} {t.name}: {sy:.4f} 亿份 (行情或交易所披露)")
        elif sy > 0:
            print(f"  {t.code} {t.name}: {sy:.4f} 亿份 (沿用或按规模估算)")
        else:
            print(f"  {t.code} {t.name}: {sy:.4f} 亿份 (无数据，按0)")

    print("\n【分项汇总】")
    print(
        f"  其余 6 只合计 (8 只总和 - 前两只): {row['other6_scale_yi']:.4f} 亿元"
    )
    print("\n【规模合计】")
    print(
        f"  华泰柏瑞(510300) + 易方达(510310) 合计: {row['top2_scale_yi']:.4f} 亿元"
    )
    print(f"  全部 8 只合计: {row['all8_scale_yi']:.4f} 亿元")
    ic = row.get("hs300_close")
    print(f"  沪深300 (sh000300) 收盘: {ic if ic is not None else '缺失'}")

    print("\n【份额合计】（单位：亿份）")
    print(
        f"  华泰柏瑞(510300) + 易方达(510310) 合计: {row['top2_shares_yi']:.4f} 亿份"
    )
    print(f"  全部 8 只合计: {row['all8_shares_yi']:.4f} 亿份")
    print(
        f"  其余 6 只合计: {row['other6_shares_yi']:.4f} 亿份"
    )

    print("\n【多周期份额 vs 指数】（日历日回溯，对比日见各条 compare_date）")
    periods = row.get("periods") or {}
    for n in PERIODS_DAYS:
        p = periods.get(str(n), {})
        if not p.get("sufficient"):
            print(f"  {n} 日: 数据不足 — {p.get('note', '')}")
            continue
        flag = " *** 背离 *** " if p.get("divergence") else ""
        print(
            f"  {n} 日: 份额 {p.get('etf_change_pct')}% ({p.get('etf_trend')}), "
            f"指数 {p.get('index_change_pct')}% ({p.get('index_trend')}) "
            f"vs {p.get('compare_date')}{flag}"
        )
        if p.get("divergence"):
            print(f"         -> {p.get('label')}")

    _persist_history_csv(
        run_date,
        scale_yuan,
        float(row["top2_scale_yi"]),
        float(row["all8_scale_yi"]),
        state,
    )

    print(f"\n已写入: {DATA_JSON}（含历史与背离字段，约保留 {MAX_HISTORY_RECORDS} 条）")
    print(
        f"已写入: {HISTORY_CSV}（分项列为「代码_中文简称」，末两列为合计；单位：亿元）"
    )
    print("=" * 60)

    if state.warnings:
        print(f"\n共 {len(state.warnings)} 条告警，请查看上文 [警告] 输出。")


if __name__ == "__main__":
    run()
