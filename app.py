import os
import re
import math
import json
import time
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

app = FastAPI(title="Polymarket MVP API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _stamp() -> str:
    return _now_utc().strftime("%Y%m%d_%H%M%S")

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None

def _num(x: Any) -> float:
    v = _safe_float(x)
    return v if v is not None else 0.0

def _parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if v is not None]
        except Exception:
            pass
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(value)]

def http_json(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Any = None,
    timeout_s: int = 30,
    max_retries: int = 6,
    backoff_s: float = 0.8,
) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "polymarket-mvp-api/1.0"}
    for attempt in range(1, max_retries + 1):
        try:
            if method.upper() == "GET":
                r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
            else:
                headers2 = dict(headers)
                headers2["Content-Type"] = "application/json"
                r = requests.request(
                    method.upper(),
                    url,
                    params=params,
                    headers=headers2,
                    json=json_body,
                    timeout=timeout_s,
                )

            if r.status_code == 429 or (500 <= r.status_code <= 599):
                time.sleep(backoff_s * (2 ** (attempt - 1)))
                continue

            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(backoff_s * (2 ** (attempt - 1)))
    raise RuntimeError("retries exhausted")

def fetch_gamma_markets(closed: bool, limit: int, max_markets: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        if max_markets > 0 and len(out) >= max_markets:
            return out[:max_markets]
        params = {"limit": limit, "offset": offset, "closed": "true" if closed else "false"}
        page = http_json("GET", f"{GAMMA_BASE}/markets", params=params)
        if not isinstance(page, list) or len(page) == 0:
            break
        out.extend(page)
        offset += limit
    return out

def fetch_books(token_ids: List[str], batch_size: int = 300) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(token_ids), batch_size):
        chunk = token_ids[i:i+batch_size]
        body = [{"token_id": t} for t in chunk]
        books = http_json("POST", f"{CLOB_BASE}/books", json_body=body)
        if isinstance(books, list):
            for b in books:
                aid = b.get("asset_id")
                if aid is not None:
                    out[str(aid)] = b
    return out

def best_bid_ask(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bid_price = None
    bid_size = None
    ask_price = None
    ask_size = None

    best_bp = -1.0
    best_bs = None
    for lvl in bids:
        p = _safe_float(lvl.get("price"))
        s = _safe_float(lvl.get("size"))
        if p is None:
            continue
        if p > best_bp:
            best_bp = p
            best_bs = s
    if best_bp >= 0:
        bid_price = best_bp
        bid_size = best_bs

    best_ap = None
    best_as = None
    for lvl in asks:
        p = _safe_float(lvl.get("price"))
        s = _safe_float(lvl.get("size"))
        if p is None:
            continue
        if best_ap is None or p < best_ap:
            best_ap = p
            best_as = s
    if best_ap is not None:
        ask_price = best_ap
        ask_size = best_as

    return bid_price, bid_size, ask_price, ask_size

def spread_pct(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 100.0

def mid_price(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0

def parse_iso_datetime(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def time_to_deadline_hours(end_iso: str) -> Optional[float]:
    d = parse_iso_datetime(end_iso)
    if d is None:
        return None
    now = _now_utc()
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return (d - now).total_seconds() / 3600.0

_year_re = re.compile(r"\b(20\d{2})\b")

def deadline_suspicious(question: str, end_date: str, ttd_h: Optional[float]) -> int:
    if ttd_h is not None and ttd_h < -24:
        return 1
    q_years = [int(y) for y in _year_re.findall(question or "")]
    d = parse_iso_datetime(end_date or "")
    if q_years and d is not None:
        max_q = max(q_years)
        end_y = d.year
        if max_q > end_y:
            return 1
    return 0

def quality_score(vol24: float, spread_mean: Optional[float], min_ask_size: float) -> float:
    sp = spread_mean if spread_mean is not None else 99.0
    vol_part = math.log10(max(vol24, 1.0))
    depth_part = math.log10(max(min_ask_size, 1.0))
    spread_part = max(0.1, (10.0 / (sp + 0.5)))
    return round((vol_part * 2.2) + (depth_part * 1.2) + (spread_part * 1.8), 4)

def tier_for(vol24: float, spread_mean: Optional[float], min_ask_size: float) -> str:
    if spread_mean is None:
        return "SKIP"
    if vol24 >= 20000 and spread_mean <= 2.5 and min_ask_size >= 50:
        return "A"
    if vol24 >= 5000 and spread_mean <= 4.0 and min_ask_size >= 20:
        return "B"
    return "C"

def _norm_outcome_name(s: str) -> str:
    return (s or "").strip().strip('"').strip("'").strip().lower()

def _find_yes_no_indices(outs: List[str]) -> Optional[Tuple[int, int]]:
    idx_yes = None
    idx_no = None
    for i, o in enumerate(outs):
        n = _norm_outcome_name(o)
        if n == "yes":
            idx_yes = i
        elif n == "no":
            idx_no = i
    if idx_yes is None or idx_no is None:
        return None
    return idx_yes, idx_no

def _bucket_for(
    vol24: float,
    liq: float,
    spread_mean: Optional[float],
    min_ask_size: float,
    ttd_h: Optional[float],
    deadline_flag: int,
    yes_ask: Optional[float],
    no_ask: Optional[float],
) -> str:
    if deadline_flag == 1:
        return "SKIP"
    if spread_mean is None:
        return "SKIP"
    if min_ask_size < 20 or vol24 < 1000 or spread_mean > 4.0:
        return "SKIP"

    if ttd_h is not None:
        if ttd_h < 2:
            return "WATCH"
        if ttd_h > 8760:
            return "WATCH"

    if yes_ask is not None and (yes_ask < 0.03 or yes_ask > 0.97):
        return "WATCH"
    if no_ask is not None and (no_ask < 0.03 or no_ask > 0.97):
        return "WATCH"

    trade_ok = (
        spread_mean <= 3.0
        and min_ask_size >= 50
        and (vol24 >= 20000 or liq >= 20000)
        and (ttd_h is None or (ttd_h >= 6 and ttd_h <= 4320))
    )
    return "TRADE" if trade_ok else "WATCH"

def build_candidates(max_markets: int = 2000) -> List[Dict[str, Any]]:
    markets = fetch_gamma_markets(closed=False, limit=200, max_markets=max_markets)

    tradable = []
    token_ids: List[str] = []
    for m in markets:
        if bool(m.get("enableOrderBook")) is not True:
            continue
        if bool(m.get("closed")) is True:
            continue
        if m.get("acceptingOrders") is False:
            continue
        if m.get("active") is False:
            continue

        tids = _parse_list(m.get("clobTokenIds"))
        outs = _parse_list(m.get("outcomes"))
        if len(tids) < 2 or len(outs) < 2:
            continue

        yn = _find_yes_no_indices(outs)
        if yn is None:
            continue

        tradable.append(m)

        for t in tids[:2]:
            token_ids.append(t)

    token_ids = list(dict.fromkeys(token_ids))
    books = fetch_books(token_ids, batch_size=300)

    out: List[Dict[str, Any]] = []
    snap_time = _now_utc().isoformat()

    for m in tradable:
        tids = _parse_list(m.get("clobTokenIds"))
        outs = _parse_list(m.get("outcomes"))
        if len(tids) < 2 or len(outs) < 2:
            continue

        yn = _find_yes_no_indices(outs)
        if yn is None:
            continue
        idx_yes, idx_no = yn

        if idx_yes >= len(tids) or idx_no >= len(tids):
            continue

        t_yes = tids[idx_yes]
        t_no = tids[idx_no]
        o_yes = outs[idx_yes]
        o_no = outs[idx_no]

        by, bsy, ay, asy = best_bid_ask(books.get(t_yes, {}))
        bn, bsn, an, asn = best_bid_ask(books.get(t_no, {}))

        has_quotes = (ay is not None and by is not None and an is not None and bn is not None)
        if not has_quotes:
            continue

        sp_yes = spread_pct(by, ay)
        sp_no = spread_pct(bn, an)
        sps = [x for x in [sp_yes, sp_no] if x is not None]
        sp_mean = sum(sps) / len(sps) if sps else None
        if sp_mean is None:
            continue

        mid_yes = mid_price(by, ay)
        mid_no = mid_price(bn, an)

        if mid_yes is None or mid_no is None:
            continue
        if abs((mid_yes + mid_no) - 1.0) > 0.08:
            continue

        vol24 = _num(m.get("volume24hrClob", m.get("volume24hr", m.get("volume24hrAmm", 0))))
        liq = _num(m.get("liquidityClob", m.get("liquidityNum", m.get("liquidity", 0))))

        end_date = str(m.get("endDateIso", m.get("endDate", "")) or "")
        ttd_h = time_to_deadline_hours(end_date)
        q = str(m.get("question", "") or "")

        min_ask_size = min(_num(asy), _num(asn)) if (asy is not None and asn is not None) else 0.0
        if min_ask_size < 20:
            continue
        if vol24 < 1000:
            continue

        ds = deadline_suspicious(q, end_date, ttd_h)

        tier = tier_for(vol24, sp_mean, min_ask_size)

        req_yes = (ay - mid_yes) if (ay is not None and mid_yes is not None) else None
        req_no = (an - mid_no) if (an is not None and mid_no is not None) else None
        req_min = min([x for x in [req_yes, req_no] if x is not None], default=None)

        bucket = _bucket_for(
            vol24=vol24,
            liq=liq,
            spread_mean=sp_mean,
            min_ask_size=min_ask_size,
            ttd_h=ttd_h,
            deadline_flag=ds,
            yes_ask=ay,
            no_ask=an,
        )

        out.append({
            "snapshot_time_utc": snap_time,
            "market_id": str(m.get("id", "")),
            "question": q,
            "slug": str(m.get("slug", "") or ""),
            "category": str(m.get("category", "") or ""),
            "end_date": end_date,
            "time_to_deadline_hours": ttd_h,
            "deadline_suspicious": ds,
            "volume24h": vol24,
            "liquidity": liq,
            "market_spread_pct_mean": sp_mean,
            "min_best_ask_size": min_ask_size,
            "tier": tier,
            "bucket": bucket,
            "quality_score": quality_score(vol24, sp_mean, min_ask_size),
            "yes_ask": ay,
            "no_ask": an,
            "yes_bid": by,
            "no_bid": bn,
            "outcomes": [
                {
                    "name": str(o_yes),
                    "token_id": str(t_yes),
                    "best_bid": by, "best_bid_size": bsy,
                    "best_ask": ay, "best_ask_size": asy,
                    "mid": mid_yes, "spread_pct": sp_yes,
                    "req_edge_vs_mid": req_yes,
                },
                {
                    "name": str(o_no),
                    "token_id": str(t_no),
                    "best_bid": bn, "best_bid_size": bsn,
                    "best_ask": an, "best_ask_size": asn,
                    "mid": mid_no, "spread_pct": sp_no,
                    "req_edge_vs_mid": req_no,
                }
            ],
            "req_edge_min": req_min,
        })

    out.sort(key=lambda x: (x.get("bucket") != "TRADE", x.get("tier") != "A", -x.get("quality_score", 0.0)))
    return out

def make_batches(
    cands: List[Dict[str, Any]],
    tiers: List[str],
    batch_size: int,
    include_watch_in_batches: bool = False,
) -> List[Dict[str, Any]]:
    filtered = [c for c in cands if c.get("tier") in tiers]
    if include_watch_in_batches:
        filtered2 = [c for c in filtered if c.get("bucket") in ("TRADE", "WATCH")]
    else:
        filtered2 = [c for c in filtered if c.get("bucket") == "TRADE"]

    batches = []
    for i in range(0, len(filtered2), batch_size):
        part = filtered2[i:i+batch_size]
        batches.append({
            "batch_id": f"{(i//batch_size)+1:02d}",
            "tiers": ",".join(tiers),
            "markets": part
        })
    return batches

@app.get("/health")
def health():
    return {"ok": True, "time_utc": _now_utc().isoformat()}

@app.get("/run")
def run(
    tiers: str = Query("A,B", description="Comma-separated tiers to include, e.g. A,B"),
    batch_size: int = Query(20, ge=5, le=50),
    max_markets: int = Query(2000, ge=100, le=5000),
    include_watch_in_batches: bool = Query(False, description="If true, include WATCH markets in LLM batches too"),
):
    tiers_list = [t.strip().upper() for t in tiers.split(",") if t.strip()]
    cands = build_candidates(max_markets=max_markets)

    batches = make_batches(
        cands,
        tiers=tiers_list,
        batch_size=batch_size,
        include_watch_in_batches=include_watch_in_batches,
    )

    trade_total = sum(1 for c in cands if c.get("bucket") == "TRADE" and c.get("tier") in tiers_list)
    watch_total = sum(1 for c in cands if c.get("bucket") == "WATCH" and c.get("tier") in tiers_list)
    skip_total = sum(1 for c in cands if c.get("bucket") == "SKIP" and c.get("tier") in tiers_list)

    return {
        "run_id": _stamp(),
        "generated_at_utc": _now_utc().isoformat(),
        "tiers": tiers_list,
        "batch_size": batch_size,
        "max_markets": max_markets,
        "include_watch_in_batches": include_watch_in_batches,
        "candidates_total": len(cands),
        "trade_total": trade_total,
        "watch_total": watch_total,
        "skip_total": skip_total,
        "batches_total": len(batches),
        "batches": batches,
    }
