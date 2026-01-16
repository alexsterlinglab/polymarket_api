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

def http_json(method: str, url: str, params: Optional[Dict[str, Any]] = None, json_body: Any = None,
             timeout_s: int = 30, max_retries: int = 6, backoff_s: float = 0.8) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "polymarket-mvp-api/1.0"}
    for attempt in range(1, max_retries + 1):
        try:
            if method.upper() == "GET":
                r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
            else:
                headers2 = dict(headers)
                headers2["Content-Type"] = "application/json"
                r = requests.request(method.upper(), url, params=params, headers=headers2, json=json_body, timeout=timeout_s)

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
    # 1) very negative deadline
    if ttd_h is not None and ttd_h < -24:
        return 1
    # 2) question contains year > end_date year (common mismatch)
    q_years = [int(y) for y in _year_re.findall(question or "")]
    d = parse_iso_datetime(end_date or "")
    if q_years and d is not None:
        max_q = max(q_years)
        end_y = d.year
        if max_q > end_y:
            return 1
    return 0

def quality_score(vol24: float, spread_mean: Optional[float], min_ask_size: float) -> float:
    # simple monotonic score: more vol, lower spread, more depth
    sp = spread_mean if spread_mean is not None else 99.0
    vol_part = math.log10(max(vol24, 1.0))  # 0..~6
    depth_part = math.log10(max(min_ask_size, 1.0))  # 0..~4
    spread_part = max(0.1, (10.0 / (sp + 0.5)))  # smaller spread -> bigger
    return round((vol_part * 2.2) + (depth_part * 1.2) + (spread_part * 1.8), 4)

def tier_for(vol24: float, spread_mean: Optional[float], min_ask_size: float) -> str:
    if spread_mean is None:
        return "SKIP"
    if vol24 >= 20000 and spread_mean <= 2.5 and min_ask_size >= 50:
        return "A"
    if vol24 >= 5000 and spread_mean <= 4.0 and min_ask_size >= 20:
        return "B"
    return "C"

def build_candidates(max_markets: int = 2000) -> List[Dict[str, Any]]:
    markets = fetch_gamma_markets(closed=False, limit=200, max_markets=max_markets)

    tradable = []
    token_ids = []
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
        if len(tids) < 2:
            continue
        tradable.append(m)
        token_ids.extend([tids[0], tids[1]])

    # dedupe tokens
    token_ids = list(dict.fromkeys(token_ids))
    books = fetch_books(token_ids, batch_size=300)

    out = []
    snap_time = _now_utc().isoformat()

    for m in tradable:
        tids = _parse_list(m.get("clobTokenIds"))
        outs = _parse_list(m.get("outcomes"))
        if len(tids) < 2:
            continue

        t0, t1 = tids[0], tids[1]
        o0 = outs[0] if len(outs) > 0 else "Yes"
        o1 = outs[1] if len(outs) > 1 else "No"

        b0, bs0, a0, as0 = best_bid_ask(books.get(t0, {}))
        b1, bs1, a1, as1 = best_bid_ask(books.get(t1, {}))

        sp0 = spread_pct(b0, a0)
        sp1 = spread_pct(b1, a1)
        sps = [x for x in [sp0, sp1] if x is not None]
        sp_mean = sum(sps)/len(sps) if sps else None

        mid0 = mid_price(b0, a0)
        mid1 = mid_price(b1, a1)

        vol24 = _num(m.get("volume24hrClob", m.get("volume24hr", m.get("volume24hrAmm", 0))))
        liq = _num(m.get("liquidityClob", m.get("liquidityNum", m.get("liquidity", 0))))

        end_date = str(m.get("endDateIso", m.get("endDate", "")) or "")
        ttd_h = time_to_deadline_hours(end_date)
        q = str(m.get("question", "") or "")

        min_ask_size = min(_num(as0), _num(as1)) if (as0 is not None and as1 is not None) else 0.0
        tier = tier_for(vol24, sp_mean, min_ask_size)

        # hard filters for "analyze"
        has_quotes = (a0 is not None and b0 is not None and a1 is not None and b1 is not None)
        if not has_quotes:
            continue
        if sp_mean is None:
            continue
        if min_ask_size < 20:
            continue
        if vol24 < 1000:
            continue

        ds = deadline_suspicious(q, end_date, ttd_h)

        req0 = (a0 - mid0) if (a0 is not None and mid0 is not None) else None
        req1 = (a1 - mid1) if (a1 is not None and mid1 is not None) else None
        req_min = min([x for x in [req0, req1] if x is not None], default=None)

        out.append({
            "snapshot_time_utc": snap_time,
            "market_id": str(m.get("id","")),
            "question": q,
            "slug": str(m.get("slug","") or ""),
            "category": str(m.get("category","") or ""),
            "end_date": end_date,
            "time_to_deadline_hours": ttd_h,
            "deadline_suspicious": ds,
            "volume24h": vol24,
            "liquidity": liq,
            "market_spread_pct_mean": sp_mean,
            "min_best_ask_size": min_ask_size,
            "tier": tier,
            "quality_score": quality_score(vol24, sp_mean, min_ask_size),
            "outcomes": [
                {
                    "name": o0,
                    "token_id": t0,
                    "best_bid": b0, "best_bid_size": bs0,
                    "best_ask": a0, "best_ask_size": as0,
                    "mid": mid0, "spread_pct": sp0,
                    "req_edge_vs_mid": req0,
                },
                {
                    "name": o1,
                    "token_id": t1,
                    "best_bid": b1, "best_bid_size": bs1,
                    "best_ask": a1, "best_ask_size": as1,
                    "mid": mid1, "spread_pct": sp1,
                    "req_edge_vs_mid": req1,
                }
            ],
            "req_edge_min": req_min,
        })

    # sort best first
    out.sort(key=lambda x: (x["tier"] != "A", -x["quality_score"]))
    return out

def make_batches(cands: List[Dict[str, Any]], tiers: List[str], batch_size: int) -> List[Dict[str, Any]]:
    filtered = [c for c in cands if c.get("tier") in tiers]
    batches = []
    for i in range(0, len(filtered), batch_size):
        part = filtered[i:i+batch_size]
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
):
    tiers_list = [t.strip().upper() for t in tiers.split(",") if t.strip()]
    cands = build_candidates(max_markets=max_markets)
    batches = make_batches(cands, tiers=tiers_list, batch_size=batch_size)
    return {
        "run_id": _stamp(),
        "generated_at_utc": _now_utc().isoformat(),
        "tiers": tiers_list,
        "batch_size": batch_size,
        "candidates_total": len(cands),
        "batches_total": len(batches),
        "batches": batches,
    }
