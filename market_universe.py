import json
import io
import time
from pathlib import Path

import requests
import pandas as pd

CACHE_FILE = Path(__file__).with_name("market_universe.json")
CACHE_TTL = 24 * 60 * 60


def _download_text(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()
    return response.text


def _us_symbols() -> list[str]:
    symbols = []
    nasdaq = _download_text("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt")
    for row in nasdaq.splitlines()[1:]:
        parts = row.split("|")
        if len(parts) > 6 and parts[3] == "N" and parts[4] == "N" and parts[6] == "N":
            symbols.append(parts[0])

    other = _download_text("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt")
    for row in other.splitlines()[1:]:
        parts = row.split("|")
        if len(parts) > 6 and parts[4] == "N" and parts[6] == "N":
            symbols.append(parts[0])
    return symbols


def _krx_symbols() -> list[str]:
    response = requests.get(
        "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    listing = pd.read_html(io.BytesIO(response.content), encoding="euc-kr")[0]
    symbols = []
    for _, row in listing.iterrows():
        market = str(row["시장구분"])
        if market not in ("코스피", "코스닥"):
            continue
        ticker = str(row["종목코드"]).strip()
        if not ticker.isdigit():
            continue
        suffix = "KS" if market == "코스피" else "KQ"
        symbols.append(f"{ticker.zfill(6)}.{suffix}")
    return symbols


def refresh_universe() -> list[str]:
    symbols = sorted(set(_krx_symbols() + _us_symbols()))
    CACHE_FILE.write_text(
        json.dumps({"updated": int(time.time()), "symbols": symbols}, ensure_ascii=True),
        encoding="utf-8",
    )
    return symbols


def get_universe(force_refresh: bool = False) -> list[str]:
    if not force_refresh and CACHE_FILE.exists():
        try:
            payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - payload.get("updated", 0) < CACHE_TTL:
                return payload.get("symbols", [])
        except (OSError, json.JSONDecodeError):
            pass
    return refresh_universe()
