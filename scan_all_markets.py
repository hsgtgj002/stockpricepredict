import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILES,
    analyze_direction,
    fetch_intraday_history,
    get_current_price,
    predict_intraday_prices,
)
from market_universe import get_universe

OUTPUT_FILE = Path(__file__).with_name("all_market_picks.json")


def scan_symbol(symbol: str) -> dict | None:
    try:
        history = fetch_intraday_history(symbol)
        current = get_current_price(symbol)
        if len(history) < 35 or not current:
            return None
        profile = MODEL_PROFILES.get("__global__", DEFAULT_MODEL_PROFILE)
        predictions = predict_intraday_prices(history, 30, profile)
        analysis = analyze_direction(current["price"], predictions)
        return {
            "symbol": symbol,
            "current_price": current["price"],
            "predicted_price": analysis["final_price"],
            "change_pct": analysis["change_pct"],
            "currency": "KRW" if symbol.endswith((".KS", ".KQ")) else "USD",
        }
    except Exception:
        return None


def main() -> None:
    symbols = get_universe(force_refresh="--refresh" in sys.argv)
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(scan_symbol, symbol) for symbol in symbols]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result:
                results.append(result)
            if index % 100 == 0:
                print(f"Scanned {index}/{len(symbols)} symbols")

    results.sort(key=lambda item: item["change_pct"], reverse=True)
    OUTPUT_FILE.write_text(
        json.dumps({"symbols_scanned": len(symbols), "results": results[:100]}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(results[:100])} ranked results from {len(symbols)} symbols")


if __name__ == "__main__":
    main()
