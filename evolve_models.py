import json
import sys
from pathlib import Path

import numpy as np

from app import (
    MODEL_PROFILE_FILE,
    fetch_intraday_history,
    genetic_forecast_parameters_multi,
)
from cnn_forecaster import train_shared_cnn
from market_universe import get_universe

def load_prices(symbol: str) -> np.ndarray | None:
    history = fetch_intraday_history(symbol)
    if len(history) < 35:
        print(f"  skipped: not enough hourly data")
        return None
    return np.array([record["close"] for record in history], dtype=float)

def main() -> None:
    symbols = sys.argv[1:]
    if not symbols:
        symbols = get_universe()

    profile_path = Path(MODEL_PROFILE_FILE)
    if profile_path.exists():
        try:
            profiles = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            profiles = {}
    else:
        profiles = {}

    price_series = []
    used_symbols = []
    for symbol in symbols:
        print(f"Loading {symbol}...")
        prices = load_prices(symbol)
        if prices is not None:
            price_series.append(prices)
            used_symbols.append(symbol)

    if not price_series:
        raise SystemExit("No usable hourly histories were downloaded")

    print(f"Evolving one shared profile from {len(price_series)} stocks for 500 generations...")
    window, weights = genetic_forecast_parameters_multi(price_series, 6, generations=500)
    print("Training shared 1D CNN-LSTM...")
    train_shared_cnn(price_series, epochs=120)
    profiles["__global__"] = {
        "window": window,
        "weights": [round(float(weight), 8) for weight in weights],
        "cnn_weight": 0.28,
        "reference_symbols": used_symbols,
        "generations": 500,
    }
    print(f"  global_profile={profiles['__global__']}")

    profile_path.write_text(
        json.dumps(profiles, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Saved one global profile trained from {len(used_symbols)} stocks to {profile_path}")


if __name__ == "__main__":
    main()
