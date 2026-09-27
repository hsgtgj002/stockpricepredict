import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request

from cnn_forecaster import predict_cnn_returns

app = Flask(__name__)
MODEL_PROFILE_FILE = Path(__file__).with_name("model_profiles.json")
ALL_MARKET_PICKS_FILE = Path(__file__).with_name("all_market_picks.json")
DEFAULT_MODEL_PROFILE = {
    "window": 25,
    "weights": [0.05, 0.05, 0.42, 0.18, 0.30],
    "cnn_weight": 0.25,
}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def load_model_profiles() -> dict:
    if not MODEL_PROFILE_FILE.exists():
        return {}
    try:
        with MODEL_PROFILE_FILE.open(encoding="utf-8") as profile_file:
            profiles = json.load(profile_file)
        return profiles if isinstance(profiles, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


MODEL_PROFILES = load_model_profiles()

TOP_PICKS_LIMIT = 5
GLOBAL_SYMBOLS_LIMIT = 8
MAX_INTRADAY_STEPS = 24

PERIOD_DAYS = {
    "1d": 1,
    "1w": 7,
    "2w": 14,
    "1m": 30,
}

INTRADAY_STEPS = {
    "1d": 6,
    "1w": 18,
    "2w": 24,
    "1m": 24,
}

KOREAN_ALIASES = {
    "삼성전자": "005930.KS",
    "삼성": "005930.KS",
    "sk하이닉스": "000660.KS",
    "하이닉스": "000660.KS",
    "lg에너지솔루션": "373220.KS",
    "현대차": "005380.KS",
    "현대자동차": "005380.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "셀트리온": "068270.KS",
    "기아": "000270.KS",
    "포스코": "005490.KS",
    "lg화학": "051910.KS",
    "삼성바이오": "207940.KS",
    "삼성바이오로직스": "207940.KS",
    "애플": "AAPL",
    "테슬라": "TSLA",
    "엔비디아": "NVDA",
    "구글": "GOOGL",
    "아마존": "AMZN",
    "마이크로소프트": "MSFT",
    "메타": "META",
}

DISPLAY_NAMES = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "373220.KS": "LG에너지솔루션",
    "005380.KS": "현대차",
    "035420.KS": "네이버",
    "035720.KS": "카카오",
    "068270.KS": "셀트리온",
    "000270.KS": "기아",
    "005490.KS": "포스코홀딩스",
    "051910.KS": "LG화학",
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "MSFT": "Microsoft",
    "META": "Meta",
    "AVGO": "Broadcom",
    "AMD": "AMD",
    "NFLX": "Netflix",
}


def search_stocks(query: str) -> list[dict]:
    if not query.strip():
        return []

    query_lower = query.strip().lower()
    alias_symbol = KOREAN_ALIASES.get(query_lower) or KOREAN_ALIASES.get(query.strip())

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    search_q = alias_symbol if alias_symbol else query
    params = {"q": search_q, "quotesCount": 10, "newsCount": 0, "lang": "ko-KR"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    results = []
    seen = set()
    for item in data.get("quotes", []):
        symbol = item.get("symbol")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        results.append(
            {
                "symbol": symbol,
                "name": item.get("longname") or item.get("shortname") or symbol,
                "exchange": item.get("exchDisp", ""),
                "type": item.get("quoteType", ""),
            }
        )
    return results


def fetch_history(symbol: str, period: str = "6mo") -> list[dict]:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        return []

    records = []
    for date, row in hist.iterrows():
        records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "volume": int(row["Volume"]),
            }
        )
    return records


def fetch_intraday_history(symbol: str) -> list[dict]:
    """Load recent hourly prices for rolling short-horizon forecasts."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="60d", interval="1h", prepost=False)
    if hist.empty:
        return []

    records = []
    for date, row in hist.iterrows():
        records.append(
            {
                "date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                "close": round(float(row["Close"]), 2),
            }
        )
    return records


def get_current_price(symbol: str) -> dict | None:
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    price = getattr(info, "last_price", None)
    if price is None:
        hist = ticker.history(period="5d")
        if hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])

    prev_close = getattr(info, "previous_close", None)
    if prev_close is None and not ticker.history(period="5d").empty:
        hist = ticker.history(period="5d")
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
        else:
            prev_close = price

    change = price - prev_close if prev_close else 0
    change_pct = (change / prev_close * 100) if prev_close else 0

    return {
        "price": round(float(price), 2),
        "prev_close": round(float(prev_close), 2),
        "change": round(float(change), 2),
        "change_pct": round(float(change_pct), 2),
    }


def forecast_log_prices(prices: np.ndarray, days_ahead: int, window: int, mode: str) -> np.ndarray:
    """Forecast log prices with a small set of explainable candidate models."""
    log_prices = np.log(prices)
    recent = log_prices[-window:]
    returns = np.diff(recent)

    if mode == "trend":
        x = np.arange(len(recent), dtype=float)
        slope, intercept = np.polyfit(x, recent, 1)
        # Long-range linear extrapolation is damped to avoid unrealistic runaway prices.
        slope *= 0.78
        steps = np.arange(1, days_ahead + 1, dtype=float)
        return intercept + slope * (len(recent) - 1 + steps)

    if mode == "ar":
        # Learn how recent returns follow one another, then feed each prediction back in.
        ar_returns = np.diff(recent)
        lag_count = min(4, max(1, len(ar_returns) // 4))
        if len(ar_returns) <= lag_count + 2:
            mode = "drift"
        else:
            rows = []
            targets = []
            for index in range(lag_count, len(ar_returns)):
                rows.append([1.0, *ar_returns[index - lag_count:index]])
                targets.append(ar_returns[index])
            design = np.asarray(rows)
            ridge = np.eye(design.shape[1]) * 0.08
            ridge[0, 0] = 0
            coefficients = np.linalg.solve(
                design.T @ design + ridge,
                design.T @ np.asarray(targets),
            )
            rolling_returns = list(ar_returns[-lag_count:])
            predicted_returns = []
            for _ in range(days_ahead):
                next_return = float(np.clip(
                    coefficients[0] + np.dot(coefficients[1:], rolling_returns[-lag_count:]),
                    -0.035,
                    0.035,
                ))
                predicted_returns.append(next_return)
                rolling_returns.append(next_return)
            return log_prices[-1] + np.cumsum(predicted_returns)

    if mode == "analog":
        all_returns = np.diff(log_prices)
        all_returns = all_returns[-180:]
        pattern_length = min(6, max(3, window // 3))
        if len(all_returns) > pattern_length + days_ahead:
            target = all_returns[-pattern_length:]
            target_scale = max(float(np.std(target)), 0.0005)
            candidates = []
            for start in range(0, len(all_returns) - pattern_length - days_ahead + 1):
                pattern = all_returns[start:start + pattern_length]
                pattern_scale = max(float(np.std(pattern)), 0.0005)
                distance = np.mean(
                    ((pattern - np.mean(pattern)) / pattern_scale
                     - (target - np.mean(target)) / target_scale) ** 2
                )
                continuation = all_returns[start + pattern_length:start + pattern_length + days_ahead]
                candidates.append((distance, continuation))
            _, continuation = min(candidates, key=lambda item: item[0])
            return log_prices[-1] + np.cumsum(continuation)

    if mode == "technical":
        # Combine trend, RSI-style momentum, and Bollinger-style mean reversion.
        returns = np.diff(log_prices)
        fast = min(8, len(log_prices))
        slow = min(21, len(log_prices))
        fast_ema = float(np.mean(log_prices[-fast:]))
        slow_ema = float(np.mean(log_prices[-slow:]))
        volatility = max(float(np.std(returns[-20:])), 0.0005)
        gains = np.maximum(returns[-14:], 0)
        losses = np.maximum(-returns[-14:], 0)
        relative_strength = float(np.mean(gains) / max(np.mean(losses), 1e-6))
        rsi = 100 - (100 / (1 + relative_strength))
        band_mean = float(np.mean(log_prices[-20:]))
        band_z = (log_prices[-1] - band_mean) / max(2 * volatility, 0.001)
        trend_signal = np.tanh((fast_ema - slow_ema) / max(volatility * 8, 0.001))
        momentum_signal = np.clip((rsi - 50) / 50, -1, 1)
        reversion_signal = np.clip(-band_z, -1, 1)
        signal = 0.45 * trend_signal + 0.35 * momentum_signal + 0.20 * reversion_signal
        base_return = float(np.clip(signal * volatility * 0.85, -0.035, 0.035))
        steps = np.arange(1, days_ahead + 1, dtype=float)
        return log_prices[-1] + base_return * (1 - 0.08 * np.minimum(steps - 1, 8)) * steps

    # Exponentially weighted drift reacts to a genuine recent move without inventing cycles.
    weights = np.exp(np.linspace(-1.5, 0, len(returns)))
    drift = float(np.average(returns, weights=weights)) if len(returns) else 0.0
    drift = float(np.clip(drift, -0.035, 0.035))
    steps = np.arange(1, days_ahead + 1, dtype=float)
    return log_prices[-1] + drift * steps


def select_forecast_model(prices: np.ndarray, days_ahead: int) -> tuple[int, str]:
    candidates = [
        (window, mode)
        for window in (8, 15, 25, 40, 60)
        for mode in ("drift", "trend", "ar")
    ]
    min_history = max(35, days_ahead * 3)
    origins = range(min_history, len(prices) - days_ahead + 1, max(1, days_ahead // 2))
    scores = {candidate: [] for candidate in candidates}

    for origin in origins:
        training = prices[:origin]
        actual = np.log(prices[origin:origin + days_ahead])
        for candidate in candidates:
            window, mode = candidate
            if len(training) <= window:
                continue
            predicted = forecast_log_prices(training, len(actual), window, mode)
            scores[candidate].append(float(np.mean(np.abs(predicted - actual))))

    valid_scores = {
        candidate: float(np.mean(errors))
        for candidate, errors in scores.items()
        if errors
    }
    return min(valid_scores, key=valid_scores.get) if valid_scores else (min(25, len(prices) - 1), "drift")


def genetic_forecast_parameters(prices: np.ndarray, days_ahead: int) -> tuple[int, np.ndarray]:
    """Optimize model weights against hidden historical futures with a small genetic search."""
    modes = ("drift", "trend", "ar", "analog", "technical")
    windows = (15, 25, 40, 60)
    # Deliberately thorough search: this runs once per rolling model refresh,
    # not once per individual forecast point.
    population_size = 48
    generations = 40
    elite_count = 8
    mutation_rate = 0.04
    rng = np.random.default_rng(42)
    population = rng.dirichlet(np.ones(len(modes)), size=population_size)
    origins = list(range(max(40, days_ahead * 3), len(prices) - days_ahead + 1, max(1, days_ahead)))
    origins = origins[-8:]
    validation_sets = []
    for origin in origins:
        training = prices[:origin]
        actual = np.log(prices[origin:origin + days_ahead])
        forecasts_by_window = {}
        for window in windows:
            if len(training) > window:
                forecasts_by_window[window] = np.array([
                    forecast_log_prices(training, len(actual), window, mode)
                    for mode in modes
                ])
        validation_sets.append((actual, forecasts_by_window))

    def score(window: int, weights: np.ndarray) -> float:
        errors = []
        for actual, forecasts_by_window in validation_sets:
            forecasts = forecasts_by_window.get(window)
            if forecasts is None:
                continue
            combined = np.average(forecasts, axis=0, weights=weights)
            errors.append(float(np.mean(np.abs(combined - actual))))
        return float(np.mean(errors)) if errors else float("inf")

    best = (float("inf"), windows[1], population[0])
    for _ in range(generations):
        scored = []
        for weights in population:
            for window in windows:
                result = (score(window, weights), window, weights.copy())
                scored.append(result)
                if result[0] < best[0]:
                    best = result
        scored.sort(key=lambda item: item[0])
        parents = [item[2] for item in scored[:elite_count]]
        next_population = parents.copy()
        while len(next_population) < population_size:
            first = parents[int(rng.integers(0, len(parents)))]
            second = parents[int(rng.integers(0, len(parents)))]
            child = (first + second) / 2
            child += rng.normal(0, mutation_rate, size=len(modes))
            child = np.clip(child, 0.001, None)
            next_population.append(child / child.sum())
        population = np.asarray(next_population)

    return best[1], best[2] / best[2].sum()


def genetic_forecast_parameters_multi(
    price_series: list[np.ndarray],
    days_ahead: int,
    generations: int = 200,
) -> tuple[int, np.ndarray]:
    """Evolve one general profile from many stocks and hidden historical futures."""
    modes = ("drift", "trend", "ar", "analog", "technical")
    windows = (15, 25, 40, 60)
    population_size = 64
    elite_count = 10
    rng = np.random.default_rng(20260906)
    population = rng.dirichlet(np.ones(len(modes)), size=population_size)
    validation_sets = []

    for prices in price_series:
        origins = list(range(max(40, days_ahead * 3), len(prices) - days_ahead + 1, max(1, days_ahead)))
        for origin in origins[-6:]:
            training = prices[:origin]
            actual = np.log(prices[origin:origin + days_ahead])
            forecasts_by_window = {}
            for window in windows:
                if len(training) > window:
                    forecasts_by_window[window] = np.array([
                        forecast_log_prices(training, len(actual), window, mode)
                        for mode in modes
                    ])
            validation_sets.append((actual, forecasts_by_window))

    def score(window: int, weights: np.ndarray) -> float:
        errors = []
        for actual, forecasts_by_window in validation_sets:
            forecasts = forecasts_by_window.get(window)
            if forecasts is not None:
                combined = np.average(forecasts, axis=0, weights=weights)
                errors.append(float(np.mean(np.abs(combined - actual))))
        return float(np.mean(errors)) if errors else float("inf")

    best = (float("inf"), windows[1], population[0].copy())
    for _ in range(generations):
        scored = []
        for weights in population:
            for window in windows:
                result = (score(window, weights), window, weights.copy())
                scored.append(result)
                if result[0] < best[0]:
                    best = result
        scored.sort(key=lambda item: item[0])
        parents = [item[2] for item in scored[:elite_count]]
        next_population = parents.copy()
        while len(next_population) < population_size:
            first = parents[int(rng.integers(0, len(parents)))]
            second = parents[int(rng.integers(0, len(parents)))]
            child = (first + second) / 2
            child += rng.normal(0, 0.035, size=len(modes))
            child = np.clip(child, 0.001, None)
            next_population.append(child / child.sum())
        population = np.asarray(next_population)

    return best[1], best[2] / best[2].sum()


def ensemble_forecast(prices: np.ndarray, days_ahead: int, window: int, weights: np.ndarray) -> np.ndarray:
    modes = ("drift", "trend", "ar", "analog", "technical")
    forecasts = np.array([
        forecast_log_prices(prices, days_ahead, window, mode)
        for mode in modes
    ])
    forecast = np.average(forecasts, axis=0, weights=weights)
    cnn_returns = predict_cnn_returns(prices, days_ahead)
    if cnn_returns is not None:
        cnn_forecast = np.log(prices[-1]) + np.cumsum(cnn_returns)
        forecast = forecast * 0.72 + cnn_forecast * 0.28
    returns = np.diff(np.log(prices))
    recent_returns = returns[-48:]
    if len(recent_returns) >= 8:
        recent_volatility = max(float(np.std(recent_returns)), 0.001)
        baseline_volatility = max(float(np.std(returns[-192:])), 0.001)
        volatility_ratio = float(np.clip(recent_volatility / baseline_volatility, 0.8, 2.5))
        residuals = recent_returns - np.mean(recent_returns)
        seed = int(abs(float(np.sum(prices[-8:])) * 1000)) % (2**32)
        rng = np.random.default_rng(seed)
        block_length = min(3, len(residuals))
        block_starts = rng.integers(0, len(residuals) - block_length + 1, size=days_ahead)
        shocks = np.concatenate([
            residuals[start:start + block_length]
            for start in block_starts
        ])[:days_ahead]
        forecast = forecast + np.cumsum(shocks * 0.72 * volatility_ratio)
    return forecast


def predict_prices(historical: list[dict], days_ahead: int) -> list[dict]:
    if len(historical) < 10:
        return []

    closes = np.array([r["close"] for r in historical], dtype=float)
    window, mode = select_forecast_model(closes, days_ahead)
    forecast = forecast_log_prices(closes, days_ahead, window, mode)

    last_date = datetime.strptime(historical[-1]["date"], "%Y-%m-%d")
    predictions = []
    pred_date = last_date
    for value in forecast:
        pred_date += timedelta(days=1)
        while pred_date.weekday() >= 5:
            pred_date += timedelta(days=1)
        predictions.append(
            {
                "date": pred_date.strftime("%Y-%m-%d"),
                "close": round(max(float(np.exp(value)), 0.01), 2),
                "is_prediction": True,
            }
        )

    return predictions


def predict_intraday_prices(
    historical: list[dict],
    steps_ahead: int,
    profile: dict | None = None,
) -> list[dict]:
    """Use a pre-evolved profile while recursively feeding each forecast forward."""
    if len(historical) < 35:
        return []

    prices = np.array([record["close"] for record in historical], dtype=float)
    last_date = datetime.fromisoformat(historical[-1]["date"])
    predictions = []

    steps_remaining = steps_ahead
    selected_profile = profile or DEFAULT_MODEL_PROFILE
    window = int(selected_profile.get("window", DEFAULT_MODEL_PROFILE["window"]))
    ensemble_weights = np.asarray(
        selected_profile.get("weights", DEFAULT_MODEL_PROFILE["weights"]),
        dtype=float,
    )
    ensemble_weights = ensemble_weights / ensemble_weights.sum()
    while steps_remaining:
        block_size = min(3, steps_remaining)
        block_forecast = ensemble_forecast(prices, block_size, window, ensemble_weights)

        for next_log_price in block_forecast:
            next_price = max(float(np.exp(next_log_price)), 0.01)
            prices = np.append(prices, next_price)

            last_date += timedelta(hours=1)
            if last_date.hour >= 16:
                last_date = (last_date + timedelta(days=1)).replace(hour=9, minute=30, second=0)
            elif last_date.hour < 9:
                last_date = last_date.replace(hour=9, minute=30, second=0)
            while last_date.weekday() >= 5:
                last_date = (last_date + timedelta(days=1)).replace(hour=9, minute=30, second=0)
            predictions.append(
                {
                    "date": last_date.strftime("%Y-%m-%dT%H:%M:%S"),
                    "close": round(next_price, 2),
                    "is_prediction": True,
                }
            )
        steps_remaining -= block_size

    return predictions


def analyze_direction(current: float, predicted: list[dict]) -> dict:
    if not predicted:
        return {"direction": "neutral", "change": 0, "change_pct": 0}

    final_price = predicted[-1]["close"]
    change = final_price - current
    change_pct = (change / current * 100) if current else 0

    if change_pct > 1.0:
        direction = "up"
    elif change_pct < -1.0:
        direction = "down"
    else:
        direction = "neutral"

    return {
        "direction": direction,
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "final_price": round(final_price, 2),
    }


def rank_stock_prediction(symbol: str) -> dict | None:
    intraday_history = fetch_intraday_history(symbol)
    current = get_current_price(symbol)
    if not intraday_history or not current:
        return None
    steps = min(INTRADAY_STEPS["1w"], MAX_INTRADAY_STEPS)
    predictions = predict_intraday_prices(
        intraday_history,
        steps,
        MODEL_PROFILES.get("__global__", DEFAULT_MODEL_PROFILE),
    )
    analysis = analyze_direction(current["price"], predictions)
    return {
        "symbol": symbol,
        "name": DISPLAY_NAMES.get(symbol, symbol),
        "currency": "KRW" if symbol.endswith((".KS", ".KQ")) else "USD",
        "current_price": current["price"],
        "predicted_price": analysis.get("final_price", current["price"]),
        "change_pct": analysis["change_pct"],
        "direction": analysis["direction"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "")
    results = search_stocks(query)
    return jsonify({"results": results})


@app.route("/api/top-picks")
def api_top_picks():
    if ALL_MARKET_PICKS_FILE.exists():
        try:
            cached = json.loads(ALL_MARKET_PICKS_FILE.read_text(encoding="utf-8"))
            cached_results = cached.get("results", [])
            if cached_results:
                return jsonify({"results": cached_results[:5]})
        except (OSError, json.JSONDecodeError):
            pass

    symbols = MODEL_PROFILES.get("__global__", {}).get("reference_symbols", [])[:GLOBAL_SYMBOLS_LIMIT]
    ranked = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(rank_stock_prediction, symbol) for symbol in symbols]
        for future in as_completed(futures):
            try:
                result = future.result(timeout=20)
            except Exception:
                result = None
            if result:
                ranked.append(result)
    ranked.sort(key=lambda item: item["change_pct"], reverse=True)
    return jsonify({"results": ranked[:TOP_PICKS_LIMIT]})


@app.route("/api/stock/<symbol>")
def api_stock(symbol):
    history = fetch_history(symbol)
    if not history:
        return jsonify({"error": "데이터를 불러올 수 없습니다."}), 404

    current = get_current_price(symbol)
    intraday_history = fetch_intraday_history(symbol)
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return jsonify(
        {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "currency": info.get("currency", "USD"),
            "current": current,
            "history": history,
            "intraday_history": intraday_history,
        }
    )


@app.route("/api/predict/<symbol>")
def api_predict(symbol):
    period_key = request.args.get("period", "1w")
    days = PERIOD_DAYS.get(period_key, 7)

    intraday_history = fetch_intraday_history(symbol)
    if intraday_history:
        profile = MODEL_PROFILES.get("__global__", DEFAULT_MODEL_PROFILE)
        steps = min(INTRADAY_STEPS.get(period_key, 18), MAX_INTRADAY_STEPS)
        predictions = predict_intraday_prices(
            intraday_history,
            steps,
            profile,
        )
    else:
        history = fetch_history(symbol)
        predictions = predict_prices(history, days) if history else []

    if not predictions:
        return jsonify({"error": "데이터를 불러올 수 없습니다."}), 404

    current = get_current_price(symbol)

    if not current or not predictions:
        return jsonify({"error": "예측을 생성할 수 없습니다."}), 400

    analysis = analyze_direction(current["price"], predictions)

    return jsonify(
        {
            "symbol": symbol,
            "period": period_key,
            "days": days,
            "current": current,
            "predictions": predictions,
            "prediction_base": intraday_history[-1] if intraday_history else None,
            "analysis": analysis,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
