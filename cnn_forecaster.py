from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


MODEL_PATH = Path(__file__).with_name("cnn_model.pt")
SEQUENCE_LENGTH = 32


if nn is not None:
    class StockCNNLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolution = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.recurrent = nn.LSTM(64, 32, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, features):
            sequence = self.convolution(features).transpose(1, 2)
            _, (hidden, _) = self.recurrent(sequence)
            return self.head(hidden[-1]).squeeze(-1)
else:
    StockCNNLSTM = None


def make_features(prices: np.ndarray) -> np.ndarray:
    log_prices = np.log(np.maximum(prices, 0.01))
    returns = np.diff(log_prices, prepend=log_prices[0])
    features = []
    for index in range(len(prices)):
        start = max(0, index - 20)
        local = log_prices[start:index + 1]
        local_returns = returns[start:index + 1]
        local_mean = float(np.mean(local))
        local_volatility = max(float(np.std(local_returns)), 0.001)
        features.append([
            returns[index],
            (log_prices[index] - local_mean) / local_volatility,
            local_volatility,
        ])
    return np.asarray(features, dtype=np.float32)


def make_training_data(price_series: list[np.ndarray]):
    inputs = []
    targets = []
    for prices in price_series:
        if len(prices) <= SEQUENCE_LENGTH + 1:
            continue
        features = make_features(prices)
        future_returns = np.diff(np.log(prices))
        for end in range(SEQUENCE_LENGTH, len(prices) - 1):
            inputs.append(features[end - SEQUENCE_LENGTH:end].T)
            targets.append(future_returns[end])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def train_shared_cnn(price_series: list[np.ndarray], epochs: int = 80) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to train the CNN")
    torch.manual_seed(20260906)
    inputs, targets = make_training_data(price_series)
    if len(inputs) < 32:
        raise RuntimeError("Not enough data to train the CNN")

    model = StockCNNLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    input_tensor = torch.from_numpy(inputs)
    target_tensor = torch.from_numpy(targets)

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(input_tensor))
        for start in range(0, len(input_tensor), 256):
            batch = permutation[start:start + 256]
            prediction = model(input_tensor[batch])
            loss = loss_function(prediction, target_tensor[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    torch.save({"state_dict": model.state_dict()}, MODEL_PATH)


_model = None


def predict_cnn_returns(prices: np.ndarray, steps_ahead: int) -> np.ndarray | None:
    global _model
    if _model is False or torch is None or not MODEL_PATH.exists() or len(prices) < SEQUENCE_LENGTH + 1:
        return None
    if _model is None:
        try:
            _model = StockCNNLSTM()
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            _model.load_state_dict(checkpoint["state_dict"])
            _model.eval()
        except (RuntimeError, KeyError, OSError):
            # A stale checkpoint must not take down the stock prediction endpoint.
            _model = False
            return None

    rolling_prices = list(prices.astype(float))
    predictions = []
    with torch.no_grad():
        for _ in range(steps_ahead):
            features = make_features(np.asarray(rolling_prices))
            sequence = torch.from_numpy(features[-SEQUENCE_LENGTH:].T).unsqueeze(0)
            next_return = float(_model(sequence).item())
            next_return = float(np.clip(next_return, -0.04, 0.04))
            predictions.append(next_return)
            rolling_prices.append(rolling_prices[-1] * np.exp(next_return))
    return np.asarray(predictions, dtype=float)
