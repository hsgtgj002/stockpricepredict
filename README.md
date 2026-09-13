# Stock Prediction App

Flask-based stock prediction dashboard for Korean and US equities.

## Run

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Deploy

1. Deploy the repository to Render using `render.yaml` to host the Flask API.
2. Copy the Render service URL into `static/js/config.js` as `window.API_BASE_URL`.
3. Deploy the repository to Netlify with `netlify.toml` and publish directory `.`.

Netlify hosts the frontend while Render runs the Python, Yahoo Finance, and CNN-LSTM backend.

## Model training

The app uses a shared genetic-optimization profile and a pretrained 1D CNN-LSTM model. To retrain from the cached Korean and US market universe:

```powershell
python evolve_models.py
```

The training script writes the shared profile to `model_profiles.json` and the neural model to `cnn_model.pt`.

To scan the full market universe and cache ranked picks:

```powershell
python scan_all_markets.py
```

## Notes

- Market data is fetched through Yahoo Finance and public market listing sources.
- Predictions are experimental estimates, not investment advice.
- Generated market-universe and scan-result caches are excluded from Git.
