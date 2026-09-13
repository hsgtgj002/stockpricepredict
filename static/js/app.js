let chart = null;
let currentSymbol = null;
let stockData = null;
let currentPeriod = '1w';
let searchTimeout = null;
const API_BASE_URL = (window.API_BASE_URL || '').replace(/\/$/, '');

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

const homeScreen = document.getElementById('home-screen');
const detailScreen = document.getElementById('detail-screen');
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const searchBtn = document.getElementById('search-btn');
const backBtn = document.getElementById('back-btn');
const loading = document.getElementById('loading');
const topPicksList = document.getElementById('top-picks-list');

function showLoading(show) {
  loading.classList.toggle('hidden', !show);
}

function showScreen(screen) {
  homeScreen.classList.toggle('active', screen === 'home');
  detailScreen.classList.toggle('active', screen === 'detail');
}

function formatPrice(price, currency = 'USD') {
  const symbols = { USD: '$', KRW: '₩', JPY: '¥', EUR: '€' };
  const sym = symbols[currency] || currency + ' ';
  if (currency === 'KRW') {
    return sym + Math.round(price).toLocaleString();
  }
  return sym + price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatChange(change, pct) {
  const sign = change >= 0 ? '+' : '';
  return `${sign}${change.toLocaleString(undefined, { minimumFractionDigits: 2 })} (${sign}${pct.toFixed(2)}%)`;
}

function renderTopPicks(results) {
  if (!results.length) {
    topPicksList.innerHTML = '<div class="top-picks-loading">예측 종목을 불러오지 못했습니다.</div>';
    return;
  }

  topPicksList.innerHTML = results.map((item, index) => `
    <button class="top-pick-item" data-symbol="${item.symbol}" data-name="${item.name}">
      <span class="top-pick-rank">${String(index + 1).padStart(2, '0')}</span>
      <span class="top-pick-info">
        <strong>${item.name}</strong>
        <small>${item.symbol} · 현재 ${formatPrice(item.current_price, item.currency)}</small>
      </span>
      <span class="top-pick-up">+${item.change_pct.toFixed(2)}%</span>
    </button>
  `).join('');

  topPicksList.querySelectorAll('.top-pick-item').forEach(item => {
    item.addEventListener('click', () => selectStock(item.dataset.symbol, item.dataset.name));
  });
}

async function loadTopPicks() {
  try {
    const res = await fetch(apiUrl('/api/top-picks'));
    if (!res.ok) throw new Error('top picks failed');
    const data = await res.json();
    renderTopPicks(data.results || []);
  } catch {
    renderTopPicks([]);
  }
}

async function searchStocks(query) {
  if (!query.trim()) {
    searchResults.classList.add('hidden');
    return;
  }

  try {
    const res = await fetch(apiUrl(`/api/search?q=${encodeURIComponent(query)}`));
    const data = await res.json();
    renderSearchResults(data.results);
  } catch {
    searchResults.innerHTML = '<div class="no-results">검색 중 오류가 발생했습니다.</div>';
    searchResults.classList.remove('hidden');
  }
}

function renderSearchResults(results) {
  if (!results.length) {
    searchResults.innerHTML = '<div class="no-results">검색 결과가 없습니다.</div>';
    searchResults.classList.remove('hidden');
    return;
  }

  searchResults.innerHTML = results.map(r => `
    <div class="result-item" data-symbol="${r.symbol}" data-name="${r.name}">
      <div class="result-info">
        <span class="result-name">${r.name}</span>
        <span class="result-meta">${r.exchange} · ${r.type}</span>
      </div>
      <span class="result-symbol">${r.symbol}</span>
    </div>
  `).join('');

  searchResults.classList.remove('hidden');

  searchResults.querySelectorAll('.result-item').forEach(item => {
    item.addEventListener('click', () => {
      selectStock(item.dataset.symbol, item.dataset.name);
    });
  });
}

async function selectStock(symbol, name) {
  currentSymbol = symbol;
  searchResults.classList.add('hidden');
  showLoading(true);

  try {
    const res = await fetch(apiUrl(`/api/stock/${encodeURIComponent(symbol)}`));
    if (!res.ok) throw new Error('데이터 로드 실패');
    stockData = await res.json();
    showDetailScreen(stockData, name);
    await loadPrediction(symbol, currentPeriod);
  } catch {
    alert('주식 데이터를 불러올 수 없습니다.');
  } finally {
    showLoading(false);
  }
}

function showDetailScreen(data, name) {
  showScreen('detail');

  document.getElementById('stock-name').textContent = data.name || name;
  document.getElementById('stock-symbol').textContent = data.symbol;

  const current = data.current;
  const currency = data.currency || 'USD';

  document.getElementById('current-price').textContent = formatPrice(current.price, currency);

  const changeEl = document.getElementById('price-change');
  changeEl.textContent = formatChange(current.change, current.change_pct);
  changeEl.className = 'price-change ' + (current.change >= 0 ? 'up' : 'down');
}

async function loadPrediction(symbol, period) {
  showLoading(true);
  try {
    const res = await fetch(apiUrl(`/api/predict/${encodeURIComponent(symbol)}?period=${period}`));
    if (!res.ok) throw new Error('예측 실패');
    const predData = await res.json();
    renderChart(
      stockData.history,
      predData.predictions,
      predData.prediction_base,
      stockData.intraday_history,
    );
    renderPredictionResult(predData, stockData.currency);
  } catch {
    alert('예측 데이터를 생성할 수 없습니다.');
  } finally {
    showLoading(false);
  }
}

function renderChart(history, predictions, predictionBase = null, intradayHistory = []) {
  const ctx = document.getElementById('stock-chart').getContext('2d');

  if (chart) {
    chart.destroy();
  }

  const intradayStart = intradayHistory[0]?.date?.slice(0, 10);
  const dailyHistory = intradayStart
    ? history.filter(point => point.date < intradayStart)
    : history;
  const actualHistory = [...dailyHistory, ...intradayHistory];
  const lastHistorical = predictionBase || actualHistory[actualHistory.length - 1];
  const bridgePoint = {
    x: lastHistorical.date,
    y: lastHistorical.close,
  };

  const actualData = actualHistory.map(h => ({ x: h.date, y: h.close }));
  const predictData = [
    bridgePoint,
    ...predictions.map(p => ({ x: p.date, y: p.close })),
  ];
  const actualValues = actualData.map(point => point.y);
  const actualMin = Math.min(...actualValues);
  const actualMax = Math.max(...actualValues);
  const actualSpan = Math.max(actualMax - actualMin, actualMax * 0.01, 0.01);
  const fixedXMax = new Date(lastHistorical.date);
  fixedXMax.setDate(fixedXMax.getDate() + 31);

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: '실제 주가',
          data: actualData,
          borderColor: '#4285f4',
          backgroundColor: 'rgba(66, 133, 244, 0.08)',
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 5,
          borderWidth: 2.5,
        },
        {
          label: 'AI 예측',
          data: predictData,
          borderColor: '#f9ab00',
          backgroundColor: 'rgba(251, 188, 4, 0.18)',
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointBackgroundColor: '#fff',
          pointBorderColor: '#f9ab00',
          pointBorderWidth: 2,
          borderWidth: 3,
          borderDash: [7, 4],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(ctx) {
              const val = ctx.parsed.y;
              return `${ctx.dataset.label}: ${val?.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;
            },
          },
        },
        annotation: undefined,
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: predictions.length > 14 ? 'day' : 'hour',
            displayFormats: { hour: 'M/d HH시', day: 'M/d' },
          },
          grid: { display: false },
          max: fixedXMax,
          ticks: { maxTicksLimit: 8 },
        },
        y: {
          min: actualMin - actualSpan * 0.45,
          max: actualMax + actualSpan * 0.45,
          grid: { color: '#f1f3f4' },
          ticks: {
            callback(v) {
              return v.toLocaleString();
            },
          },
        },
      },
    },
    plugins: [{
      id: 'predictionBackground',
      beforeDraw(chartInstance) {
        const { ctx, chartArea, scales } = chartInstance;
        if (!chartArea) return;

        const splitDate = lastHistorical.date;
        if (!splitDate) return;

        const xPos = scales.x.getPixelForValue(new Date(splitDate).getTime());

        ctx.save();
        ctx.fillStyle = 'rgba(251, 188, 4, 0.07)';
        ctx.fillRect(xPos, chartArea.top, chartArea.right - xPos, chartArea.bottom - chartArea.top);

        ctx.strokeStyle = 'rgba(249, 171, 0, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(xPos, chartArea.top);
        ctx.lineTo(xPos, chartArea.bottom);
        ctx.stroke();
        ctx.restore();
      },
    }],
  });
}

function renderPredictionResult(predData, currency = 'USD') {
  const { analysis } = predData;
  const dirEl = document.getElementById('prediction-direction');
  const icons = { up: '📈', down: '📉', neutral: '➡️' };
  const texts = { up: '상승 예상', down: '하락 예상', neutral: '횡보 예상' };

  dirEl.className = 'prediction-direction ' + analysis.direction;
  dirEl.innerHTML = `
    <span class="direction-icon">${icons[analysis.direction]}</span>
    <span class="direction-text">${texts[analysis.direction]}</span>
  `;

  const changeClass = analysis.change >= 0 ? 'up' : 'down';
  const sign = analysis.change >= 0 ? '+' : '';

  document.getElementById('predicted-price').textContent = formatPrice(analysis.final_price, currency);
  document.getElementById('predicted-change').textContent = `${sign}${analysis.change.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;
  document.getElementById('predicted-change').className = 'value ' + changeClass;
  document.getElementById('predicted-change-pct').textContent = `${sign}${analysis.change_pct.toFixed(2)}%`;
  document.getElementById('predicted-change-pct').className = 'value ' + changeClass;
}

// 이벤트 리스너
searchInput.addEventListener('input', () => {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => searchStocks(searchInput.value), 300);
});

searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    clearTimeout(searchTimeout);
    searchStocks(searchInput.value);
  }
});

searchBtn.addEventListener('click', () => searchStocks(searchInput.value));

backBtn.addEventListener('click', () => {
  showScreen('home');
  searchInput.value = '';
  searchResults.classList.add('hidden');
  if (chart) {
    chart.destroy();
    chart = null;
  }
});

document.querySelectorAll('.period-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentPeriod = btn.dataset.period;
    if (currentSymbol) {
      loadPrediction(currentSymbol, currentPeriod);
    }
  });
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrapper')) {
    searchResults.classList.add('hidden');
  }
});

searchInput.focus();
loadTopPicks();
