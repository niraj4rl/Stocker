const state = {
  mode: "live",
  liveChart: null,
  monthForecastChart: null,
  equityChart: null,
  tickers: [],
};

const signalColors = {
  Buy: "#16A34A",
  Hold: "#F59E0B",
  Sell: "#DC2626",
};

const $ = (id) => document.getElementById(id);
const formatINR = (value) => new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", minimumFractionDigits: 2, maximumFractionDigits: 2,
}).format(Number(value));
const signedPercent = (value, digits = 2) => {
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
};

function setStatus(msg, isError = false) {
  const bar = $("statusBar");
  bar.textContent = msg;
  bar.style.color = isError ? "#DC2626" : "#6B7280";
}

async function api(path, method = "GET", body = null) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function switchMode(mode) {
  state.mode = mode;
  const livePanel = $("livePanel");
  if (livePanel) {
    livePanel.classList.toggle("hidden", mode !== "live");
  }
}

function asTable(rows, columns) {
  if (!rows || !rows.length) {
    return "<p class='text-sm text-ink/70'>No data.</p>";
  }

  const thead = columns.map((col) => `<th>${col.label}</th>`).join("");
  const tbody = rows
    .map((row) => {
      const tds = columns
        .map((col) => {
          const value = row[col.key];
          if (value === null || value === undefined) {
            return "<td>-</td>";
          }
          if (typeof value === "number") {
            return `<td>${value.toFixed(3)}</td>`;
          }
          const regimeClass = col.key === "regime"
            ? ` class="regime-${String(value).toLowerCase().replaceAll(" ", "").replaceAll("_", "")}"`
            : "";
          return `<td${regimeClass}>${String(value)}</td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");

  return `<table class='table'><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
}

function renderLiveChart(points) {
  const ctx = $("liveChart").getContext("2d");
  if (state.liveChart) {
    state.liveChart.destroy();
  }

  const labels = points.map((p) => p.date);
  const close = points.map((p) => p.close);
  const area = ctx.createLinearGradient(0, 0, 0, 340);
  area.addColorStop(0, "rgba(34, 197, 94, 0.18)");
  area.addColorStop(1, "rgba(34, 197, 94, 0)");

  state.liveChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Close",
          data: close,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#b7791f",
          borderWidth: 1.5,
          backgroundColor: area,
          fill: true,
          pointRadius: 0,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: { x: { display: false }, y: { grid: { color: "rgba(128,128,128,0.16)" } } },
    },
  });
}

function renderMonthForecastChart(curve) {
  const canvas = $("monthForecastChart");
  if (!canvas || !curve || !curve.points?.length) return;
  const ctx = canvas.getContext("2d");
  if (state.monthForecastChart) state.monthForecastChart.destroy();

  const history = curve.historical_points || [];
  const labels = [
    ...history.map((point) => point.date),
    ...curve.points.map((point) => point.date),
  ];
  const actualValues = [
    ...history.map((point) => Number(point.close)),
    ...curve.points.map(() => null),
  ];
  const historyLastClose = history.length
    ? Number(history[history.length - 1].close)
    : Number(curve.current_price);
  const toChangePct = (price) => ((Number(price) / historyLastClose) - 1) * 100;
  const forecastValues = [
    ...history.slice(0, -1).map(() => null),
    0,
    ...curve.points.map((point) => toChangePct(point.predicted_price)),
  ];
  const lowerValues = [
    ...history.slice(0, -1).map(() => null),
    0,
    ...curve.points.map((point) => toChangePct(point.lower_price)),
  ];
  const upperValues = [
    ...history.slice(0, -1).map(() => null),
    0,
    ...curve.points.map((point) => toChangePct(point.upper_price)),
  ];
  const historyChanges = actualValues.map((value) => (
    value === null ? null : toChangePct(value)
  ));

  state.monthForecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual close",
          data: historyChanges,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || "#6b706b",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "Forecasted price",
          data: forecastValues,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#b7791f",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: (context) => context.dataIndex === history.length - 1 ? 4 : 2,
          tension: 0.2,
        },
        {
          label: "Validation-error range",
          data: upperValues,
          borderColor: "transparent",
          backgroundColor: "rgba(128, 128, 128, 0.10)",
          pointRadius: 0,
          borderWidth: 1,
          fill: "-1",
          tension: 0.2,
        },
        {
          label: "Validation-error lower bound",
          data: lowerValues,
          borderColor: "transparent",
          pointRadius: 0,
          borderWidth: 1,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => {
              const raw = Number(context.raw);
              const price = historyLastClose * (1 + raw / 100);
              return `${raw >= 0 ? "+" : ""}${raw.toFixed(2)}% · ${formatINR(price)}`;
            },
          },
        },
      },
      scales: {
        x: { title: { display: false }, grid: { display: false } },
        y: {
          title: { display: true, text: "Change from latest close" },
          ticks: { callback: (value) => `${value >= 0 ? "+" : ""}${Number(value).toFixed(1)}%` },
          grid: { color: "rgba(128,128,128,0.16)" },
        },
      },
    },
  });
}

function renderEquityChart(allReturns) {
  const ctx = $("equityChart").getContext("2d");
  if (state.equityChart) {
    state.equityChart.destroy();
  }

  const palette = {
    adaptive: "#16A34A",
    buy_and_hold: "#2563EB",
    static_regression: "#F59E0B",
    static_classification: "#6B7280",
    naive_regime_regression: "#94A3B8",
  };

  const datasets = Object.entries(allReturns)
    .filter(([, series]) => series && series.length)
    .map(([name, series]) => {
      let cumulative = 1;
      const values = series.map((x) => {
        cumulative *= 1 + x;
        return cumulative;
      });
      return {
        label: name.replaceAll("_", " "),
        data: values,
        borderColor: palette[name] || "#888",
        borderWidth: name === "adaptive" ? 2.8 : 1.8,
        pointRadius: 0,
        tension: 0.25,
      };
    });

  state.equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: datasets[0] ? datasets[0].data.map((_, i) => i + 1) : [],
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        x: { title: { display: true, text: "Trading Days" } },
        y: { grid: { color: "rgba(209,213,219,0.5)" } },
      },
    },
  });
}

function updateDatalist(query) {
  const list = $("tickerList");
  list.innerHTML = "";
  const cleanQuery = (query || "").trim().toUpperCase();

  const filtered = state.tickers.filter((t) => {
    const base = t.replace(".NS", "");
    return t.startsWith(cleanQuery) || base.startsWith(cleanQuery);
  });

  filtered.slice(0, 100).forEach((ticker) => {
    const option = document.createElement("option");
    option.value = ticker;
    list.appendChild(option);
  });
}

async function loadTickers() {
  const data = await api("/api/tickers");
  state.tickers = data.tickers || [];

  updateDatalist("");

  $("tickerInput").value = data.default || "RELIANCE.NS";

  const quick = $("quickPicks");
  quick.innerHTML = "";
  (data.quick_picks || []).forEach((ticker) => {
    const btn = document.createElement("button");
    btn.className = "btn-secondary text-xs";
    btn.textContent = ticker.replace(".NS", "");
    btn.addEventListener("click", () => {
      $("tickerInput").value = ticker;
      updateDatalist(ticker);
    });
    quick.appendChild(btn);
  });

}

const priceSourceLabels = {
  upstox_ltp: { label: "Upstox LTP", color: "#16A34A" },
  yfinance_live: { label: "Live • Realtime Quote", color: "#16A34A" },
  yahoo_chart_live: { label: "Live • Yahoo Chart", color: "#16A34A" },
  nse_quote: { label: "Live • NSE Quote", color: "#16A34A" },
  historical_close: { label: "Historical Close", color: "#D97706" },
};

async function runLivePrediction(forceRefresh = false) {
  const ticker = $("tickerInput").value.trim().toUpperCase();
  if (!ticker) {
    setStatus("Enter a ticker first.", true);
    return;
  }

  const payload = {
    ticker,
    force_refresh: Boolean(forceRefresh),
  };

  try {
    const actionDesc = forceRefresh ? "Refreshing live data and prediction" : "Running live prediction";
    setStatus(`${actionDesc} for ${payload.ticker}...`);
    const data = await api("/api/live-prediction", "POST", payload);

    const result = data.result;
    const cache = data.cache || {};
    const srcInfo = priceSourceLabels[result.current_price_source] || {
      label: result.current_price_source || "Market Data",
      color: "#2563EB",
    };

    const priceCardHtml = `<div class='metric'><div class='metric-label'>Current price</div><div class='metric-value'>${formatINR(result.current_price)}</div></div>`;

    const otherMetrics = [
      { label: "Market mood", value: result.regime },
      { label: "Strategy", value: result.paradigm },
      { label: "Method used", value: result.ml_model || "Price history" },
    ];

    $("liveMetrics").innerHTML =
      priceCardHtml +
      otherMetrics
        .map(
          (m) =>
            `<div class='metric'><div class='metric-label'>${m.label}</div><div class='metric-value'>${m.value}</div></div>`
        )
        .join("");

    const signalColor = signalColors[result.signal] || "#B7791F";
    document.documentElement.style.setProperty("--accent", signalColor);
    $("stockTitle").textContent = result.ticker || payload.ticker;
    $("stockPrice").textContent = formatINR(result.current_price);
    const displayedSignal = result.signal_reliable === false ? "No reliable signal" : (result.signal || "Hold");
    $("signalBadge").innerHTML = `<span class='verdict'><i class='verdict-dot'></i>${displayedSignal}</span>`;

    if (result.paradigm === "regression") {
      const ret = Number(result.next_day_predicted_return_pct || result.predicted_return_pct || 0);
      $("predictionText").textContent = `Possible next-day move ${signedPercent(ret, 3)}. The verdict above refers to the next-month outlook.`;
    } else {
      $("predictionText").textContent = `Next-day direction: ${result.prediction || "-"}. The verdict above refers to the next-month outlook.`;
    }

    const routing = Object.entries(result.routing_table || {})
      .map(([regime, paradigm]) => `<div class='flex justify-between'><span>${regime}</span><span class='text-ink/70'>${paradigm}</span></div>`)
      .join("");
    $("routingTable").innerHTML = routing || "<p class='text-sm text-ink/70'>No routing table.</p>";

    const refreshDesc = forceRefresh ? "force refreshed" : (cache.hit ? "cache hit" : "retrained");
    setStatus(`Live prediction completed for ${payload.ticker} (${srcInfo.label}, ${refreshDesc}).`);

    renderLiveChart((data.chart && data.chart.points) || []);

    const statsCols = [
      { key: "regime", label: "Condition" },
      { key: "mean_daily_return", label: "Avg daily return" },
      { key: "mean_vol_ann", label: "Volatility" },
      { key: "pct_of_days", label: "% of days" },
    ];
    const regimeLabels = { Bull: "Bull", Bear: "Bear", HighVol: "High Vol" };
    const regimeStatsByName = Object.fromEntries((data.regime_stats || []).map((row) => [row.regime, row]));
    const regimeRows = ["Bull", "Bear", "HighVol"].map((name) => ({
      ...(regimeStatsByName[name] || { count: 0, pct_of_days: 0, mean_daily_return: null, mean_vol_ann: null }),
      regime: regimeLabels[name],
    }));
    const regimeCountTotal = regimeRows.reduce((sum, row) => sum + Number(row.count || 0), 0);
    if (regimeCountTotal > 0) {
      let roundedTotal = 0;
      regimeRows.forEach((row, index) => {
        if (index === regimeRows.length - 1) {
          row.pct_of_days = Number((100 - roundedTotal).toFixed(1));
        } else {
          row.pct_of_days = Number(((Number(row.count || 0) / regimeCountTotal) * 100).toFixed(1));
          roundedTotal += row.pct_of_days;
        }
      });
    }
    $("regimeStats").innerHTML =
      asTable(regimeRows, statsCols) +
      "<p class='regime-total'>Total share of days: <strong>100.0%</strong></p>";

    const month = result.one_month_forecast;
    if (month && !month.error) {
      const returnPct = Number(month.predicted_return_pct);
      const returnClass = returnPct >= 0 ? "text-sage" : "text-rose";
      $("verdictReturn").textContent = signedPercent(returnPct, 2);
      $("verdictTarget").textContent = formatINR(month.predicted_price);
      const dayReturn = Number(result.predicted_return_pct || 0);
      $("stockChange").textContent = `Next-day view ${signedPercent(dayReturn, 2)}`;
      $("stockChange").className = `stock-change ${dayReturn >= 0 ? "gain" : "loss"}`;
      $("liveMetrics").innerHTML = [
        `<div class="metric"><div class="metric-label">Expected return</div><div class="metric-value ${returnClass}">${signedPercent(returnPct, 2)}</div></div>`,
        `<div class="metric"><div class="metric-label">Target price</div><div class="metric-value">${formatINR(month.predicted_price)}</div></div>`,
        `<div class="metric"><div class="metric-label">Model used</div><div class="metric-value model-name">${result.ml_model || "Price history"}</div></div>`,
        `<div class="metric"><div class="metric-label">Market regime</div><div class="metric-value">${regimeLabels[result.regime] || result.regime || "Neutral"}</div></div>`,
      ].join("");
      $("monthForecast").innerHTML = `
        <div class="forecast-stats">
          <div>
            <p class="label">Expected return</p>
            <p class="metric-value ${returnClass}">${signedPercent(returnPct, 3)}</p>
          </div>
          <div>
            <p class="label">Target price</p>
            <p class="metric-value">${formatINR(month.predicted_price)}</p>
          </div>
        </div>
        <p class="forecast-meta">
          Based on recent price movement · Updated ${month.as_of || "recently"}
        </p>
        <div class="chart-container">
          <canvas id="monthForecastChart"></canvas>
        </div>
        <p class="forecast-meta">
          The chart shows percentage change from the latest close. The dashed line is the point estimate; the shaded cone shows the uncertainty range.
        </p>`;
      renderMonthForecastChart(month.curve);
    } else {
      $("monthForecast").textContent = month?.error || "One-month forecast unavailable.";
    }

  } catch (err) {
    setStatus(err.message, true);
  }
}

async function runBacktest() {
  const ticker = $("tickerInput").value.trim().toUpperCase();
  if (!ticker) {
    setStatus("Enter a ticker first.", true);
    return;
  }

  try {
    setStatus(`Running fast backtest for ${ticker}...`);
    const data = await api("/api/backtest", "POST", { ticker, mode: "fast" });

    const rows = Object.entries(data.overall || {}).map(([strategy, stats]) => ({
      strategy,
      sharpe: stats.sharpe,
      max_drawdown: stats.max_drawdown,
      total_return: stats.total_return,
      calmar: stats.calmar,
    }));

    const cols = [
      { key: "strategy", label: "Strategy" },
      { key: "sharpe", label: "Sharpe" },
      { key: "max_drawdown", label: "Max Drawdown" },
      { key: "total_return", label: "Total Return" },
      { key: "calmar", label: "Calmar" },
    ];

    $("backtestSummary").innerHTML = asTable(rows, cols);

    renderEquityChart(data.all_returns || {});
    setStatus(`Backtest completed for ${ticker} (${data.profile || "fast"} mode).`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function loadAnalysis() {
  const ticker = $("analysisTicker").value;
  const query = ticker ? `?ticker=${encodeURIComponent(ticker)}` : "";

  try {
    setStatus("Loading model analysis...");
    const data = await api(`/api/analysis${query}`);

    if (!data.has_data) {
      $("leaderboardTable").innerHTML = `<p class='text-sm text-ink/70'>${data.message || "No data found."}</p>`;
      $("statsTable").innerHTML = "";
      setStatus("No analysis data yet. Run a backtest first.");
      return;
    }

    const tickerSelect = $("analysisTicker");
    const current = tickerSelect.value;
    tickerSelect.innerHTML = `<option value=''>All Tickers</option>`;
    (data.tickers || []).forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      tickerSelect.appendChild(opt);
    });
    tickerSelect.value = current;

    const leaderCols = [
      { key: "model_name", label: "Model" },
      { key: "paradigm", label: "Paradigm" },
      { key: "regime", label: "Regime" },
      { key: "avg_unified_score", label: "Unified Score" },
      { key: "avg_sharpe", label: "Avg Sharpe" },
      { key: "avg_return", label: "Avg Return" },
      { key: "avg_hit_rate", label: "Avg Hit Rate" },
      { key: "n_evaluations", label: "N" },
      { key: "times_selected", label: "Selected" },
    ];
    $("leaderboardTable").innerHTML = asTable(data.leaderboard || [], leaderCols);

    const statRows = Object.entries(data.stats || {}).map(([regime, s]) => ({
      regime,
      n_pairs: s.n_pairs,
      reg_mean_sharpe: s.reg_mean_sharpe,
      cls_mean_sharpe: s.cls_mean_sharpe,
      paired_ttest_pval: s.paired_ttest_pval,
      wilcoxon_pval: s.wilcoxon_pval,
    }));
    const statCols = [
      { key: "regime", label: "Regime" },
      { key: "n_pairs", label: "Pairs" },
      { key: "reg_mean_sharpe", label: "Reg Sharpe" },
      { key: "cls_mean_sharpe", label: "Cls Sharpe" },
      { key: "paired_ttest_pval", label: "T-Test p" },
      { key: "wilcoxon_pval", label: "Wilcoxon p" },
    ];
    $("statsTable").innerHTML = asTable(statRows, statCols);

    setStatus("Model analysis loaded.");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function exportCsv(path) {
  const ticker = $("analysisTicker").value || null;
  try {
    const data = await api(path, "POST", { ticker });
    setStatus(`Exported successfully: ${data.path}`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function bindEvents() {
  $("runLive").addEventListener("click", () => runLivePrediction(false));
  const refreshBtn = $("forceRefreshBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => runLivePrediction(true));
  }
  $("tickerInput").addEventListener("input", (e) => {
    updateDatalist(e.target.value);
  });
}

async function init() {
  bindEvents();
  switchMode("live");
  await loadTickers();

  const params = new URLSearchParams(window.location.search);
  const tickerParam = params.get("ticker");
  if (tickerParam) {
    const cleanTicker = tickerParam.trim().toUpperCase();
    $("tickerInput").value = cleanTicker;
    await runLivePrediction(false);
  } else {
    await runLivePrediction(false);
  }
}

init();
