const state = {
  mode: "live",
  liveChart: null,
  equityChart: null,
  tickers: [],
};

const signalColors = {
  Buy: "#16A34A",
  Hold: "#6B7280",
  Sell: "#DC2626",
};

const $ = (id) => document.getElementById(id);

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
          return `<td>${String(value)}</td>`;
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

  state.liveChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Close",
          data: close,
          borderColor: "#2563EB",
          borderWidth: 2,
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
      scales: {
        x: { display: false },
        y: { grid: { color: "rgba(209,213,219,0.5)" } },
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

async function runLivePrediction() {
  const payload = {
    ticker: $("tickerInput").value.trim().toUpperCase(),
  };

  if (!payload.ticker) {
    setStatus("Enter a ticker first.", true);
    return;
  }

  try {
    setStatus(`Running live prediction for ${payload.ticker}...`);
    const data = await api("/api/live-prediction", "POST", payload);

    const result = data.result;
    const cache = data.cache || {};
    const metrics = [
      { label: "Current Price", value: `INR ${Number(result.current_price).toFixed(2)}` },
      { label: "Regime", value: result.regime },
      { label: "Model", value: result.paradigm },
    ];

    $("liveMetrics").innerHTML = metrics
      .map(
        (m) =>
          `<div class='soft-card p-3'><p class='text-xs uppercase tracking-[0.1em] text-ink/60'>${m.label}</p><p class='text-lg font-semibold mt-1'>${m.value}</p></div>`
      )
      .join("");

    const signalColor = signalColors[result.signal] || "#6B7280";
    $("signalBadge").innerHTML = `<span class='badge' style='background:${signalColor}22;color:${signalColor};border:1px solid ${signalColor}66'>${result.signal}</span>`;

    if (result.paradigm === "regression") {
      const ret = Number(result.predicted_return_pct || 0);
      const nextPrice = result.predicted_price ? Number(result.predicted_price).toFixed(2) : "-";
      $("predictionText").textContent = `Predicted return: ${ret > 0 ? "+" : ""}${ret.toFixed(3)}% | Predicted close: INR ${nextPrice}`;
    } else {
      const conf = result.confidence !== null && result.confidence !== undefined ? `${result.confidence}%` : "N/A";
      $("predictionText").textContent = `Direction: ${result.prediction || "-"} | Confidence: ${conf}`;
    }

    const routing = Object.entries(result.routing_table || {})
      .map(([regime, paradigm]) => `<div class='flex justify-between'><span>${regime}</span><span class='text-ink/70'>${paradigm}</span></div>`)
      .join("");
    $("routingTable").innerHTML = routing || "<p class='text-sm text-ink/70'>No routing table.</p>";

    const cacheText = cache.hit ? "cache hit" : "retrained";
    setStatus(`Live prediction completed for ${payload.ticker} (${cacheText}).`);

    renderLiveChart((data.chart && data.chart.points) || []);

    const statsCols = [
      { key: "regime", label: "Regime" },
      { key: "mean_daily_return", label: "Mean Daily Return %" },
      { key: "mean_vol_ann", label: "Annual Vol %" },
      { key: "pct_of_days", label: "Days %" },
    ];
    $("regimeStats").innerHTML = asTable(data.regime_stats || [], statsCols);

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
  $("runLive").addEventListener("click", runLivePrediction);
  $("tickerInput").addEventListener("input", (e) => {
    updateDatalist(e.target.value);
  });
}

async function init() {
  bindEvents();
  switchMode("live");
  await loadTickers();
  setStatus("Ready. Select a ticker and run a mode.");
}

init();
