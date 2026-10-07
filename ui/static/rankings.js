const $ = (id) => document.getElementById(id);
let pollTimer = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, method = "GET") {
  const response = await fetch(path, { method });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}

function render(data) {
  $("progress").textContent =
    `${data.completed}/${data.total} scanned · ${data.successful} ranked · ${data.failed} failed`;

  const rows = (data.results || []).map((item, index) => {
    const returnPct = Number(item.predicted_return_pct);
    const returnClass = returnPct >= 0 ? "text-sage" : "text-rose";
    return `<tr>
      <td>${index + 1}</td>
      <td><strong>${escapeHtml(item.ticker)}</strong></td>
      <td class="${returnClass}"><strong>${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(4)}%</strong></td>
      <td>${Number(item.one_month_return_pct) >= 0 ? "+" : ""}${Number(item.one_month_return_pct).toFixed(4)}% (${escapeHtml(item.one_month_confidence)})</td>
      <td>INR ${Number(item.one_month_predicted_price).toFixed(2)} <span class="text-xs text-ink/60">(historical close)</span></td>
      <td>INR ${Number(item.current_price).toFixed(2)}</td>
      <td>${escapeHtml(item.regime)}</td>
      <td>${escapeHtml(item.model)}</td>
      <td>${escapeHtml(item.current_price_source)}${item.current_price_is_fallback ? " (fallback)" : ""}</td>
    </tr>`;
  });
  $("rankingRows").innerHTML = rows.length
    ? rows.join("")
    : "<tr><td colspan='9'>No successful screening forecasts yet.</td></tr>";

  const errors = (data.errors || []).map(
    (item) => `<div>${escapeHtml(item.ticker)}: ${escapeHtml(item.error)}</div>`
  );
  $("errors").innerHTML = errors.length ? errors.join("") : "None.";
  $("status").textContent =
    data.status === "completed"
      ? `Scan completed. Results are sorted by next-trading-day return, highest first. Forecast data as of ${data.results?.[0]?.forecast_as_of || "latest available candle"}.`
      : "Scan in progress. Results update as tickers finish.";
}

async function poll(jobId) {
  try {
    const data = await api(`/api/rankings/${encodeURIComponent(jobId)}`);
    render(data);
    if (data.status === "queued" || data.status === "running") {
      pollTimer = setTimeout(() => poll(jobId), 2000);
    } else {
      $("startScan").disabled = false;
      $("startScan").textContent = "Scan all NSE stocks";
    }
  } catch (error) {
    $("status").textContent = error.message;
    $("startScan").disabled = false;
  }
}

async function startScan() {
  if (pollTimer) clearTimeout(pollTimer);
  $("startScan").disabled = true;
  $("startScan").textContent = "Scanning...";
  $("status").textContent = "Starting all-NSE scan...";
  try {
    const data = await api("/api/rankings/start", "POST");
    await poll(data.job_id);
  } catch (error) {
    $("status").textContent = error.message;
    $("startScan").disabled = false;
    $("startScan").textContent = "Scan all NSE stocks";
  }
}

$("startScan").addEventListener("click", startScan);
