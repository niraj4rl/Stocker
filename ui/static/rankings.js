const $ = (id) => document.getElementById(id);
let pollTimer = null;
let latestResults = [];

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
function formatINR(value) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", minimumFractionDigits: 2 }).format(Number(value));
}
function signedPercent(value) {
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)}%`;
}
async function api(path, method = "GET") {
  const response = await fetch(path, { method });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}
function renderRows() {
  const query = ($("rankingFilter").value || "").trim().toUpperCase();
  const positiveOnly = $("positiveOnly").checked;
  const filtered = latestResults.filter((item) => {
    const ticker = String(item.ticker || "").toUpperCase();
    return (!query || ticker.includes(query)) && (!positiveOnly || Number(item.one_month_return_pct) > 0);
  });
  const maxAbs = Math.max(...latestResults.map((item) => Math.abs(Number(item.one_month_return_pct) || 0)), 1);
  $("rankingRows").innerHTML = filtered.length ? filtered.map((item, index) => {
    const month = Number(item.one_month_return_pct);
    const day = Number(item.one_day_return_pct ?? item.predicted_return_pct);
    const color = month >= 0 ? "gain" : "loss";
    const width = Math.min(Math.abs(month) / maxAbs * 100, 100);
    return `<tr data-href="/app?ticker=${encodeURIComponent(item.ticker)}" tabindex="0">
      <td class="mono">${item.rank || index + 1}</td>
      <td><a class="mono" href="/app?ticker=${encodeURIComponent(item.ticker)}"><strong>${escapeHtml(item.ticker.replace(".NS", ""))}</strong></a></td>
      <td class="${color}"><span class="return-cell"><b>${signedPercent(month)}</b><span class="return-bar"><i style="width:${width}%"></i></span></span></td>
      <td class="${day >= 0 ? "gain" : "loss"}">${signedPercent(day)}</td>
      <td class="mono" title="Based on the latest available historical close">${formatINR(item.one_month_predicted_price)}</td>
      <td class="mono">${formatINR(item.current_price)}</td>
    </tr>`;
  }).join("") : "<tr><td colspan='6' class='empty'>No stocks match this view.</td></tr>";
}
function render(data) {
  latestResults = data.results || [];
  const total = Number(data.total) || 0;
  const completed = Number(data.completed) || 0;
  $("progress").textContent = `${completed.toLocaleString("en-IN")} / ${total.toLocaleString("en-IN")} scanned`;
  $("scanProgressBar").style.width = total ? `${completed / total * 100}%` : "0%";
  renderRows();
  const errors = (data.errors || []).map((item) => `<div>${escapeHtml(item.ticker)}: ${escapeHtml(item.error)}</div>`);
  $("errors").innerHTML = errors.length ? errors.join("") : "None.";
  $("status").textContent = data.status === "completed"
    ? "Updated just now. Select a row to explore that stock."
    : "Checking stocks and adding them as results arrive.";
}
async function poll(jobId) {
  try {
    const data = await api(`/api/rankings/${encodeURIComponent(jobId)}`);
    render(data);
    if (data.status === "queued" || data.status === "running") pollTimer = setTimeout(() => poll(jobId), 2000);
    else { $("startScan").disabled = false; $("startScan").textContent = "Scan stocks"; }
  } catch (error) {
    $("status").textContent = error.message;
    $("startScan").disabled = false;
  }
}
async function startScan() {
  if (pollTimer) clearTimeout(pollTimer);
  $("startScan").disabled = true;
  $("startScan").textContent = "Checking…";
  try { const data = await api("/api/rankings/start", "POST"); await poll(data.job_id); }
  catch (error) { $("status").textContent = error.message; $("startScan").disabled = false; }
}
$("startScan").addEventListener("click", startScan);
$("rankingFilter").addEventListener("input", renderRows);
$("positiveOnly").addEventListener("change", renderRows);
$("rankingRows").addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-href]");
  if (row && !event.target.closest("a")) window.location.href = row.dataset.href;
});
$("rankingRows").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest("tr[data-href]");
  if (row) { event.preventDefault(); window.location.href = row.dataset.href; }
});
