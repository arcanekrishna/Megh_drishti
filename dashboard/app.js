/**
 * MEGH-DRISHTI: AI Weather Nowcasting & Early Warning Dashboard  v3.0
 * -------------------------------------------------------------------
 * Features:
 *  - Bilinear interpolation for dense visual heatmap from sparse API grid
 *  - Chart.js probability timeline (all 6 lead-time steps simultaneously)
 *  - 60-second auto-refresh with animated SVG countdown ring
 *  - RED alert visual pulse + optional audio beep
 *  - Full XAI attribution modal with 6-metric diagnostics
 *  - Scenario switching (simulation + live Open-Meteo)
 *  - Graceful fallback to calibrated synthetic data when API is offline
 */

// ============================================================
// STATE
// ============================================================
const state = {
  map:             null,
  currentStep:     1,
  currentLayer:    'cloudburst',
  currentScenario: 'himachal_cloudburst_2023',
  isPlaying:       false,
  playTimer:       null,
  nowcastData:     null,  // cache of latest /api/nowcast response
  metricsData:     null,  // cache of latest /api/metrics response
  activeAlerts:    [],
  trendChart:      null,
  mapLayers: {
    hazardLayerGroup:  null,
    hotspotLayerGroup: null,
  },
  refreshTimer:    null,
  refreshSecondsLeft: 60,
  apiOnline:       false,
};

const REFRESH_INTERVAL_SEC = 60;
const RING_CIRCUMFERENCE   = 2 * Math.PI * 15; // r=15 → 94.25

// ============================================================
// FALLBACK DATA (standalone demo, no backend)
// ============================================================
const FALLBACK_ZONES = [
  { name:'Mandi - Beas Valley, HP',         lat:31.70, lon:76.93, basin:'Beas Basin',        cb:0.91, ts:0.78, ff:0.87, sev:'RED' },
  { name:'Kullu - Parvati Catchment, HP',   lat:31.95, lon:77.10, basin:'Upper Beas Basin',  cb:0.85, ts:0.73, ff:0.82, sev:'RED' },
  { name:'Dharamsala - Kangra Valley, HP',  lat:32.21, lon:76.32, basin:'Gaj Khad Catchment',cb:0.79, ts:0.69, ff:0.75, sev:'RED' },
  { name:'Kedarnath - Mandakini Gorge, UK', lat:30.73, lon:79.06, basin:'Mandakini Basin',   cb:0.82, ts:0.66, ff:0.80, sev:'RED' },
  { name:'Shimla - Rampur Corridor, HP',    lat:31.10, lon:77.17, basin:'Sutlej Basin',      cb:0.64, ts:0.60, ff:0.57, sev:'ORANGE' },
  { name:'Uttarkashi - Bhagirathi, UK',     lat:30.72, lon:78.44, basin:'Bhagirathi Basin',  cb:0.55, ts:0.62, ff:0.50, sev:'ORANGE' },
  { name:'Dehradun - Doon Valley, UK',      lat:30.31, lon:78.03, basin:'Song River',        cb:0.42, ts:0.58, ff:0.34, sev:'YELLOW' },
];

const FALLBACK_CURVES = {
  lead_times:  [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
  cloudburst:  [55, 68, 80, 88, 82, 74],
  thunderstorm:[48, 62, 71, 75, 68, 60],
  flash_flood: [38, 52, 68, 84, 86, 79],
};

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupEventListeners();
  initTrendChart();
  startAutoRefresh();
  bootData();
  startStatusBarClock();
});

async function bootData() {
  await fetchSystemStatus();

  // Always boot into the default simulation — ensures map & alerts are populated
  // even if the server was previously left in live-data mode or a stale state.
  try {
    await fetch(`/api/simulate?scenario=${state.currentScenario}`,
                { method:'POST', signal:AbortSignal.timeout(8000) });
  } catch (_) { /* offline mode — fallback data will render */ }

  await Promise.all([
    fetchNowcastData(state.currentStep),
    fetchAlerts(),
    fetchMetrics(),
  ]);
}

// ============================================================
// MAP INITIALISATION
// ============================================================
function initMap() {
  state.map = L.map('map', {
    center:     [31.1, 77.8],
    zoom:       8,
    minZoom:    6,
    maxZoom:    14,
    zoomControl: false,
  });
  L.control.zoom({ position:'topright' }).addTo(state.map);

  // Dark basemap — Stadia Alidade Smooth Dark (free, no API key required)
  // Falls back to OpenTopoMap then plain OSM if unavailable
  const tileOptions = { maxZoom: 20, crossOrigin: true };

  const stadiaLayer = L.tileLayer(
    'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png',
    {
      ...tileOptions,
      attribution:'&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }
  );

  // Try Stadia; on tile-load error fall back to OSM dark (Jawg) or plain OSM
  let stediaFailed = false;
  stadiaLayer.on('tileerror', () => {
    if (!stediaFailed) {
      stediaFailed = true;
      console.warn('[Map] Stadia tiles failed — falling back to OSM');
      // Esri Dark Gray Canvas — no key needed
      L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        { ...tileOptions, attribution:'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ', maxZoom:16 }
      ).addTo(state.map);
    }
  });
  stadiaLayer.addTo(state.map);

  state.mapLayers.hazardLayerGroup  = L.layerGroup().addTo(state.map);
  state.mapLayers.hotspotLayerGroup = L.layerGroup().addTo(state.map);

  // Map click → XAI
  state.map.on('click', (e) => {
    const { lat, lng } = e.latlng;
    fetchXAIAttribution(lat, lng, state.currentStep,
      `Map Location (${lat.toFixed(2)}°N, ${lng.toFixed(2)}°E)`);
  });
}

// ============================================================
// CHART.JS INITIALISATION
// ============================================================
function initTrendChart() {
  const ctx = document.getElementById('trendChart').getContext('2d');
  state.trendChart = new Chart(ctx, {
    type:'line',
    data:{
      labels: ['0.5h','1.0h','1.5h','2.0h','2.5h','3.0h'],
      datasets:[
        {
          label:'Cloudburst (%)',
          data: FALLBACK_CURVES.cloudburst,
          borderColor:'#ef4444', backgroundColor:'rgba(239,68,68,0.08)',
          tension:0.42, fill:true, pointRadius:4, pointHoverRadius:6,
          pointBackgroundColor:'#ef4444', borderWidth:2,
        },
        {
          label:'Thunderstorm (%)',
          data: FALLBACK_CURVES.thunderstorm,
          borderColor:'#f59e0b', backgroundColor:'rgba(245,158,11,0.06)',
          tension:0.42, fill:true, pointRadius:4, pointHoverRadius:6,
          pointBackgroundColor:'#f59e0b', borderWidth:2,
        },
        {
          label:'Flash Flood (%)',
          data: FALLBACK_CURVES.flash_flood,
          borderColor:'#06b6d4', backgroundColor:'rgba(6,182,212,0.06)',
          tension:0.42, fill:true, pointRadius:4, pointHoverRadius:6,
          pointBackgroundColor:'#06b6d4', borderWidth:2,
        },
      ],
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      animation:{ duration:600, easing:'easeInOutQuart' },
      plugins:{
        legend:{
          labels:{
            color:'#8da0c4', font:{ size:10, family:'Inter' },
            boxWidth:14, padding:12,
          },
        },
        tooltip:{
          backgroundColor:'rgba(8,13,23,0.92)',
          titleColor:'#f0f4ff', bodyColor:'#8da0c4',
          borderColor:'rgba(59,130,246,0.3)', borderWidth:1,
          padding:8,
        },
      },
      scales:{
        x:{
          ticks:{ color:'#4d6182', font:{ size:9 } },
          grid:{ color:'rgba(255,255,255,0.04)' },
        },
        y:{
          min:0, max:100,
          ticks:{ color:'#4d6182', font:{ size:9 }, callback:v=>`${v}%` },
          grid:{ color:'rgba(255,255,255,0.04)' },
        },
      },
    },
  });
}

function updateTrendChart(curves) {
  if (!state.trendChart || !curves) return;
  const c = curves.lead_time_curves || FALLBACK_CURVES;
  state.trendChart.data.datasets[0].data = c.cloudburst   || [];
  state.trendChart.data.datasets[1].data = c.thunderstorm || [];
  state.trendChart.data.datasets[2].data = c.flash_flood  || [];
  state.trendChart.update();
}

// ============================================================
// AUTO REFRESH
// ============================================================
function startAutoRefresh() {
  state.refreshSecondsLeft = REFRESH_INTERVAL_SEC;
  tickCountdown();
  state.refreshTimer = setInterval(() => {
    state.refreshSecondsLeft--;
    if (state.refreshSecondsLeft <= 0) {
      state.refreshSecondsLeft = REFRESH_INTERVAL_SEC;
      fetchNowcastData(state.currentStep);
      fetchAlerts();
      fetchMetrics();
    }
    tickCountdown();
  }, 1000);
}

function tickCountdown() {
  const sec  = state.refreshSecondsLeft;
  const frac = sec / REFRESH_INTERVAL_SEC;                          // 0..1
  const offset = RING_CIRCUMFERENCE * (1 - frac);
  const ring   = document.getElementById('countdownRingFill');
  const label  = document.getElementById('countdownSec');
  if (ring)  ring.style.strokeDashoffset = offset;
  if (label) label.textContent = sec;
}

// ============================================================
// STATUS BAR CLOCK
// ============================================================
function startStatusBarClock() {
  function tick() {
    const el = document.getElementById('statusBarTime');
    if (el) {
      const now = new Date();
      el.innerHTML = `<i class="fa-solid fa-clock"></i> ${now.toLocaleTimeString('en-IN', { hour12:false })} IST`;
    }
  }
  tick();
  setInterval(tick, 1000);
}

// ============================================================
// API CALLS
// ============================================================
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status', { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const data = await res.json();
      state.apiOnline = true;
      document.getElementById('systemStatusText').textContent =
        `LIVE · ${data.grid_size} GRID · ${data.prediction_lead_time}`;
      document.getElementById('statusPill').style.cssText =
        'background:rgba(16,185,129,0.15);border-color:rgba(16,185,129,0.4);color:#10b981;';
      return;
    }
  } catch (_) {}
  state.apiOnline = false;
  document.getElementById('systemStatusText').textContent = 'OFFLINE MODE — Synthetic Demo Data';
  document.getElementById('statusPill').style.cssText =
    'background:rgba(245,158,11,0.15);border-color:rgba(245,158,11,0.4);color:#f59e0b;';
}

async function fetchNowcastData(stepIndex) {
  const icon = document.querySelector('#btnRefreshNowcast i');
  if (icon) icon.classList.add('spinning');
  try {
    const res = await fetch(`/api/nowcast?step=${stepIndex}`, { signal:AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      state.nowcastData = data;
      renderNowcastOnMap(data);
      updateTimelineLabels(data);
      if (icon) icon.classList.remove('spinning');
      return;
    }
  } catch (_) {}
  renderFallbackOnMap();
  if (icon) icon.classList.remove('spinning');
}

async function fetchAlerts() {
  try {
    const res = await fetch('/api/alerts', { signal:AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      renderAlerts(data.alerts || []);
      return;
    }
  } catch (_) {}
  renderAlerts(buildFallbackAlerts());
}

async function fetchMetrics() {
  try {
    const res = await fetch('/api/metrics', { signal:AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      state.metricsData = data;
      updateMetricGauges(data);
      updateTrendChart(data);
      return;
    }
  } catch (_) {}
  // Use fallback metrics
  updateMetricGauges(null);
  updateTrendChart(null);
}

async function fetchXAIAttribution(lat, lon, step, title) {
  openXAIModal(title, lat, lon);
  try {
    const res = await fetch(
      `/api/xai/point?lat=${lat}&lon=${lon}&step=${step}`,
      { signal:AbortSignal.timeout(8000) }
    );
    if (res.ok) {
      const data = await res.json();
      populateXAIData(data, title);
      return;
    }
  } catch (_) {}
  // Synthetic fallback XAI
  populateXAIData(buildFallbackXAI(lat, lon, step), title);
}

// ============================================================
// MAP RENDERING
// ============================================================

/**
 * Bilinear interpolation on a sparse grid to generate a visually
 * rich heatmap — maps API (n_lat × n_lon) grid to screen-density circles.
 */
function bilinearSample(grid, lats, lons, targetLat, targetLon) {
  const nLat = lats.length, nLon = lons.length;
  let iy = 0, ix = 0;
  const latsDesc = lats[0] > lats[lats.length - 1];
  for (let i = 0; i < nLat - 1; i++) { 
      if (latsDesc) { if (targetLat <= lats[i]) iy = i; }
      else { if (targetLat >= lats[i]) iy = i; }
  }
  for (let j = 0; j < nLon - 1; j++) { if (targetLon >= lons[j]) ix = j; }
  iy = Math.min(iy, nLat - 2);
  ix = Math.min(ix, nLon - 2);

  const lat0 = lats[iy], lat1 = lats[iy + 1];
  const lon0 = lons[ix], lon1 = lons[ix + 1];
  const tLat = (targetLat - lat0) / (lat1 - lat0 + 1e-9);
  const tLon = (targetLon - lon0) / (lon1 - lon0 + 1e-9);

  const q00 = (grid[iy]   || [])[ix]   || 0;
  const q10 = (grid[iy+1] || [])[ix]   || 0;
  const q01 = (grid[iy]   || [])[ix+1] || 0;
  const q11 = (grid[iy+1] || [])[ix+1] || 0;

  return (1 - tLat) * ((1 - tLon) * q00 + tLon * q01)
       +      tLat  * ((1 - tLon) * q10 + tLon * q11);
}

function getActiveGrid(probs) {
  const layer = state.currentLayer;
  if (layer === 'cloudburst')   return probs.cloudburst || probs.severe_thunderstorm;
  if (layer === 'flash_flood')  return probs.flash_flood;
  if (layer === 'thunderstorm') return probs.severe_thunderstorm;
  if (layer === 'sat_iwv')      return probs.cloudburst;   // proxy
  if (layer === 'dem_slope')    return probs.flash_flood;  // proxy
  return probs.cloudburst;
}

function getHazardColor(prob, layer) {
  // Smooth continuous gradient for high-end aesthetic
  const p = Math.max(0, Math.min(1, prob));
  
  if (layer === 'flash_flood') {
      // Dark Blue to Cyan to White-ish Cyan
      const r = Math.floor(16 + p * (0 - 16));
      const g = Math.floor(185 + p * (255 - 185));
      const b = Math.floor(129 + p * (255 - 129));
      if (p > 0.75) return '#06b6d4';
      if (p > 0.40) return '#0ea5e9';
      return '#3b82f6';
  }
  
  if (layer === 'thunderstorm') {
      // Yellow to Orange to Purple
      if (p > 0.80) return '#9333ea';
      if (p > 0.50) return '#f97316';
      if (p > 0.30) return '#eab308';
      return '#84cc16';
  }
  
  // Cloudburst: Green -> Yellow -> Orange -> Red -> Deep Red
  if (p >= 0.85) return '#991b1b';
  if (p >= 0.70) return '#dc2626';
  if (p >= 0.50) return '#f97316';
  if (p >= 0.30) return '#eab308';
  return '#22c55e';
}

function renderNowcastOnMap(data) {
  state.mapLayers.hazardLayerGroup.clearLayers();
  state.mapLayers.hotspotLayerGroup.clearLayers();

  const { lats, lons } = data.grid_meta;
  const probs           = data.probabilities;
  const activeGrid      = getActiveGrid(probs);
  if (!activeGrid) return;

  const nLat = lats.length, nLon = lons.length;

  // Dense synthetic overlay: interpolate every 0.1° step for rich heatmap
  const step = 0.12;
  const latRange = [Math.min(...lats), Math.max(...lats)];
  const lonRange = [Math.min(...lons), Math.max(...lons)];

  for (let lat = latRange[0]; lat <= latRange[1]; lat += step) {
    for (let lon = lonRange[0]; lon <= lonRange[1]; lon += step) {
      const p = bilinearSample(activeGrid, lats, lons, lat, lon);
      if (p >= 0.20) {
        const color  = getHazardColor(p, state.currentLayer);
        const radius = 7000 + p * 6000;

        const circle = L.circle([lat, lon], {
          color:       'transparent',
          fillColor:   color,
          fillOpacity: Math.min(0.72, 0.3 + p * 0.55),
          radius,
          interactive: true,
        });
        circle.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          fetchXAIAttribution(lat, lon, state.currentStep,
            `Convective Cell (${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E)`);
        });
        circle.addTo(state.mapLayers.hazardLayerGroup);
      }
    }
  }

  // Hotspot markers with pulsing ring
  if (data.hotspots) {
    data.hotspots.forEach(pt => {
      const maxP   = Math.max(pt.cloudburst_risk, pt.thunderstorm_risk, pt.flash_flood_risk);
      const mColor = pt.threat_level === 'RED' ? '#ef4444'
                   : pt.threat_level === 'ORANGE' ? '#f97316' : '#f59e0b';
      const r      = pt.threat_level === 'RED' ? 9 : pt.threat_level === 'ORANGE' ? 7 : 5;

      const marker = L.circleMarker([pt.lat, pt.lon], {
        radius: r, color: mColor, weight:2,
        fillColor: mColor, fillOpacity:0.85,
      });
      marker.bindPopup(buildHotspotPopup(pt));
      marker.addTo(state.mapLayers.hotspotLayerGroup);

      // Pulsing outer ring for RED
      if (pt.threat_level === 'RED') {
        const pulse = L.divIcon({
          className:'',
          html:`<div class="pulse-marker" style="
            width:${r*4}px;height:${r*4}px;
            background:rgba(239,68,68,0.25);
            border:2px solid rgba(239,68,68,0.6);
          "></div>`,
          iconSize:[r*4, r*4],
          iconAnchor:[r*2, r*2],
        });
        L.marker([pt.lat, pt.lon], { icon:pulse, interactive:false })
          .addTo(state.mapLayers.hotspotLayerGroup);
      }
    });
  }
}

function buildHotspotPopup(pt) {
  const mColor = pt.threat_level === 'RED' ? '#ef4444' : '#f97316';
  return `
    <div style="font-family:'Inter',sans-serif;min-width:180px;">
      <div style="font-size:11px;font-weight:700;color:${mColor};margin-bottom:5px;">
        ⚡ ${pt.threat_level} ALERT ZONE
      </div>
      <div style="font-size:10px;color:#8da0c4;margin-bottom:6px;">
        ${pt.lat.toFixed(2)}°N, ${pt.lon.toFixed(2)}°E
      </div>
      <table style="width:100%;font-size:11px;border-collapse:collapse;">
        <tr><td style="color:#8da0c4">Cloudburst</td>
            <td style="color:${mColor};font-weight:700;text-align:right">${Math.round(pt.cloudburst_risk*100)}%</td></tr>
        <tr><td style="color:#8da0c4">Thunderstorm</td>
            <td style="color:#f59e0b;font-weight:700;text-align:right">${Math.round(pt.thunderstorm_risk*100)}%</td></tr>
        <tr><td style="color:#8da0c4">Flash Flood</td>
            <td style="color:#06b6d4;font-weight:700;text-align:right">${Math.round(pt.flash_flood_risk*100)}%</td></tr>
      </table>
      <button style="margin-top:8px;width:100%;background:#1e3a5f;color:#60a5fa;
        border:1px solid #3b82f6;padding:4px 8px;border-radius:5px;font-size:10px;cursor:pointer;"
        onclick="fetchXAIAttribution(${pt.lat},${pt.lon},${state.currentStep},'Alert Cluster (${pt.lat}°N,${pt.lon}°E)')">
        🔍 View XAI Trigger
      </button>
    </div>`;
}

function renderFallbackOnMap() {
  state.mapLayers.hazardLayerGroup.clearLayers();
  state.mapLayers.hotspotLayerGroup.clearLayers();

  // Synthetic blobs centered on fallback zones
  FALLBACK_ZONES.forEach(zone => {
    const risk  = state.currentLayer === 'cloudburst' ? zone.cb
                : state.currentLayer === 'flash_flood' ? zone.ff : zone.ts;
    const color = getHazardColor(risk, state.currentLayer);

    // Gaussian blob with a few halos for visual richness
    for (let dr = 0; dr <= 3; dr++) {
      const p = Math.max(0, risk - dr * 0.1);
      if (p < 0.15) break;
      L.circle([zone.lat + dr * 0.04, zone.lon + dr * 0.04], {
        color:'transparent', fillColor: color,
        fillOpacity: 0.65 - dr * 0.15, radius: 14000 + dr * 5000,
      }).addTo(state.mapLayers.hazardLayerGroup);
    }

    const mColor = zone.sev === 'RED' ? '#ef4444' : zone.sev === 'ORANGE' ? '#f97316' : '#f59e0b';
    const r      = zone.sev === 'RED' ? 9 : 7;
    const m = L.circleMarker([zone.lat, zone.lon], {
      radius: r, color: mColor, weight:2,
      fillColor: mColor, fillOpacity:0.9,
    });
    m.bindPopup(buildHotspotPopup({
      lat: zone.lat, lon: zone.lon,
      threat_level: zone.sev,
      cloudburst_risk: zone.cb, thunderstorm_risk: zone.ts, flash_flood_risk: zone.ff,
    }));
    m.on('click', () => fetchXAIAttribution(zone.lat, zone.lon, state.currentStep, zone.name));
    m.addTo(state.mapLayers.hotspotLayerGroup);

    if (zone.sev === 'RED') {
      const pulse = L.divIcon({
        className:'',
        html:`<div class="pulse-marker" style="width:36px;height:36px;
          background:rgba(239,68,68,0.2);border:2px solid rgba(239,68,68,0.55);"></div>`,
        iconSize:[36,36], iconAnchor:[18,18],
      });
      L.marker([zone.lat, zone.lon], { icon:pulse, interactive:false })
        .addTo(state.mapLayers.hotspotLayerGroup);
    }
  });
}

// ============================================================
// ALERTS PANEL
// ============================================================
function renderAlerts(alerts) {
  state.activeAlerts = alerts;
  const container    = document.getElementById('alertFeedContainer');

  // Count RED
  const redCount = alerts.filter(a => a.severity === 'RED').length;
  document.getElementById('alertCountNumber').textContent   = redCount;
  document.getElementById('activeAlertCount').textContent   = `${alerts.length} Active`;

  // Show/hide RED pulse banner
  const banner = document.getElementById('redAlertBanner');
  const navbar  = document.getElementById('mainNavbar');
  if (redCount > 0) {
    banner.classList.add('show');
    document.getElementById('redAlertBannerText').textContent =
      `${redCount} RED alert${redCount>1?'s':''} — ${alerts.filter(a=>a.severity==='RED').map(a=>a.target_region.split(',')[0]).join(', ')}`;
    navbar.classList.add('red-alert-active');
    if (document.getElementById('soundToggle').checked) {
      playAlertBeep();
    }
  } else {
    banner.classList.remove('show');
    navbar.classList.remove('red-alert-active');
  }

  if (!alerts || alerts.length === 0) {
    container.innerHTML = `<div style="text-align:center;padding:30px;color:#4d6182;font-size:13px;">
      <i class="fa-solid fa-shield-check" style="font-size:24px;display:block;margin-bottom:8px;color:#10b981;"></i>
      No significant weather threats detected.
    </div>`;
    return;
  }

  container.innerHTML = '';
  alerts.forEach(alert => {
    const card = document.createElement('div');
    card.className = `alert-card ${alert.severity}`;
    const cMin = alert.impact_countdown_minutes || Math.round((alert.lead_time_hours || 1) * 60);
    card.innerHTML = `
      <div class="alert-top">
        <span class="severity-pill ${alert.severity}">${alert.severity} • ${alert.hazard_type.replace('_',' ')}</span>
        <span class="lead-countdown">
          <i class="fa-solid fa-stopwatch"></i>
          <span class="countdown-mins">${cMin}m</span> Lead
        </span>
      </div>
      <div class="alert-region">${alert.target_region}</div>
      <div class="alert-basin"><i class="fa-solid fa-water"></i> ${alert.drainage_basin}</div>
      <div class="hazard-badge-row">
        <span class="hazard-badge">☁️ Cloudburst: <strong>${alert.hazard_probabilities?.cloudburst || '--'}%</strong></span>
        <span class="hazard-badge">🌊 Flash Flood: <strong>${alert.hazard_probabilities?.flash_flood || '--'}%</strong></span>
        <span class="hazard-badge">⚡ T-Storm: <strong>${alert.hazard_probabilities?.severe_thunderstorm || '--'}%</strong></span>
      </div>
      <div class="alert-sop-list">
        <h5><i class="fa-solid fa-clipboard-check"></i> Standard Operating Protocol:</h5>
        <ul>
          ${(alert.actionable_sop || []).slice(0, 2).map(s => `<li>${s}</li>`).join('')}
        </ul>
      </div>
      <button class="btn-inspect-xai"
        onclick="fetchXAIAttribution(${alert.coordinates?.lat}, ${alert.coordinates?.lon},
                  ${state.currentStep}, '${alert.target_region.replace(/'/g,"\\'")}')">
        <i class="fa-solid fa-magnifying-glass-chart"></i> Explain AI Trigger Attribution
      </button>`;
    container.appendChild(card);
  });
}

// ============================================================
// METRIC GAUGES
// ============================================================
function updateMetricGauges(m) {
  const iwv   = m?.iwv_mm        ?? 62.4;
  const trend = m?.iwv_trend_pct ?? 28;
  const ctt   = m?.ctt_drop_rate ?? -34;
  const cape  = m?.cape_jkg      ?? 2850;
  const risk  = m?.risk_level    ?? 'RED';
  const prob  = m?.max_probability ?? 88.0;

  setText('metricIWV',     `${iwv.toFixed(1)} mm`);
  setText('metricIWVTrend', trend >= 0
    ? `<i class="fa-solid fa-arrow-up"></i> +${trend.toFixed(0)}% Surge`
    : `<i class="fa-solid fa-arrow-down"></i> ${trend.toFixed(0)}%`);
  setText('metricCTT',     `${ctt >= 0 ? '+' : ''}${ctt.toFixed(1)} °C/hr`);
  setText('metricCTTStatus', ctt < -20
    ? '<i class="fa-solid fa-bolt"></i> Violent Updraft'
    : '<i class="fa-solid fa-bolt"></i> Moderate Lift');
  setText('metricCAPE',    `${Math.round(cape).toLocaleString()} J/kg`);
  setText('metricCAPEStatus', cape > 2000 ? 'Extreme Buoyancy' : cape > 1000 ? 'High Instability' : 'Moderate');
  setText('metricMaxProb', `${prob.toFixed(1)}%`);

  const badge = document.getElementById('metricRiskBadge');
  if (badge) {
    badge.textContent  = risk;
    badge.className    = `risk-badge ${risk}`;
  }

  const now = new Date();
  setText('metricsLastUpdate', `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}`);
}

// ============================================================
// XAI MODAL
// ============================================================
function openXAIModal(title, lat, lon) {
  setText('xaiLocationTitle', title);
  setText('xaiSubCoords',
    `Coordinates: ${Number(lat).toFixed(3)}°N, ${Number(lon).toFixed(3)}°E | Lead: +${((state.currentStep+1)*0.5).toFixed(1)} h`);
  // Reset to loading state
  setText('xaiPeakProb', '--');
  setText('xaiRationaleText', 'Computing XAI attribution…');
  document.getElementById('xaiModal').classList.add('active');
}

function populateXAIData(data, title) {
  const risks = data.predicted_risks || {};
  const peak  = Math.max(risks.cloudburst || 0, risks.flash_flood || 0, risks.severe_thunderstorm || 0);
  const peakPct = Math.round(peak * 100);

  // Pick dominant hazard label
  let label = 'CLOUDBURST RISK';
  if (risks.flash_flood >= risks.cloudburst && risks.flash_flood >= risks.severe_thunderstorm)
    label = 'FLASH FLOOD RISK';
  else if (risks.severe_thunderstorm >= risks.cloudburst)
    label = 'THUNDERSTORM RISK';

  setText('xaiPeakProb',   `${peakPct}%`);
  setText('xaiHazardLabel', label);
  setText('xaiRationaleText', data.meteorological_rationale || '—');

  // Threat banner colour
  const banner = document.getElementById('xaiThreatBanner');
  if (banner) {
    const col = peak >= 0.75 ? 'rgba(239,68,68,0.1)'
              : peak >= 0.5  ? 'rgba(249,115,22,0.1)'
              :                'rgba(245,158,11,0.08)';
    banner.style.background = col;
  }

  // Attribution bars
  const brk = data.trigger_breakdown_percent || {};
  setBar('xaiMoisturePct',    'xaiMoistureBar',    brk.moisture_iwv    || 0);
  setBar('xaiInstabilityPct', 'xaiInstabilityBar', brk.instability_cape || 0);
  setBar('xaiLiftPct',        'xaiLiftBar',        brk.kinematics_lift  || 0);
  setBar('xaiTopoPct',        'xaiTopoBar',        brk.topography_slope || 0);

  // Raw diagnostics
  const diag = data.raw_diagnostics || {};
  setText('diagCTT',   `${(diag.ctt_drop_index   || 0).toFixed(1)} °C/hr`);
  setText('diagIWV',   `${(diag.iwv_moisture_level || 0).toFixed(1)} mm`);
  setText('diagCAPE',  `${Math.round((diag.cape_energy_level || 0) * 2500)} J/kg`);
  setText('diagSlope', `${((diag.slope_steepness || 0) * 45).toFixed(1)}°`);
  setText('diagShear', `${((diag.bulk_shear || 0) * 40).toFixed(1)} m/s`);
  setText('diagCIN',   `${Math.round((diag.cin_inhibition || 0) * 300)} J/kg`);
}

// ============================================================
// EVENT LISTENERS
// ============================================================
function setupEventListeners() {
  // Time slider
  const slider = document.getElementById('timeSlider');
  slider.addEventListener('input', e => {
    state.currentStep = parseInt(e.target.value);
    const hours = (state.currentStep + 1) * 0.5;
    document.getElementById('leadTimeDisplay').textContent = `+${hours.toFixed(1)} h`;
    fetchNowcastData(state.currentStep);
  });

  // Play / Pause timeline
  const btnPlay = document.getElementById('btnPlayTimeline');
  btnPlay.addEventListener('click', () => {
    if (state.isPlaying) {
      clearInterval(state.playTimer);
      state.isPlaying = false;
      btnPlay.innerHTML = '<i class="fa-solid fa-play"></i> Play Sequence';
    } else {
      state.isPlaying = true;
      btnPlay.innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
      state.playTimer = setInterval(() => {
        const next = (state.currentStep + 1) % 6;
        slider.value = next;
        slider.dispatchEvent(new Event('input'));
      }, 1400);
    }
  });

  // Layer selector
  document.querySelectorAll('.layer-radio').forEach(label => {
    label.addEventListener('click', () => {
      document.querySelectorAll('.layer-radio').forEach(l => l.classList.remove('active'));
      label.classList.add('active');
      state.currentLayer = label.getAttribute('data-layer');
      if (state.nowcastData) {
        renderNowcastOnMap(state.nowcastData);
      } else {
        renderFallbackOnMap();
      }
    });
  });

  // Scenario selector
  document.getElementById('scenarioSelect').addEventListener('change', async e => {
    const chosen = e.target.value;
    state.currentScenario = chosen;
    try {
      if (chosen === 'live_data') {
        await fetch('/api/live-refresh', { method:'POST', signal:AbortSignal.timeout(15000) });
      } else {
        await fetch(`/api/simulate?scenario=${chosen}`, { method:'POST', signal:AbortSignal.timeout(10000) });
      }
    } catch (_) {}
    fetchNowcastData(state.currentStep);
    fetchAlerts();
    fetchMetrics();
  });

  // Refresh button
  document.getElementById('btnRefreshNowcast').addEventListener('click', () => {
    state.refreshSecondsLeft = REFRESH_INTERVAL_SEC;
    fetchNowcastData(state.currentStep);
    fetchAlerts();
    fetchMetrics();
  });

  // Close XAI modal
  document.getElementById('btnCloseXai').addEventListener('click', () => {
    document.getElementById('xaiModal').classList.remove('active');
  });
  document.getElementById('xaiModal').addEventListener('click', e => {
    if (e.target === e.currentTarget)
      e.currentTarget.classList.remove('active');
  });

  // Chart toggle
  document.getElementById('btnToggleChart').addEventListener('click', () => {
    const body = document.getElementById('chartPanelBody');
    const icon = document.getElementById('chartToggleIcon');
    body.classList.toggle('collapsed');
    icon.className = body.classList.contains('collapsed')
      ? 'fa-solid fa-chevron-up' : 'fa-solid fa-chevron-down';
  });
}

// ============================================================
// HELPERS
// ============================================================
function setText(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function setBar(textId, barId, value) {
  const pct = typeof value === 'number' ? value.toFixed(1) : '0.0';
  setText(textId, `${pct}%`);
  const bar = document.getElementById(barId);
  if (bar) bar.style.width = `${Math.min(100, value)}%`;
}

function updateTimelineLabels(data) {
  if (data?.timestamp) {
    document.getElementById('forecastTimestamp').textContent = data.timestamp;
  }
}

function playAlertBeep() {
  const audio = document.getElementById('alertBeep');
  if (audio) {
    audio.currentTime = 0;
    audio.play().catch(() => {});
  } else {
    // Programmatic oscillator fallback
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      [880, 660].forEach((freq, i) => {
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.frequency.value = freq;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.18, ctx.currentTime + i * 0.18);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.18 + 0.15);
        osc.start(ctx.currentTime + i * 0.18);
        osc.stop(ctx.currentTime + i * 0.18 + 0.15);
      });
    } catch (_) {}
  }
}

function buildFallbackAlerts() {
  const now = new Date();
  return FALLBACK_ZONES.map((z, i) => ({
    alert_id:                `ALRT-F${String(i+1).padStart(3,'0')}`,
    severity:                z.sev,
    category_title:          z.sev === 'RED' ? 'CRITICAL EMERGENCY WARNING'
                            : z.sev === 'ORANGE' ? 'SEVERE WEATHER ALERT' : 'WEATHER WATCH ADVISORY',
    hazard_type:             'CLOUDBURST',
    target_region:           z.name,
    drainage_basin:          z.basin,
    coordinates:             { lat: z.lat, lon: z.lon },
    lead_time_hours:         1.5,
    impact_countdown_minutes: 90,
    estimated_time_to_impact: new Date(now.getTime() + 90*60000).toLocaleTimeString('en-IN', {hour12:false}) + ' IST',
    peak_probability:        Math.round(Math.max(z.cb, z.ts, z.ff) * 100),
    hazard_probabilities:    {
      cloudburst: Math.round(z.cb*100),
      severe_thunderstorm: Math.round(z.ts*100),
      flash_flood: Math.round(z.ff*100),
    },
    actionable_sop: z.sev === 'RED'
      ? ['🚨 IMMEDIATE EVACUATION: Clear riverside settlements within 45 minutes.',
         '🚁 NDRF / SDRF ALERT: Deploy rescue teams to vulnerable bridges.']
      : ['⚠️ CIVIC ALERT: Restrict movement near mountain streams.',
         '📡 MONITORING: Track IWV updates at 15-minute intervals.'],
  }));
}

function buildFallbackXAI(lat, lon, step) {
  const seed = Math.abs(lat * 10 + lon * 7) % 1;
  return {
    coordinates: { lat, lon },
    lead_time_hours: (step + 1) * 0.5,
    predicted_risks: { cloudburst: 0.83 + seed * 0.1, severe_thunderstorm: 0.71, flash_flood: 0.79 },
    trigger_breakdown_percent: {
      moisture_iwv:    42.8,
      instability_cape: 26.3,
      kinematics_lift:  18.4,
      topography_slope: 12.5,
    },
    meteorological_rationale:
      'Extreme moisture convergence detected via INSAT-3DR IWV (+36% surge) combined with ' +
      'explosive cloud-top cooling (−32 °C/hr) and steep Himalayan valley topography amplifying ' +
      'runoff into downstream catchments.',
    raw_diagnostics: {
      ctt_drop_index:    -0.32,
      iwv_moisture_level: 0.80,
      cape_energy_level:  0.92,
      cin_inhibition:     0.04,
      bulk_shear:         0.55,
      slope_steepness:    0.74,
    },
  };
}
