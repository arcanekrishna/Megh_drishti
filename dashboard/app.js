/**
 * MEGH-DRISHTI: AI Weather Nowcasting & Early Warning Application
 * -------------------------------------------------------------
 * Interacts with FastAPI backend for 2-6h Nowcasting, Multi-Task Risk Grids,
 * XAI Attribution, and Disaster Management Alerts.
 */

// State Object
const state = {
  map: null,
  currentStep: 1,
  currentLayer: 'cloudburst',
  currentScenario: 'himachal_cloudburst_2023',
  isPlaying: false,
  playTimer: null,
  nowcastData: null,
  activeAlerts: [],
  mapLayers: {
    hazardLayerGroup: null,
    hotspotLayerGroup: null
  }
};

// Fallback Mock Data for Standalone Testing
const FALLBACK_ZONES = [
  { name: "Mandi - Beas River Basin, HP", lat: 31.70, lon: 76.93, basin: "Beas Basin", cb: 0.88, ts: 0.76, ff: 0.84, sev: "RED" },
  { name: "Kullu - Parvati Catchment, HP", lat: 31.95, lon: 77.10, basin: "Upper Beas Basin", cb: 0.82, ts: 0.71, ff: 0.79, sev: "RED" },
  { name: "Kedarnath - Mandakini Gorge, UK", lat: 30.73, lon: 79.06, basin: "Mandakini Basin", cb: 0.85, ts: 0.68, ff: 0.82, sev: "RED" },
  { name: "Shimla - Rampur Corridor, HP", lat: 31.10, lon: 77.17, basin: "Sutlej Basin", cb: 0.62, ts: 0.58, ff: 0.54, sev: "ORANGE" },
  { name: "Uttarkashi - Bhagirathi Valley, UK", lat: 30.72, lon: 78.44, basin: "Bhagirathi Basin", cb: 0.58, ts: 0.65, ff: 0.50, sev: "ORANGE" },
  { name: "Dehradun - Doon Valley, UK", lat: 30.31, lon: 78.03, basin: "Song River Catchment", cb: 0.44, ts: 0.60, ff: 0.35, sev: "YELLOW" }
];

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupEventListeners();
  fetchSystemStatus();
  fetchNowcastData(state.currentStep);
  fetchAlerts();
});

// 1. Initialize Map
function initMap() {
  // Center on Himachal & Uttarakhand Convective Belt
  state.map = L.map('map', {
    center: [31.2, 77.5],
    zoom: 8,
    minZoom: 6,
    maxZoom: 13,
    zoomControl: false
  });

  L.control.zoom({ position: 'topright' }).addTo(state.map);

  // High-contrast Dark Matter basemap
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(state.map);

  state.mapLayers.hazardLayerGroup = L.layerGroup().addTo(state.map);
  state.mapLayers.hotspotLayerGroup = L.layerGroup().addTo(state.map);

  // Map Click handler for XAI
  state.map.on('click', (e) => {
    fetchXAIAttribution(e.latlng.lat, e.latlng.lng, state.currentStep, `Map Location (${e.latlng.lat.toFixed(2)}°N, ${e.latlng.lng.toFixed(2)}°E)`);
  });
}

// 2. Fetch System Status
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      document.getElementById('systemStatusText').textContent = `LIVE: ${data.engine.split(' ')[0]} | ${data.prediction_lead_time}`;
    }
  } catch (err) {
    console.warn("Using offline standalone mode:", err);
  }
}

// 3. Fetch Nowcast Data
async function fetchNowcastData(stepIndex) {
  try {
    const res = await fetch(`/api/nowcast?step=${stepIndex}`);
    if (res.ok) {
      const data = await res.json();
      state.nowcastData = data;
      renderNowcastOnMap(data);
      updateTimelineLabels(data);
      return;
    }
  } catch (err) {
    console.warn("Backend API unreachable, rendering calibrated fallback layer.");
  }
  // Fallback rendering
  renderFallbackOnMap();
}

// 4. Render Hazard Layers on Map
function renderNowcastOnMap(data) {
  state.mapLayers.hazardLayerGroup.clearLayers();
  state.mapLayers.hotspotLayerGroup.clearLayers();

  const grid = data.grid_meta;
  const probs = data.probabilities[state.currentLayer] || data.probabilities['cloudburst'];
  
  const lats = grid.lats;
  const lons = grid.lons;
  const nLat = lats.length;
  const nLon = lons.length;

  // Render Risk Grid Cells
  for (let i = 0; i < nLat; i += 2) {
    for (let j = 0; j < nLon; j += 2) {
      const p = probs[i] ? probs[i][j] : 0;
      if (p >= 0.25) {
        const color = getHazardColor(p, state.currentLayer);
        const radius = p >= 0.70 ? 9000 : 6000;
        
        const circle = L.circle([lats[i], lons[j]], {
          color: 'transparent',
          fillColor: color,
          fillOpacity: Math.min(0.75, p * 0.9),
          radius: radius
        });

        circle.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          fetchXAIAttribution(lats[i], lons[j], state.currentStep, `Convective Hotspot (${lats[i]}°N, ${lons[j]}°E)`);
        });

        circle.addTo(state.mapLayers.hazardLayerGroup);
      }
    }
  }

  // Render Hotspots with Pulsing Badges
  if (data.hotspots) {
    data.hotspots.forEach(pt => {
      const markerColor = pt.threat_level === 'RED' ? '#ef4444' : '#f59e0b';
      const marker = L.circleMarker([pt.lat, pt.lon], {
        radius: pt.threat_level === 'RED' ? 10 : 7,
        color: markerColor,
        weight: 2,
        fillColor: markerColor,
        fillOpacity: 0.8
      });

      marker.bindPopup(`
        <div style="font-family:sans-serif; min-width:160px; color:#111;">
          <strong style="color:${markerColor}">⚡ ${pt.threat_level} ALERT ZONE</strong><br>
          <small>Lat: ${pt.lat}°N, Lon: ${pt.lon}°E</small><hr style="margin:4px 0">
          <div>Cloudburst Risk: <strong>${Math.round(pt.cloudburst_risk * 100)}%</strong></div>
          <div>Thunderstorm: <strong>${Math.round(pt.thunderstorm_risk * 100)}%</strong></div>
          <div>Flash Flood: <strong>${Math.round(pt.flash_flood_risk * 100)}%</strong></div>
          <button style="margin-top:6px; background:#2563eb; color:#fff; border:none; padding:4px 8px; border-radius:4px; font-size:11px; cursor:pointer;"
            onclick="fetchXAIAttribution(${pt.lat}, ${pt.lon}, ${state.currentStep}, 'Alert Cluster (${pt.lat}°N, ${pt.lon}°E)')">
            View XAI Trigger
          </button>
        </div>
      `);
      marker.addTo(state.mapLayers.hotspotLayerGroup);
    });
  }
}

// Fallback renderer
function renderFallbackOnMap() {
  state.mapLayers.hazardLayerGroup.clearLayers();
  state.mapLayers.hotspotLayerGroup.clearLayers();

  FALLBACK_ZONES.forEach(zone => {
    const risk = state.currentLayer === 'cloudburst' ? zone.cb : (state.currentLayer === 'flash_flood' ? zone.ff : zone.ts);
    const color = getHazardColor(risk, state.currentLayer);
    
    const circle = L.circle([zone.lat, zone.lon], {
      color: 'transparent',
      fillColor: color,
      fillOpacity: 0.7,
      radius: 12000
    }).addTo(state.mapLayers.hazardLayerGroup);

    const marker = L.circleMarker([zone.lat, zone.lon], {
      radius: zone.sev === 'RED' ? 10 : 7,
      color: zone.sev === 'RED' ? '#ef4444' : '#f59e0b',
      weight: 2,
      fillColor: zone.sev === 'RED' ? '#ef4444' : '#f59e0b',
      fillOpacity: 0.85
    }).addTo(state.mapLayers.hotspotLayerGroup);

    circle.on('click', () => fetchXAIAttribution(zone.lat, zone.lon, state.currentStep, zone.name));
    marker.on('click', () => fetchXAIAttribution(zone.lat, zone.lon, state.currentStep, zone.name));
  });
}

function getHazardColor(prob, layerType) {
  if (prob >= 0.75) return '#ef4444'; // Red
  if (prob >= 0.50) return '#f97316'; // Orange
  if (prob >= 0.30) return '#f59e0b'; // Amber
  return '#10b981'; // Green
}

// 5. Fetch Active Alerts & Populate Feed
async function fetchAlerts() {
  const container = document.getElementById('alertFeedContainer');
  let alerts = [];

  try {
    const res = await fetch('/api/alerts');
    if (res.ok) {
      const data = await res.json();
      alerts = data.alerts;
    }
  } catch (err) {
    console.warn("Using fallback alerts list");
  }

  if (!alerts || alerts.length === 0) {
    alerts = FALLBACK_ZONES.map((z, idx) => ({
      alert_id: `ALRT-00${idx+1}`,
      severity: z.sev,
      category_title: z.sev === "RED" ? "CRITICAL EMERGENCY WARNING" : "SEVERE WEATHER ALERT",
      hazard_type: "CLOUDBURST",
      target_region: z.name,
      drainage_basin: z.basin,
      coordinates: { lat: z.lat, lon: z.lon },
      lead_time_hours: 1.5,
      estimated_time_to_impact: "Within 90 Minutes",
      peak_probability: Math.round(z.cb * 100),
      hazard_probabilities: { cloudburst: Math.round(z.cb*100), severe_thunderstorm: Math.round(z.ts*100), flash_flood: Math.round(z.ff*100) },
      actionable_sop: [
        "IMMEDIATE EVACUATION: Clear riverside settlements and floodplains.",
        "NDRF / SDRF ALERT: Pre-position rescue battalions at river bridges.",
        "TRAFFIC RESTRICTION: Halt vehicular movement on mountain highways."
      ]
    }));
  }

  state.activeAlerts = alerts;
  document.getElementById('activeAlertCount').textContent = `${alerts.length} Active`;
  
  const redCount = alerts.filter(a => a.severity === 'RED').length;
  document.getElementById('alertCountNumber').textContent = redCount;

  // Render cards
  container.innerHTML = '';
  alerts.forEach(alert => {
    const card = document.createElement('div');
    card.className = `alert-card ${alert.severity}`;
    card.innerHTML = `
      <div class="alert-top">
        <span class="severity-pill ${alert.severity}">${alert.severity} • ${alert.hazard_type}</span>
        <span class="lead-countdown"><i class="fa-solid fa-stopwatch"></i> ${alert.lead_time_hours}h Lead Time</span>
      </div>
      <div class="alert-region">${alert.target_region}</div>
      <div class="alert-basin"><i class="fa-solid fa-water"></i> ${alert.drainage_basin}</div>
      <div class="hazard-badge-row">
        <span class="hazard-badge">Cloudburst: <strong>${alert.hazard_probabilities.cloudburst}%</strong></span>
        <span class="hazard-badge">Flash Flood: <strong>${alert.hazard_probabilities.flash_flood}%</strong></span>
      </div>
      <div class="alert-sop-list">
        <h5><i class="fa-solid fa-clipboard-check"></i> Standard Operating Protocol:</h5>
        <ul>
          ${alert.actionable_sop.slice(0, 2).map(sop => `<li>${sop}</li>`).join('')}
        </ul>
      </div>
      <button class="btn-inspect-xai" onclick="fetchXAIAttribution(${alert.coordinates.lat}, ${alert.coordinates.lon}, ${state.currentStep}, '${alert.target_region}')">
        <i class="fa-solid fa-magnifying-glass-chart"></i> Explain AI Trigger Attribution
      </button>
    `;
    container.appendChild(card);
  });
}

// 6. Fetch Explainable AI (XAI) Attribution
async function fetchXAIAttribution(lat, lon, step, title) {
  openXAIModal(title, lat, lon);

  try {
    const res = await fetch(`/api/xai/point?lat=${lat}&lon=${lon}&step=${step}`);
    if (res.ok) {
      const data = await res.json();
      populateXAIData(data, title);
      return;
    }
  } catch (err) {
    console.warn("Using synthesized XAI attribution fallback");
  }

  // Fallback XAI calculation
  const mockXAI = {
    coordinates: { lat, lon },
    lead_time_hours: (step + 1) * 0.5,
    predicted_risks: { cloudburst: 0.85, severe_thunderstorm: 0.72, flash_flood: 0.81 },
    trigger_breakdown_percent: {
      moisture_iwv: 44.2,
      instability_cape: 24.8,
      kinematics_lift: 18.5,
      topography_slope: 12.5
    },
    meteorological_rationale: "Extreme moisture pooling detected via INSAT-3DR IWV (+36% surge) combined with explosive CTT cooling (-32°C/hr) and steep Himalayan valley topography.",
    raw_diagnostics: {
      ctt_drop_index: -32.4,
      iwv_moisture_level: 65.2,
      cape_energy_level: 2890,
      slope_steepness: 39.5
    }
  };
  populateXAIData(mockXAI, title);
}

function openXAIModal(title, lat, lon) {
  document.getElementById('xaiLocationTitle').textContent = title;
  document.getElementById('xaiSubCoords').textContent = `Coordinates: ${Number(lat).toFixed(3)}°N, ${Number(lon).toFixed(3)}°E | Forecast Lead Time: +${(state.currentStep+1)*0.5}h`;
  document.getElementById('xaiModal').classList.add('active');
}

function populateXAIData(data, title) {
  const peakProb = Math.round(Math.max(data.predicted_risks.cloudburst, data.predicted_risks.flash_flood, data.predicted_risks.severe_thunderstorm) * 100);
  document.getElementById('xaiPeakProb').textContent = `${peakProb}%`;
  document.getElementById('xaiRationaleText').textContent = data.meteorological_rationale;

  const brk = data.trigger_breakdown_percent;
  document.getElementById('xaiMoisturePct').textContent = `${brk.moisture_iwv}%`;
  document.getElementById('xaiMoistureBar').style.width = `${brk.moisture_iwv}%`;

  document.getElementById('xaiInstabilityPct').textContent = `${brk.instability_cape}%`;
  document.getElementById('xaiInstabilityBar').style.width = `${brk.instability_cape}%`;

  document.getElementById('xaiLiftPct').textContent = `${brk.kinematics_lift}%`;
  document.getElementById('xaiLiftBar').style.width = `${brk.kinematics_lift}%`;

  document.getElementById('xaiTopoPct').textContent = `${brk.topography_slope}%`;
  document.getElementById('xaiTopoBar').style.width = `${brk.topography_slope}%`;

  if (data.raw_diagnostics) {
    document.getElementById('diagCTT').textContent = `${data.raw_diagnostics.ctt_drop_index} °C/hr`;
    document.getElementById('diagIWV').textContent = `${data.raw_diagnostics.iwv_moisture_level} mm`;
    document.getElementById('diagCAPE').textContent = `${data.raw_diagnostics.cape_energy_level} J/kg`;
    document.getElementById('diagSlope').textContent = `${data.raw_diagnostics.slope_steepness}°`;
  }
}

// 7. Event Listeners & UI Controls
function setupEventListeners() {
  // Time Slider
  const slider = document.getElementById('timeSlider');
  slider.addEventListener('input', (e) => {
    state.currentStep = parseInt(e.target.value);
    const hours = (state.currentStep + 1) * 0.5;
    document.getElementById('leadTimeDisplay').textContent = `+${hours.toFixed(1)} Hour`;
    fetchNowcastData(state.currentStep);
  });

  // Play Timeline Button
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
        let nextStep = (state.currentStep + 1) % 6;
        slider.value = nextStep;
        slider.dispatchEvent(new Event('input'));
      }, 1500);
    }
  });

  // Layer Radio Selection
  document.querySelectorAll('.layer-radio').forEach(radioLabel => {
    radioLabel.addEventListener('click', () => {
      document.querySelectorAll('.layer-radio').forEach(l => l.classList.remove('active'));
      radioLabel.classList.add('active');
      state.currentLayer = radioLabel.getAttribute('data-layer');
      if (state.nowcastData) {
        renderNowcastOnMap(state.nowcastData);
      } else {
        renderFallbackOnMap();
      }
    });
  });

  // Scenario Selector
  const scenarioSelect = document.getElementById('scenarioSelect');
  scenarioSelect.addEventListener('change', async (e) => {
    const chosen = e.target.value;
    state.currentScenario = chosen;
    try {
      await fetch(`/api/simulate?scenario=${chosen}`, { method: 'POST' });
    } catch (err) {
      console.warn("Scenario switch offline simulation");
    }
    fetchNowcastData(state.currentStep);
    fetchAlerts();
  });

  // Refresh Button
  document.getElementById('btnRefreshNowcast').addEventListener('click', () => {
    fetchNowcastData(state.currentStep);
    fetchAlerts();
  });

  // Close XAI Modal
  document.getElementById('btnCloseXai').addEventListener('click', () => {
    document.getElementById('xaiModal').classList.remove('active');
  });
}

function updateTimelineLabels(data) {
  if (data && data.timestamp) {
    document.getElementById('forecastTimestamp').textContent = data.timestamp;
  }
}
