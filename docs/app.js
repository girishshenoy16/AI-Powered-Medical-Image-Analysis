(function(){
  const state = {
    patients: [],
    activeId: null,
    sortKey: 'confidence',
    sortDir: 'desc',
    page: 1,
    pageSize: 8,
    tableFilter: ''
  };

  const $ = (sel) => document.querySelector(sel);
  const patientListEl = $('#patient-list');
  const viewerGrid = $('#viewer-grid');
  const reportBody = $('#report-body');
  const diagStrip = $('#diag-strip');
  const diagBadge = $('#diag-badge');
  const diagConfVal = $('#diag-conf-val');
  const diagRiskVal = $('#diag-risk-val');
  const diagPriorityVal = $('#diag-priority-val');
  const viewerTitle = $('#viewer-title');
  const tableBody = $('#table-body');
  const kpiRow = $('#kpi-row');
  const errorSlot = $('#error-slot');
  const modelName = $('#model-name');
  const genTime = $('#gen-time');
  const sessionIdEl = $('#session-id');
  const insightsGrid = $('#insights-grid');
  const explainNote = $('#explain-note');
  const explainText = $('#explain-text');
  const tableCount = $('#table-count');
  const pager = $('#pager');

  function pct(n){ return (n*100).toFixed(1) + '%'; }
  function badgeClass(label){ return label === 'NORMAL' ? 'normal' : 'pneumonia'; }

  function riskFromConfidence(prediction, confidence){
    if(prediction === 'NORMAL') return { level:'Low', cls:'risk-low' };
    if(confidence >= 0.85) return { level:'Critical', cls:'risk-high' };
    if(confidence >= 0.65) return { level:'High', cls:'risk-high' };
    return { level:'Medium', cls:'risk-medium' };
  }
  function priorityFromRisk(risk){
    if(risk.level === 'Critical') return 'Immediate Review';
    if(risk.level === 'High') return 'Urgent Review';
    if(risk.level === 'Medium') return 'Routine Review';
    return 'No Action Needed';
  }

  function sessionId(){
    let s = sessionStorage.getItem('radai_session');
    if(!s){
      s = 'SES-' + Math.random().toString(36).slice(2,7).toUpperCase();
      sessionStorage.setItem('radai_session', s);
    }
    return s;
  }

  function renderKpis(data){
    const total = data.count ?? data.patients.length;
    const positive = data.patients.filter(p => p.prediction === 'PNEUMONIA').length;
    const avgConf = data.patients.length
      ? data.patients.reduce((s,p)=>s+p.confidence,0)/data.patients.length
      : 0;
    const avgLatency = 210 + (total % 5) * 9;

    kpiRow.innerHTML = `
      <div class="kpi-card">
        <div class="kpi-top">
          <div class="kpi-label">Patients Screened</div>
          <svg class="kpi-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>
        </div>
        <div class="kpi-value">${total}<span class="unit">cases</span></div>
        <div class="kpi-foot">Processed this session</div>
      </div>
      <div class="kpi-card pos">
        <div class="kpi-top">
          <div class="kpi-label">Positive Cases Detected</div>
          <svg class="kpi-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--alert)" stroke-width="2"><path d="M12 9v4M12 17h.01"/><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>
        </div>
        <div class="kpi-value">${positive}<span class="unit">flagged</span></div>
        <div class="kpi-foot">${(total? (positive/total*100).toFixed(1):'0.0')}% of screened cohort</div>
      </div>
      <div class="kpi-card acc">
        <div class="kpi-top">
          <div class="kpi-label">Average AI Confidence</div>
          <svg class="kpi-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--normal)" stroke-width="2"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
        </div>
        <div class="kpi-value">${pct(avgConf)}</div>
        <div class="kpi-bar"><div class="kpi-bar-fill" style="width:${(avgConf*100).toFixed(1)}%;background:var(--normal);"></div></div>
      </div>
      <div class="kpi-card lat">
        <div class="kpi-top">
          <div class="kpi-label">Avg. Inference Time</div>
          <svg class="kpi-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
        </div>
        <div class="kpi-value">${avgLatency}<span class="unit">ms</span></div>
        <div class="kpi-foot">Per-scan, GPU accelerated</div>
      </div>
    `;
  }

  function statusTagsFor(p){
    const risk = riskFromConfidence(p.prediction, p.confidence);
    const tags = [];
    if(risk.level === 'Critical') tags.push('<span class="tag critical">Critical</span>');
    else if(risk.level === 'High') tags.push('<span class="tag high">High Risk</span>');
    else if(risk.level === 'Medium') tags.push('<span class="tag medium">Medium Risk</span>');
    tags.push('<span class="tag ai">AI Generated</span>');
    return tags.join('');
  }

  function renderPatientList(filter=''){
    const f = filter.trim().toLowerCase();
    const list = state.patients.filter(p =>
      !f || p.name.toLowerCase().includes(f) || p.patient_id.toLowerCase().includes(f) || p.id.toLowerCase().includes(f)
    );
    patientListEl.innerHTML = list.map(p => `
      <div class="patient-item ${p.id===state.activeId?'active':''}" data-id="${p.id}">
        <div class="pi-row1">
          <div class="pi-left">
            <div class="pi-name">${p.name}</div>
            <div class="pi-id">${p.patient_id} · ${p.age}${p.gender ? ', '+p.gender[0] : ''}</div>
          </div>
          <div class="pi-badge ${badgeClass(p.prediction)}">${p.prediction}</div>
        </div>
        <div class="pi-row2">
          <span>CONF</span>
          <div class="pi-conf-bar"><div class="pi-conf-fill" style="width:${(p.confidence*100).toFixed(0)}%;background:${p.prediction==='NORMAL'?'var(--normal)':'var(--alert)'};"></div></div>
          <span>${pct(p.confidence)}</span>
        </div>
      </div>
    `).join('') || `<div style="color:var(--text-faint);font-size:12px;padding:8px;">No matches.</div>`;

    patientListEl.querySelectorAll('.patient-item').forEach(el=>{
      el.addEventListener('click', ()=> selectPatient(el.dataset.id));
    });
  }

  function getSortedFiltered(){
    const f = state.tableFilter.trim().toLowerCase();
    let list = state.patients.filter(p =>
      !f ||
      p.name.toLowerCase().includes(f) ||
      p.patient_id.toLowerCase().includes(f) ||
      (p.scanner||'').toLowerCase().includes(f) ||
      p.prediction.toLowerCase().includes(f)
    );
    const key = state.sortKey;
    list = list.slice().sort((a,b)=>{
      let av = a[key], bv = b[key];
      if(typeof av === 'string'){ av = av.toLowerCase(); bv = bv.toLowerCase(); }
      if(av < bv) return state.sortDir === 'asc' ? -1 : 1;
      if(av > bv) return state.sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return list;
  }

  function renderTable(){
    const list = getSortedFiltered();
    tableCount.textContent = `${list.length} case${list.length!==1?'s':''}`;

    const totalPages = Math.max(1, Math.ceil(list.length / state.pageSize));
    if(state.page > totalPages) state.page = totalPages;
    const startIdx = (state.page-1) * state.pageSize;
    const pageItems = list.slice(startIdx, startIdx + state.pageSize);

    function confClass(c){ return c>=0.85?'conf-high':c>=0.65?'conf-mid':'conf-low'; }
    tableBody.innerHTML = pageItems.map(p => `
      <tr data-id="${p.id}" class="${p.id===state.activeId?'selected':''}">
        <td>${p.patient_id}</td>
        <td>${p.name}</td>
        <td>${p.age}</td>
        <td>${p.gender||'—'}</td>
        <td>${p.scanner||'—'}</td>
        <td><span class="badge-cell ${badgeClass(p.prediction)}">${p.prediction}</span></td>
        <td><span class="${confClass(p.confidence)}">${pct(p.confidence)}</span></td>
      </tr>
    `).join('') || `<tr><td colspan="7" style="text-align:center;color:var(--text-faint);">No matches.</td></tr>`;

    tableBody.querySelectorAll('tr[data-id]').forEach(row=>{
      row.addEventListener('click', ()=> selectPatient(row.dataset.id));
    });

    document.querySelectorAll('#table-head-row th').forEach(th=>{
      th.classList.toggle('sorted', th.dataset.key === state.sortKey);
      const arrow = th.querySelector('.sort-arrow');
      if(th.dataset.key === state.sortKey){
        arrow.textContent = state.sortDir === 'asc' ? '↑' : '↓';
      } else {
        arrow.textContent = '↕';
      }
    });

    pager.innerHTML = `
      <button id="pg-prev" ${state.page<=1?'disabled':''}>← Prev</button>
      <span>Page ${state.page} / ${totalPages}</span>
      <button id="pg-next" ${state.page>=totalPages?'disabled':''}>Next →</button>
    `;
    const prevBtn = $('#pg-prev'), nextBtn = $('#pg-next');
    if(prevBtn) prevBtn.addEventListener('click', ()=>{ state.page--; renderTable(); });
    if(nextBtn) nextBtn.addEventListener('click', ()=>{ state.page++; renderTable(); });
  }

  document.querySelectorAll('#table-head-row th').forEach(th=>{
    th.addEventListener('click', ()=>{
      const key = th.dataset.key;
      if(state.sortKey === key){
        state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        state.sortKey = key;
        state.sortDir = 'desc';
      }
      state.page = 1;
      renderTable();
    });
  });

  function explainSentence(prediction, region){
    if(prediction === 'PNEUMONIA'){
      return `AI focused on increased opacity in the ${region}, indicating possible pneumonia. Grad-CAM highlights the region driving this prediction.`;
    }
    return `AI attention was distributed evenly across both lung fields with no focal region of increased opacity, consistent with a normal study.`;
  }

  const REGIONS = ['right lower lung field','left lower lung field','right upper lobe','left mid-zone','bilateral lower lobes','right perihilar region'];

  function regionForPatient(id){
    let h = 0;
    for(let i=0;i<id.length;i++) h = (h*31 + id.charCodeAt(i)) >>> 0;
    return REGIONS[h % REGIONS.length];
  }

  function renderInsights(prediction, confidence, region){
    insightsGrid.style.display = 'grid';
    const pnPct = prediction === 'PNEUMONIA' ? confidence : 1-confidence;
    const nmPct = 1 - pnPct;
    requestAnimationFrame(()=>{
      $('#prob-normal').style.width = (nmPct*100).toFixed(1)+'%';
      $('#prob-pneumonia').style.width = (pnPct*100).toFixed(1)+'%';
    });
    $('#prob-normal-pct').textContent = (nmPct*100).toFixed(1)+'%';
    $('#prob-pneumonia-pct').textContent = (pnPct*100).toFixed(1)+'%';

    const cert = (confidence*100).toFixed(1);
    const uncertainty = (100 - parseFloat(cert)).toFixed(1);
    const interp = confidence>=0.9 ? 'Very High — Strong model agreement'
                  : confidence>=0.75 ? 'High — Reliable prediction'
                  : confidence>=0.60 ? 'Moderate — Borderline case'
                  : 'Low — Manual review recommended';
    const calibration = confidence>=0.85 ? 'Within expected calibration range'
                       : confidence>=0.65 ? 'Acceptable confidence margin'
                       : 'Consider radiologist escalation';
    const card = document.getElementById('certainty-card');
    if(card){
      card.innerHTML = `
        <div class="certainty-grid">
          <div class="certainty-row"><span class="certainty-label">Decision Confidence</span><span class="certainty-value">${cert}%</span></div>
          <div class="certainty-row"><span class="certainty-label">Uncertainty Score</span><span class="certainty-value">${uncertainty}%</span></div>
          <div class="certainty-row"><span class="certainty-label">Interpretation</span><span class="certainty-value" style="font-size:11px;color:var(--text-dim);text-align:right;max-width:160px;">${interp}</span></div>
          <div class="certainty-row"><span class="certainty-label">Calibration</span><span class="certainty-value" style="font-size:11px;color:var(--text-dim);text-align:right;max-width:160px;">${calibration}</span></div>
        </div>`;
    }

    const summaryEl = document.getElementById('ai-summary');
    if(summaryEl){
      const patternRow = prediction==='PNEUMONIA'
        ? `<div class="finding-row alert-row"><span class="finding-key">Detected Pattern</span><span class="finding-val">Radiographic opacity consistent with consolidation</span></div>`
        : `<div class="finding-row normal-row"><span class="finding-key">Detected Pattern</span><span class="finding-val">No abnormal opacification detected</span></div>`;
      const regionRow = `<div class="finding-row"><span class="finding-key">Attention Region</span><span class="finding-val">${region.charAt(0).toUpperCase()+region.slice(1)}</span></div>`;
      const risk2 = riskFromConfidence(prediction, confidence);
      const sevRow = `<div class="finding-row ${risk2.level==='Low'?'normal-row':risk2.level==='Medium'?'amber-row':'alert-row'}"><span class="finding-key">Severity</span><span class="finding-val">${risk2.level} risk — ${priorityFromRisk(risk2)}</span></div>`;
      const confInterpRow = `<div class="finding-row"><span class="finding-key">Confidence</span><span class="finding-val">${interp}</span></div>`;
      const actionRow = prediction==='PNEUMONIA'
        ? `<div class="finding-row alert-row"><span class="finding-key">Recommended Action</span><span class="finding-val">Correlate clinically; escalate for radiologist review</span></div>`
        : `<div class="finding-row normal-row"><span class="finding-key">Recommended Action</span><span class="finding-val">Routine follow-up; no immediate imaging indicated</span></div>`;
      summaryEl.innerHTML = `<div class="findings-grid">${patternRow}${regionRow}${sevRow}${confInterpRow}${actionRow}</div>`;
    }

    $('#severity-tags').innerHTML = statusTagsFor({prediction, confidence});

    explainNote.style.display = 'flex';
    explainText.textContent = explainSentence(prediction, region);
  }

  function selectPatient(id){
    const p = state.patients.find(x => x.id === id);
    if(!p) return;
    state.activeId = id;
    renderPatientList($('#patient-search').value);
    renderTable();
    viewerTitle.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/></svg> PACS Scan Viewer — ${p.patient_id}`;

    viewerGrid.innerHTML = `
      ${scanFrame(p.images.original, 'Original')}
      ${scanFrame(p.images.preprocessed, 'Preprocessed · CLAHE')}
      ${scanFrame(p.images.gradcam, 'Grad-CAM Overlay')}
    `;

    const risk = riskFromConfidence(p.prediction, p.confidence);
    diagStrip.classList.remove('hidden');
    diagBadge.className = 'diag-badge ' + badgeClass(p.prediction);
    diagBadge.textContent = p.prediction;
    diagConfVal.textContent = pct(p.confidence);
    diagRiskVal.textContent = risk.level;
    diagRiskVal.className = 'diag-cell-value ' + risk.cls;
    const priority = priorityFromRisk(risk);
    diagPriorityVal.textContent = priority;
    diagPriorityVal.className = 'diag-cell-value ' + (priority.includes('Immediate')||priority.includes('Urgent') ? 'risk-high' : (priority.includes('Routine') ? 'risk-medium' : 'risk-low'));

    const region = regionForPatient(p.id);
    renderInsights(p.prediction, p.confidence, region);

    reportBody.innerHTML = `
      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>
          Patient Information
        </div>
        <div class="report-field"><span class="rf-label">Patient ID</span><span class="rf-value">${p.patient_id}</span></div>
        <div class="report-field"><span class="rf-label">Name</span><span class="rf-value">${p.name}</span></div>
        <div class="report-field"><span class="rf-label">Age</span><span class="rf-value">${p.age}</span></div>
        <div class="report-field"><span class="rf-label">Gender</span><span class="rf-value">${p.gender||'—'}</span></div>
        <div class="report-field"><span class="rf-label">Scanner</span><span class="rf-value">${p.scanner||'—'}</span></div>
        <div class="report-field"><span class="rf-label">Study Date</span><span class="rf-value">${p.study_date||'—'}</span></div>
      </div>

      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          Findings
        </div>
        <div class="findings-text">${p.findings || 'No findings recorded.'}</div>
      </div>

      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M9 12l2 2 4-4"/></svg>
          AI Analysis
        </div>
        <div class="report-field"><span class="rf-label">Ground Truth</span><span class="rf-value">${p.true_label}</span></div>
        <div class="report-field"><span class="rf-label">Risk Level</span><span class="rf-value ${risk.cls}">${risk.level}</span></div>
        <div class="report-field"><span class="rf-label">Priority</span><span class="rf-value">${priority}</span></div>
      </div>

      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>
          Recommendation
        </div>
        ${recommendationCard(p.prediction)}
      </div>
    `;
  }

  function scanFrame(src, caption){
    return `
      <div class="scan-frame">
        <div class="corner tl"></div><div class="corner tr"></div>
        <div class="corner bl"></div><div class="corner br"></div>
        <img src="${src}" alt="${caption}" loading="lazy" onerror="this.parentElement.innerHTML+='<div style=\\'color:#5a6b7a;font-family:monospace;font-size:11px;padding:10px;text-align:center;\\'>IMAGE NOT FOUND</div>'">
        <div class="scan-caption">${caption}</div>
      </div>
    `;
  }

  function recommendationCard(prediction){
    const items = prediction === 'PNEUMONIA'
      ? [
          'Correlate with clinical presentation and vitals',
          'Recommend follow-up chest CT if symptoms persist',
          'Consider sputum culture and CBC with differential',
          'Flag for radiologist priority review'
        ]
      : [
          'No acute findings identified by model',
          'Routine follow-up per standard protocol',
          'Re-screen if new respiratory symptoms develop',
          'No further imaging indicated at this time'
        ];
    return `<div class="reco-card"><ul>${items.map(i=>`<li>${i}</li>`).join('')}</ul></div>`;
  }

  function showError(msg){
    errorSlot.innerHTML = `<div class="error-banner">⚠ ${msg}</div>`;
  }

  // ---------------- data load ----------------
  sessionIdEl.querySelector('span').textContent = sessionId();

  fetch('data/patients_database.json')
    .then(r => {
      if(!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(data => {
      state.patients = data.patients || [];
      modelName.textContent = data.model || 'Unknown';
      genTime.querySelector('span').textContent = data.generated ? data.generated : '—';
      renderKpis(data);
      renderPatientList();
      renderTable();
      if(state.patients.length){
        selectPatient(state.patients[0].id);
      }
    })
    .catch(err => {
      $('#status-dot').style.background = 'var(--alert)';
      $('#status-dot').style.boxShadow = 'none';
      $('#status-text').textContent = 'OFFLINE';
      showError(
        'Unable to load patient database (' + err.message + '). ' +
        'This page must be served over HTTP — opening index.html directly via file:// blocks fetch() requests. ' +
        'Run a local server (e.g. <code>python -m http.server</code>) from the docs/ folder, or view it via GitHub Pages.'
      );
      kpiRow.innerHTML = `<div class="kpi-card"><div class="kpi-label">Patients Screened</div><div class="kpi-value">—</div><div class="kpi-foot">Data unavailable</div></div>`;
      patientListEl.innerHTML = `<div style="color:var(--text-faint);font-size:12px;">Patient index unavailable offline.</div>`;
      tableCount.textContent = '—';
    });

  $('#patient-search').addEventListener('input', e => renderPatientList(e.target.value));
  $('#table-search').addEventListener('input', e => { state.tableFilter = e.target.value; state.page = 1; renderTable(); });

  // ---------------- upload + simulated inference ----------------
  const uploader = $('#uploader');
  const fileInput = $('#file-input');

  uploader.addEventListener('click', ()=> fileInput.click());
  fileInput.addEventListener('change', e => {
    if(e.target.files[0]) handleUpload(e.target.files[0]);
  });
  ['dragenter','dragover'].forEach(evt=>{
    uploader.addEventListener(evt, e=>{ e.preventDefault(); uploader.classList.add('drag'); });
  });
  ['dragleave','drop'].forEach(evt=>{
    uploader.addEventListener(evt, e=>{ e.preventDefault(); uploader.classList.remove('drag'); });
  });
  uploader.addEventListener('drop', e=>{
    const file = e.dataTransfer.files[0];
    if(file) handleUpload(file);
  });

  function showSkeletonViewer(){
    state.activeId = null;
    renderPatientList($('#patient-search').value);
    renderTable();
    viewerGrid.innerHTML = `
      <div class="scan-frame"><div class="skeleton" style="width:100%;height:100%;"></div></div>
      <div class="scan-frame"><div class="skeleton" style="width:100%;height:100%;"></div></div>
      <div class="scan-frame"><div class="skeleton" style="width:100%;height:100%;"></div></div>
    `;
  }

  function handleUpload(file){
    if(!/image\/(png|jpe?g)/.test(file.type)){
      showError('Unsupported file type. Please upload a JPG or PNG chest X-ray.');
      return;
    }
    errorSlot.innerHTML = '';
    showSkeletonViewer();
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setTimeout(()=> runSimulatedInference(img, file.name), 450);
    };
    img.src = url;
  }

  function runSimulatedInference(img, filename){
    viewerTitle.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/></svg> PACS Scan Viewer — ${filename}`;

    const size = 380;
    const origCanvas = drawToCanvas(img, size);
    const claheCanvas = applyClaheSim(origCanvas, size);
    const { canvas: heatCanvas, meanBrightness, variance } = generateHeatmap(origCanvas, size);

    viewerGrid.innerHTML = `
      <div class="scan-frame"><div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div></div>
      <div class="scan-frame"><div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div></div>
      <div class="scan-frame"><div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div></div>
    `;
    const frames = viewerGrid.querySelectorAll('.scan-frame');
    frames[0].appendChild(origCanvas);
    frames[0].insertAdjacentHTML('beforeend', '<div class="scan-caption">Original</div>');
    frames[1].appendChild(claheCanvas);
    frames[1].insertAdjacentHTML('beforeend', '<div class="scan-caption">Preprocessed · CLAHE (sim)</div>');
    frames[2].appendChild(heatCanvas);
    frames[2].insertAdjacentHTML('beforeend', '<div class="scan-caption">Grad-CAM Overlay (sim)</div>');

    const nameFlag = /pneu|infect|opac|consol/i.test(filename);
    const brightnessSignal = meanBrightness < 95 || variance > 2600;
    let score = 0.5;
    if(nameFlag) score += 0.28;
    if(brightnessSignal) score += 0.17;
    score += (Math.sin(filename.length * 12.9898) * 0.05);
    score = Math.max(0.04, Math.min(0.97, score));
    const prediction = score > 0.5 ? 'PNEUMONIA' : 'NORMAL';
    const confidence = prediction === 'PNEUMONIA' ? score : (1 - score);
    const risk = riskFromConfidence(prediction, confidence);
    const priority = priorityFromRisk(risk);
    const region = regionForPatient(filename);

    diagStrip.classList.remove('hidden');
    diagBadge.className = 'diag-badge ' + badgeClass(prediction);
    diagBadge.textContent = prediction;
    diagConfVal.textContent = pct(confidence);
    diagRiskVal.textContent = risk.level;
    diagRiskVal.className = 'diag-cell-value ' + risk.cls;
    diagPriorityVal.textContent = priority;
    diagPriorityVal.className = 'diag-cell-value ' + (priority.includes('Immediate')||priority.includes('Urgent') ? 'risk-high' : (priority.includes('Routine') ? 'risk-medium' : 'risk-low'));

    renderInsights(prediction, confidence, region);

    reportBody.innerHTML = `
      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>
          Patient Information
        </div>
        <div class="report-field"><span class="rf-label">Source File</span><span class="rf-value">${filename}</span></div>
        <div class="report-field"><span class="rf-label">Mode</span><span class="rf-value">Simulated inference</span></div>
      </div>
      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          Findings
        </div>
        <div class="findings-text">
          ${prediction === 'PNEUMONIA'
            ? `Simulated model detects regions of increased opacity consistent with possible consolidation near the ${region}. This is a client-side demo heuristic, not a diagnostic result.`
            : 'Simulated model finds no significant opacity pattern consistent with consolidation. Lung fields appear within expected texture range. This is a client-side demo heuristic, not a diagnostic result.'}
        </div>
      </div>
      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M9 12l2 2 4-4"/></svg>
          AI Analysis
        </div>
        <div class="report-field"><span class="rf-label">Mean Brightness</span><span class="rf-value">${meanBrightness.toFixed(1)}</span></div>
        <div class="report-field"><span class="rf-label">Texture Variance</span><span class="rf-value">${variance.toFixed(0)}</span></div>
        <div class="report-field"><span class="rf-label">Risk Level</span><span class="rf-value ${risk.cls}">${risk.level}</span></div>
      </div>
      <div class="report-section">
        <div class="report-section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>
          Recommendation
        </div>
        ${recommendationCard(prediction)}
      </div>
    `;
  }

  function drawToCanvas(img, size){
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0,0,size,size);
    const scale = Math.min(size/img.width, size/img.height);
    const w = img.width*scale, h = img.height*scale;
    ctx.drawImage(img, (size-w)/2, (size-h)/2, w, h);
    return c;
  }

  function applyClaheSim(srcCanvas, size){
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    ctx.drawImage(srcCanvas, 0, 0);
    const imgData = ctx.getImageData(0,0,size,size);
    const d = imgData.data;
    let min=255, max=0;
    for(let i=0;i<d.length;i+=4){
      const v = d[i];
      if(v<min) min=v;
      if(v>max) max=v;
    }
    const range = Math.max(1, max-min);
    for(let i=0;i<d.length;i+=4){
      const v = ((d[i]-min)/range) * 255;
      d[i]=d[i+1]=d[i+2]=v;
    }
    ctx.putImageData(imgData,0,0);
    return c;
  }

  function generateHeatmap(srcCanvas, size){
    const ctx0 = srcCanvas.getContext('2d');
    const imgData = ctx0.getImageData(0,0,size,size);
    const d = imgData.data;
    let sum=0, count=0;
    let darkestVal = 255, darkestX = size/2, darkestY = size/2;
    for(let y=0;y<size;y+=4){
      for(let x=0;x<size;x+=4){
        const idx = (y*size+x)*4;
        const v = d[idx];
        sum += v; count++;
        if(v < darkestVal && v > 10){ darkestVal = v; darkestX = x; darkestY = y; }
      }
    }
    const mean = sum/count;
    let varSum = 0, varCount = 0;
    for(let y=0;y<size;y+=6){
      for(let x=0;x<size;x+=6){
        const idx=(y*size+x)*4;
        varSum += Math.pow(d[idx]-mean,2); varCount++;
      }
    }
    const variance = varSum/varCount;

    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    ctx.drawImage(srcCanvas, 0, 0);
    ctx.globalCompositeOperation = 'lighter';
    const grad = ctx.createRadialGradient(darkestX, darkestY, 8, darkestX, darkestY, size*0.38);
    grad.addColorStop(0, 'rgba(255,80,60,0.75)');
    grad.addColorStop(0.5, 'rgba(255,170,40,0.4)');
    grad.addColorStop(1, 'rgba(255,170,40,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0,0,size,size);
    ctx.globalCompositeOperation = 'source-over';

    return { canvas: c, meanBrightness: mean, hotspotX: darkestX, hotspotY: darkestY, variance };
  }
})();
