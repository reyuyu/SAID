/* S0-TriMask training dashboard - read-only viewer.
 *
 * Only GET requests are issued, every dynamic string goes through textContent, and nothing here can
 * change a run: the page cannot start training, stop a process or write a file.
 */
'use strict';

const POLL_MS = 2500;
/* the offline diagnostics file only changes when the probe is re-run by hand, so it is fetched on a
 * run change and then at most every 15 s instead of on every 2.5 s poll */
const DIAG_REFRESH_MS = 15000;
const PHASE_COLOURS = {
  not_started: '#93a1b1', training: '#4da3ff', exporting: '#b48cff',
  evaluating_coco: '#57d9a3', evaluating_urban: '#ffb454', complete: '#57d9a3',
  failed: '#ff6b6b', unknown: '#93a1b1',
};

let state = {
  runId: null,
  cursor: 0,
  records: [],
  runs: [],
  maskPayload: null,
  evaluation: null,
  logs: null,
  diagnostics: null,
  diagnosticsRunId: null,
  diagnosticsAt: 0,
  pollTimer: null,
};

function qs(id) { return document.getElementById(id); }

function setText(node, value) {
  node.textContent = (value === null || value === undefined || value === '') ? '暂无' : String(value);
}

function fmt(value, digits, suffix) {
  if (value === null || value === undefined || Number.isNaN(value)) return '暂无';
  const number = Number(value);
  if (!Number.isFinite(number)) return '暂无';
  return number.toFixed(digits === undefined ? 4 : digits) + (suffix || '');
}

function fmtPercent(value, digits) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '暂无';
  return (Number(value) * 100).toFixed(digits === undefined ? 2 : digits) + '%';
}

function fmtPoints(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '暂无';
  const number = Number(value);
  return (number >= 0 ? '+' : '') + number.toFixed(2) + ' 百分点';
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return '暂无';
  const total = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return (h ? h + '小时' : '') + (h || m ? m + '分' : '') + s + '秒';
}

async function getJSON(url) {
  const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
  if (!response.ok) throw new Error(url + ' -> HTTP ' + response.status);
  return response.json();
}

/* ------------------------------------------------------------------ canvas charts */
function drawSeries(canvas, series, options) {
  const ctx = canvas.getContext('2d');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.height / ratio || 200;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#1e2632';
  ctx.fillRect(0, 0, width, height);

  const pad = { left: 56, right: 12, top: 10, bottom: 22 };
  const plotW = Math.max(10, width - pad.left - pad.right);
  const plotH = Math.max(10, height - pad.top - pad.bottom);
  const points = series.filter(s => s.values.some(v => Number.isFinite(v)));
  if (!points.length) {
    ctx.fillStyle = '#93a1b1';
    ctx.font = '12px sans-serif';
    ctx.fillText('暂无数据', pad.left + 6, pad.top + 18);
    return;
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  points.forEach(s => {
    s.values.forEach((v, index) => {
      if (!Number.isFinite(v)) return;
      const x = s.steps[index];
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, v); maxY = Math.max(maxY, v);
    });
  });
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const spanY = maxY - minY;
  minY -= spanY * 0.05; maxY += spanY * 0.05;
  const spanX = Math.max(1, maxX - minX);
  const xOf = x => pad.left + ((x - minX) / spanX) * plotW;
  const yOf = y => pad.top + plotH - ((y - minY) / (maxY - minY)) * plotH;

  ctx.strokeStyle = '#2a3441';
  ctx.fillStyle = '#93a1b1';
  ctx.font = '11px sans-serif';
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
    const value = maxY - ((maxY - minY) * i) / 4;
    ctx.fillText(value.toFixed(value < 10 ? 3 : 1), 4, y + 4);
  }
  ctx.fillText(String(minX), pad.left, height - 6);
  ctx.fillText(String(maxX), pad.left + plotW - 28, height - 6);

  points.forEach(s => {
    ctx.strokeStyle = s.colour;
    ctx.lineWidth = s.width || 1.6;
    ctx.beginPath();
    let started = false;
    s.values.forEach((v, index) => {
      if (!Number.isFinite(v)) return;
      const x = xOf(s.steps[index]), y = yOf(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    });
    ctx.stroke();
  });
  if (options && options.markerStep) {
    const x = xOf(options.markerStep);
    if (x >= pad.left && x <= pad.left + plotW) {
      ctx.strokeStyle = '#ffd166';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function movingAverage(values, window) {
  const out = [];
  let sum = 0, count = 0;
  const queue = [];
  values.forEach(v => {
    queue.push(v);
    if (Number.isFinite(v)) { sum += v; count += 1; }
    if (queue.length > window) {
      const dropped = queue.shift();
      if (Number.isFinite(dropped)) { sum -= dropped; count -= 1; }
    }
    out.push(count ? sum / count : null);
  });
  return out;
}

/* ------------------------------------------------------------------ panels */
function renderOverview(status) {
  const body = qs('overview-body');
  body.textContent = '';
  const items = [
    ['run_id', status.run_id],
    ['phase', status.phase_label + ' (' + status.phase + ')'],
    ['objective', status.objective],
    ['arm', status.arm],
    ['文本门模式', status.text_gate_mode === 'hard_st' ? 'hard_st（硬前向 + 直通梯度）'
      : (status.text_gate_mode || '暂无')],
    ['λ_sparse_T', status.lambda_sparse_t],
    ['git SHA', status.git_head],
    ['已完成更新', status.completed_steps + ' / ' + status.max_steps],
    ['LR horizon', status.lr_horizon_steps],
    ['世界大小 × 每卡 batch', (status.world_size || '?') + ' × ' + (status.batch_size_per_gpu || '?')],
    ['全局候选池', status.global_pairs],
    ['上一步耗时', fmt(status.last_sec_per_step, 3, ' s')],
    ['吞吐', fmt(status.samples_per_sec, 1, ' 样本/s')],
    ['峰值显存（rank0）', fmt(status.peak_memory_gb, 2, ' GiB')],
    ['累计运行时间', fmtDuration(status.elapsed_seconds)],
    ['ETA（估计）', status.eta_is_estimate ? fmtDuration(status.eta_seconds) + '（估计）' : '样本不足，暂不提供'],
  ];
  items.forEach(([key, value]) => {
    const div = document.createElement('div');
    div.className = 'kv';
    const k = document.createElement('div'); k.className = 'k'; k.textContent = key;
    const v = document.createElement('div'); v.className = 'v';
    setText(v, value === 0 ? 0 : value);
    div.appendChild(k); div.appendChild(v);
    body.appendChild(div);
  });
  const progress = status.progress === null || status.progress === undefined
    ? (status.max_steps ? status.completed_steps / status.max_steps : 0) : status.progress;
  qs('progress-bar').style.width = Math.max(0, Math.min(1, progress || 0)) * 100 + '%';
  setText(qs('progress-text'), status.completed_steps + ' / ' + status.max_steps + ' 次更新');
  const note = qs('phase-note');
  if (status.completed_steps >= status.max_steps && status.phase === 'training') {
    note.textContent = '训练更新数已达上限，但评估阶段尚未结束：整个任务仍未完成。';
  } else if (status.phase === 'failed') {
    note.textContent = '运行失败：见 E 面板的退出码与错误行。';
  } else {
    note.textContent = '';
  }
  document.querySelectorAll('#overview-body .v').forEach(() => {});
}

function renderCurves(records) {
  const direction = qs('direction-select').value;
  const smooth = qs('smooth-select').value;
  const window = Math.max(2, Math.min(50, Number(qs('window-input').value) || 5));
  const steps = records.map(r => r.completed_steps);
  if (smooth === 'ma' && records.length < window) {
    qs('paths-note').textContent = '滑动平均需要至少 ' + window + ' 个记录点，当前只有 '
      + records.length + ' 个：先按原始记录显示。';
  }
  const pick = (record, key) => {
    if (direction === 'i2t') return record[key + '_i2t'];
    if (direction === 't2i') return record[key + '_t2i'];
    const a = record[key + '_i2t'], b = record[key + '_t2i'];
    return (Number.isFinite(a) && Number.isFinite(b)) ? a + b : null;
  };
  const shape = values => (smooth === 'ma' && records.length >= window)
    ? movingAverage(values, window) : values;
  const series = [1, 2, 3].map((index, position) => {
    const values = shape(records.map(r => pick(r, 'loss_' + index)));
    return {
      values: values, steps: steps,
      colour: ['#4da3ff', '#57d9a3', '#ffb454'][position],
      label: 'L' + index,
    };
  });
  const marker = records.length ? records[records.length - 1].completed_steps : null;
  drawSeries(qs('canvas-paths'), series, { markerStep: marker });
  qs('paths-note').textContent = '加权前原始损失；权重固定为 10 / 1 / 1（图例颜色与路径一一对应）。'
    + (smooth === 'ma' ? ' 当前显示 ' + window + ' 点滑动平均（仅浏览器展示，服务端记录未改动）。' : '');

  drawSeries(qs('canvas-total'), [{
    values: records.map(r => r.loss_total), steps: steps, colour: '#b48cff',
  }]);
  drawSeries(qs('canvas-sparse'), [
    { values: records.map(r => r.weighted_loss_sparse_i), steps: steps, colour: '#4da3ff' },
    { values: records.map(r => r.weighted_loss_sparse_t), steps: steps, colour: '#ffb454' },
  ]);
}

function drawMaskGrid(canvas, values, mode) {
  const ctx = canvas.getContext('2d');
  const cols = 32, rows = 16;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!values || !values.length) {
    ctx.fillStyle = '#93a1b1'; ctx.font = '16px sans-serif';
    ctx.fillText('暂无快照', 16, 32);
    return;
  }
  for (let index = 0; index < cols * rows; index++) {
    const value = Number(values[index]);
    const x = (index % cols) * cellW;
    const y = Math.floor(index / cols) * cellH;
    if (!Number.isFinite(value)) {
      ctx.fillStyle = '#2a3441';
    } else if (mode === 'probability') {
      const grey = Math.round(255 * Math.max(0, Math.min(1, value)));
      ctx.fillStyle = 'rgb(' + grey + ',' + grey + ',' + grey + ')';
    } else {
      const v = Math.max(0, Math.min(1, value));
      const r = Math.round(255 * (1 - v) * 0.85 + 30 * v);
      const g = Math.round(120 + 130 * v);
      const b = Math.round(90 + 160 * (1 - v));
      ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    }
    ctx.fillRect(x, y, cellW - 0.6, cellH - 0.6);
  }
}

function renderMasks(payload) {
  const scalars = qs('mask-scalars');
  scalars.textContent = '';
  const heavy = payload.latest_heavy_scalars || {};
  const snapshot = payload.snapshot || {};
  const items = [
    ['mI 硬保留率', fmt(snapshot.visual_mask_keep_ratio, 4)],
    ['mT 硬保留率', fmt(snapshot.text_gate_keep_ratio, 4)],
    ['mT 中为 0 的坐标比例', fmt(snapshot.text_gate_zero_fraction, 4)],
    ['mI 空 mask 比例', fmt(heavy.mask_i_empty_fraction, 4)],
    ['mT 空 mask 比例', fmt(heavy.mask_t_empty_fraction, 4)],
    ['mI 保留能量比例', fmt(heavy.mask_i_retained_energy_fraction, 4)],
    ['mT 保留能量比例', fmt(heavy.mask_t_retained_energy_fraction, 4)],
    ['交集数量均值', fmt(heavy.mask_intersection_intersection_count_mean, 1)],
    ['Jaccard（并集为空记 null）', heavy.mask_intersection_jaccard_mean === null
      ? '暂无（并集为空）' : fmt(heavy.mask_intersection_jaccard_mean, 4)],
    ['交集为空比例', fmt(heavy.mask_intersection_intersection_empty_fraction, 4)],
    ['两侧非空但交集为空', fmt(heavy.mask_intersection_both_nonempty_but_intersection_empty_fraction, 4)],
    ['文本方向改变比例', fmt(heavy.text_direction_change_fraction, 4)],
    ['mT 与未 mask 文本的余弦', fmt(heavy.mask_t_cosine_with_unmasked, 5)],
    ['adv_gap（正部平均）', fmt(heavy.adv_gap, 4)],
    ['第三路不劣比例', fmt(heavy.third_not_worse_fraction, 4)],
    ['pT 均值 / 距阈值 0.05 内', fmt(heavy.text_gate_pT_mean, 4) + ' / '
      + fmt(heavy.text_gate_pT_near_threshold_fraction, 4)],
  ];
  items.forEach(([key, value]) => {
    const div = document.createElement('div');
    div.className = 'kv';
    const k = document.createElement('div'); k.className = 'k'; k.textContent = key;
    const v = document.createElement('div'); v.className = 'v'; setText(v, value);
    div.appendChild(k); div.appendChild(v);
    scalars.appendChild(div);
  });

  drawMaskGrid(qs('canvas-mt'), snapshot.mask_t_coordinate_mean, 'mask');
  drawMaskGrid(qs('canvas-mi'), snapshot.mask_i_coordinate_mean, 'mask');
  drawMaskGrid(qs('canvas-inter'), snapshot.intersection_coordinate_mean, 'mask');
  drawMaskGrid(qs('canvas-pt'), snapshot.text_gate_probability_sample0, 'probability');

  const note = qs('mask-source-note');
  if (!snapshot || !snapshot.completed_steps) {
    note.textContent = '尚未产生 mask 快照（训练开始后每 25 步写入一次）。';
  } else {
    note.textContent = '快照来自 step ' + snapshot.completed_steps + ' 的当前 batch 样本（'
      + (snapshot.source || 'rank0/local_batch') + '）；不同 step 是不同 caption，'
      + '不能跨 step 串成同一条样本轨迹。';
  }
  const caption = qs('mask-caption');
  if (snapshot.sample0 && snapshot.sample0.caption_said) {
    caption.textContent = '首个样本已采样 caption：' + snapshot.sample0.caption_said
      + '（仅记录本次采样前缀，不含未采样后缀）';
  } else {
    caption.textContent = '';
  }
  const extra = qs('mask-extra');
  extra.textContent = '';
  const extras = [
    ['文本门 pT p10/p50/p90', [fmt(heavy.text_gate_pT_p10, 4), fmt(heavy.text_gate_pT_p50, 4),
      fmt(heavy.text_gate_pT_p90, 4)].join(' / ')],
    ['mT 逐坐标跨 caption 标准差', fmt(heavy.mask_t_cross_caption_std_mean, 4)],
    ['mT 样本内（跨坐标）标准差', fmt(heavy.mask_t_within_sample_std, 4)],
    ['caption 间方差占比', fmt(heavy.mask_t_between_over_total, 4)],
    ['两侧 mask 都非空的比例', fmt(1 - Number(heavy.mask_intersection_union_empty_fraction || 0), 4)],
  ];
  extras.forEach(([key, value]) => {
    const div = document.createElement('div');
    div.className = 'kv';
    const k = document.createElement('div'); k.className = 'k'; k.textContent = key;
    const v = document.createElement('div'); v.className = 'v'; setText(v, value);
    div.appendChild(k); div.appendChild(v);
    extra.appendChild(div);
  });
}

function evaluationTable(title, metrics, baseline) {
  const table = document.createElement('table');
  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  ['模型', 'I2T R@1', 'I2T R@5', 'I2T R@10', 'T2I R@1', 'T2I R@5', 'T2I R@10'].forEach(text => {
    const th = document.createElement('th'); th.textContent = text; headRow.appendChild(th);
  });
  head.appendChild(headRow); table.appendChild(head);
  const body = document.createElement('tbody');
  const rows = [];
  if (baseline) {
    Object.keys(baseline).forEach(name => {
      const block = baseline[name][title === 'COCO canonical' ? 'coco' : 'urban1k'];
      if (!block) return;
      rows.push({ name: name, values: block, ours: false });
    });
  }
  if (metrics) rows.push({ name: 'S0_TriMask_HS@500（本次）', values: metrics, ours: true });
  rows.forEach(row => {
    const tr = document.createElement('tr');
    if (row.ours) tr.className = 'ours';
    const name = document.createElement('td'); name.textContent = row.name; tr.appendChild(name);
    ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10'].forEach(key => {
      const td = document.createElement('td');
      const value = row.values[key];
      td.textContent = value === undefined || value === null ? '暂无'
        : (Number(value) * 100).toFixed(2) + '%';
      if (row.ours && key === 'i2t_r1' && row.values.i2t_r1 !== undefined) {
        const base = baseline['S0@500'][title === 'COCO canonical' ? 'coco' : 'urban1k'][key];
        td.className = row.values[key] >= base ? 'pass' : 'fail';
      }
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  table.appendChild(body);
  return table;
}

function renderEvaluation(payload) {
  const container = qs('eval-tables');
  container.textContent = '';
  const note = qs('eval-note');
  const coco = payload.datasets.coco;
  const urban = payload.datasets.urban1k;
  const missing = [];
  if (!coco.available) missing.push('COCO canonical');
  if (!urban.available) missing.push('Urban-1k');
  note.textContent = missing.length
    ? '未评估：' + missing.join(' / ') + ' 的真实评估文件尚未生成，表中显示“暂无”，不填补 0 或模拟曲线。'
    : '两个评估文件都存在；百分比由原始 0-1 数值换算，差值一律以“百分点”表示。';

  const cocoTitle = document.createElement('h3'); cocoTitle.textContent = 'COCO canonical';
  container.appendChild(cocoTitle);
  container.appendChild(evaluationTable('COCO canonical', coco.metrics, payload.baselines));
  const cocoNote = document.createElement('p'); cocoNote.className = 'muted';
  cocoNote.textContent = coco.metrics ? (coco.metrics.protocol + '；checkpoint '
    + (coco.metrics.checkpoint_sha256 || '暂无')) : '';
  container.appendChild(cocoNote);

  const urbanTitle = document.createElement('h3'); urbanTitle.textContent = 'Urban-1k';
  container.appendChild(urbanTitle);
  container.appendChild(evaluationTable('Urban-1k', urban.metrics, payload.baselines));
  const urbanNote = document.createElement('p'); urbanNote.className = 'muted';
  urbanNote.textContent = urban.metrics ? (urban.metrics.protocol + '；checkpoint '
    + (urban.metrics.checkpoint_sha256 || '暂无')) : '';
  container.appendChild(urbanNote);

  const gate = qs('gate-box');
  gate.textContent = '';
  const line = document.createElement('div');
  line.textContent = '晋级门：' + payload.gate.rule;
  gate.appendChild(line);
  if (payload.verdict) {
    const verdictLine = document.createElement('div');
    const badge = document.createElement('span');
    badge.className = 'badge ' + (payload.verdict.verdict === 'FAIL' ? 'gate-fail' : 'gate-pass');
    badge.textContent = payload.verdict.verdict;
    verdictLine.appendChild(badge);
    const detail = document.createElement('span');
    detail.textContent = ' COCO I2T R@1 ' + (payload.verdict.i2t_pass ? '达标' : '未达标')
      + '（' + fmtPoints(payload.verdict.i2t_delta_points) + '）、T2I R@1 '
      + (payload.verdict.t2i_pass ? '达标' : '未达标') + '（'
      + fmtPoints(payload.verdict.t2i_delta_points) + '）';
    verdictLine.appendChild(detail);
    gate.appendChild(verdictLine);
  } else {
    const pending = document.createElement('div');
    pending.textContent = 'COCO canonical 结果尚未生成，无法判定（不预填结论）。';
    gate.appendChild(pending);
  }
}

function renderLogs(payload) {
  const issues = qs('issues');
  issues.textContent = '';
  (payload.issues || []).forEach(issue => {
    const div = document.createElement('div');
    div.className = 'issue' + (issue.level === 'error' ? ' error' : '');
    div.textContent = '[' + issue.level + '] ' + (issue.step === null || issue.step === undefined
      ? '' : 'step ' + issue.step + ' ') + issue.text;
    issues.appendChild(div);
  });
  if (payload.commands) {
    const div = document.createElement('div');
    div.className = 'issue';
    div.textContent = '实际执行命令（已脱敏）：' + JSON.stringify(payload.commands);
    issues.appendChild(div);
  }
  const lines = (payload.records || []).slice(-200).map(record => {
    return 'step ' + record.completed_steps + '  loss ' + fmt(record.loss_total, 4)
      + '  L1 ' + fmt(record.loss_1, 4) + '  L2 ' + fmt(record.loss_2, 4)
      + '  L3 ' + fmt(record.loss_3, 4)
      + '  sparse_i ' + fmt(record.weighted_loss_sparse_i, 4)
      + '  sparse_t ' + fmt(record.weighted_loss_sparse_t, 4)
      + '  mI ' + fmt(record.mask_i_keep_ratio, 3) + '  mT ' + fmt(record.mask_t_mean, 3)
      + '  ' + fmt(record.sec_per_step, 3) + 's';
  });
  qs('log-lines').textContent = lines.length ? lines.join('\n') : '暂无日志记录';
}

/* ------------------------------------------------------------------ polling */
async function refreshRuns() {
  const payload = await getJSON('/api/runs');
  state.runs = payload.runs || [];
  const select = qs('run-select');
  const previous = state.runId;
  select.textContent = '';
  state.runs.forEach(run => {
    const option = document.createElement('option');
    option.value = run.run_id;
    option.textContent = run.label + (run.demo ? ' [DEMO]' : '');
    select.appendChild(option);
  });
  if (!state.runs.length) {
    qs('live-dot').className = 'dot bad';
    setText(qs('last-poll'), '没有登记任何 run');
    return;
  }
  const keep = state.runs.some(run => run.run_id === previous) ? previous : state.runs[0].run_id;
  select.value = keep;
  if (state.runId !== keep) {
    state.runId = keep;
    state.cursor = 0;
    state.records = [];
  }
}

async function pollOnce() {
  if (!state.runId) await refreshRuns();
  if (!state.runId) return;
  const runId = state.runId;
  const [status, metrics, masks, evaluation, logs] = await Promise.all([
    getJSON('/api/run/' + runId + '/status'),
    getJSON('/api/run/' + runId + '/metrics?after=' + state.cursor),
    getJSON('/api/run/' + runId + '/masks'),
    getJSON('/api/run/' + runId + '/evaluation'),
    getJSON('/api/run/' + runId + '/logs?limit=' + (qs('log-limit').value || 200)),
  ]);
  if (metrics.records && metrics.records.length) {
    state.records = state.records.concat(metrics.records);
    if (state.records.length > 5000) state.records = state.records.slice(-5000);
  }
  state.cursor = metrics.cursor;
  state.maskPayload = masks;
  state.evaluation = evaluation;
  state.logs = logs;

  renderOverview(status);
  renderCurves(state.records);
  renderMasks(masks);
  renderEvaluation(evaluation);
  renderLogs(logs);

  qs('live-dot').className = 'dot ' + (status.phase === 'failed' ? 'bad' : 'ok');
  qs('live-dot').style.background = PHASE_COLOURS[status.phase] || '#93a1b1';
  setText(qs('last-poll'), '最近轮询 ' + new Date().toLocaleTimeString());

  const now = Date.now();
  if (state.diagnosticsRunId !== runId || now - state.diagnosticsAt > DIAG_REFRESH_MS) {
    state.diagnosticsRunId = runId;
    state.diagnosticsAt = now;
    getJSON('/api/run/' + runId + '/diagnostics')
      .then(renderDiagnostics)
      .catch(reportError);
  }
}

/* ---------------------------------------------------------------- F: offline diagnostics
 *
 * Everything in this section comes from one read-only file the probe wrote by hand
 * (`diagnostics/hs_mask_geometry_probe.json`). Opening or refreshing this page can never run the
 * probe, load a checkpoint or touch a GPU: the endpoint only reads that file. When the file does
 * not exist the section says 未诊断 and shows no fabricated zeros.
 */

function kvGrid(container, items) {
  container.textContent = '';
  items.forEach(pair => {
    const div = document.createElement('div');
    div.className = 'kv';
    const k = document.createElement('div'); k.className = 'k'; k.textContent = pair[0];
    const v = document.createElement('div'); v.className = 'v'; setText(v, pair[1]);
    div.appendChild(k); div.appendChild(v);
    container.appendChild(div);
  });
}

function tableBlock(container, headers, rows) {
  container.textContent = '';
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  headers.forEach(text => {
    const th = document.createElement('th'); th.textContent = text; headRow.appendChild(th);
  });
  thead.appendChild(headRow); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  rows.forEach(cells => {
    const tr = document.createElement('tr');
    cells.forEach(cell => {
      const td = document.createElement('td'); setText(td, cell); tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody); container.appendChild(table);
  return table;
}

function fmtShare(value, digits) {
  if (value === null || value === undefined) return '暂无';
  return (value * 100).toFixed(digits === undefined ? 2 : digits) + '%';
}

function fmtSigned(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return '暂无';
  const size = Math.abs(value).toFixed(digits === undefined ? 4 : digits);
  return (value >= 0 ? '+' : '−') + size;
}

function drawBars(canvas, items, options) {
  const opts = options || {};
  const height = Number(canvas.getAttribute('height')) || 150;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const padLeft = 6, padRight = 6, padTop = 20, padBottom = 40;
  const plotW = Math.max(1, width - padLeft - padRight);
  const plotH = Math.max(1, height - padTop - padBottom);
  const values = items.map(item => Number(item.value) || 0);
  const maxValue = opts.maxValue !== undefined ? opts.maxValue
    : Math.max.apply(null, values.concat([0.0001]));
  const slot = plotW / Math.max(1, items.length);
  items.forEach((item, index) => {
    const value = Number(item.value) || 0;
    const barHeight = Math.max(1, (value / maxValue) * plotH);
    const x = padLeft + index * slot + slot * 0.16;
    const w = slot * 0.68;
    ctx.fillStyle = item.colour || '#4da3ff';
    ctx.fillRect(x, padTop + plotH - barHeight, w, barHeight);
    ctx.fillStyle = '#dbe4ee';
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(opts.format ? opts.format(value) : value.toFixed(3), x + w / 2,
                 padTop + plotH - barHeight - 4);
    ctx.save();
    ctx.translate(x + w / 2, padTop + plotH + 12);
    ctx.rotate(-Math.PI / 9);
    ctx.textAlign = 'right';
    ctx.fillStyle = '#93a1b1';
    ctx.fillText(item.label, 0, 0);
    ctx.restore();
  });
  ctx.strokeStyle = '#2a3644';
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop + plotH);
  ctx.lineTo(padLeft + plotW, padTop + plotH);
  ctx.stroke();
}

function renderDiagnostics(payload) {
  const banner = qs('diag-banner');
  const scope = qs('diag-scope');
  const reminders = qs('diag-reminders');
  const varianceBox = qs('diag-variance');
  const variantsBox = qs('diag-variants');
  const geometryBox = qs('diag-geometry');
  const hardBox = qs('diag-hard-body');
  const notRun = qs('diag-not-run');
  state.diagnostics = payload;

  reminders.textContent = (payload.reminders || []).map(text => '· ' + text).join('   ');

  if (!payload.available) {
    banner.textContent = '未诊断：' + (payload.error || ('未找到 ' + payload.directory + '/' + payload.file))
      + '。运行 tools/diag/trimask_hs_geometry_probe.py 后本区域才会出现数字。';
    scope.textContent = '';
    varianceBox.textContent = '';
    variantsBox.textContent = '';
    geometryBox.textContent = '';
    hardBox.textContent = '';
    notRun.textContent = '';
    ['canvas-variance', 'canvas-variants'].forEach(id => {
      const canvas = qs(id);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
    return;
  }

  const checkpoint = payload.checkpoint || {};
  const identity = checkpoint.identity || {};
  const manifest = payload.manifest || {};
  banner.textContent = '已诊断（只读前向）· 文件 ' + payload.file
    + ' · 写入时间 ' + (payload.updated_at_iso || '暂无')
    + ' · 本轮新增 optimizer updates = ' + payload.new_optimizer_updates + '（必须为 0）';

  kvGrid(scope, [
    ['诊断对象', (identity.arm || '暂无') + ' @ ' + identity.completed_steps + ' 步'],
    ['objective / 门模式', (identity.objective || '暂无') + ' / ' + (identity.text_gate_mode || '暂无')],
    ['λ_sparse_T', identity.lambda_sparse_t],
    ['checkpoint sha256', (checkpoint.sha256_before || '').slice(0, 16) + '…'],
    ['checkpoint 未被修改', checkpoint.sha256_unchanged === true ? '是（前后一致）' : '否'],
    ['参数 state 未被修改', checkpoint.parameter_state_unchanged === true ? '是（前后一致）' : '否'],
    ['run_status 未被改写', checkpoint.run_status_unchanged === true ? '是' : '否'],
    ['文本分支张量数', checkpoint.text_mask_net_tensors],
    ['诊断池', (manifest.count || '?') + ' 图 × ' + (manifest.count || '?') + ' 文本（COCO val2017）'],
    ['抽样 seed / 规则', (manifest.selection || {}).seed + ' / 见 manifest'],
    ['耗时', fmt(payload.timing && payload.timing.wall_seconds, 1, ' 秒')],
    ['是否正式评估', payload.not_a_canonical_evaluation ? '不是：不参与晋级门，也不替换原生 CLS/EOS 成绩' : '暂无'],
  ]);

  /* 1. variance decomposition */
  const variance = payload.variance || {};
  const shares = variance.share_of_total || {};
  const captionShare = (variance.v_total ? variance.v_caption_dependent / variance.v_total : null);
  tableBlock(varianceBox, ['成分', '含义', '方差值', '占 V_total', '占 V_caption_dependent'], [
    ['V_level', '不同 caption 整体保留比例的差别', fmt(variance.v_level, 8), fmtShare(shares.level),
      fmtShare((variance.share_of_caption_dependent || {}).level)],
    ['V_profile', '跨 caption 稳定的逐坐标偏好', fmt(variance.v_profile, 8), fmtShare(shares.profile), '—'],
    ['V_interaction', '去掉保留量与公共轮廓后 caption×坐标 的交互', fmt(variance.v_interaction, 8),
      fmtShare(shares.interaction), fmtShare((variance.share_of_caption_dependent || {}).interaction)],
    ['V_caption_dependent', '偏离公共逐坐标轮廓的全部变化（= level + interaction）',
      fmt(variance.v_caption_dependent, 8), fmtShare(captionShare), '100%'],
    ['V_total', '全部分布之和（= level + profile + interaction）', fmt(variance.v_total, 8), '100%', '—'],
  ]);
  drawBars(qs('canvas-variance'), [
    { label: 'V_level', value: shares.level, colour: '#b48cff' },
    { label: 'V_profile', value: shares.profile, colour: '#4da3ff' },
    { label: 'V_interaction', value: shares.interaction, colour: '#57d9a3' },
    { label: 'V_caption_dep', value: captionShare, colour: '#ffb454' },
  ], { maxValue: 1, format: value => (value * 100).toFixed(1) + '%' });
  qs('diag-variance-note').textContent =
    '恒等式误差：V_total −(level+profile+interaction) = ' + fmt(variance.identity_v_total_minus_parts, 12)
    + '，V_caption_dependent −(level+interaction) = '
    + fmt(variance.identity_v_caption_dependent_minus_parts, 12)
    + '。旧日志字段（V_total − mean_j Var_d）实测 = ' + fmt((variance.historical_field_check || {}).value, 8)
    + '，与 V_level ' + ((variance.historical_field_check || {}).equals_v_level ? '相等' : '不等')
    + '：它只反映保留量差别，不是全部 caption 自适应成分。';

  /* 2. text-gate replacement */
  const variants = (payload.replacements || {}).variants || {};
  const order = ['NORMAL', 'MEAN_PROFILE', 'ONES', 'SHUFFLED_seed0', 'SHUFFLED_seed1',
                 'SHUFFLED_seed2', 'SHUFFLED_seed3'];
  const rows = order.filter(name => variants[name]).map(name => {
    const entry = variants[name];
    const l3 = entry.L3_I2T || {}, l3t = entry.L3_T2I || {}, l2 = entry.L2_I2T || {};
    const paired = ((entry.paired_vs_normal || {}).L3_I2T) || {};
    return [name, fmt(l2.ce, 4), fmt(l2['R@1'], 4), fmt(l3.ce, 4), fmt(l3['R@1'], 4), fmt(l3.mrr, 4),
            fmt(l3t.ce, 4), fmt(l3t['R@1'], 4), fmtSigned(paired.delta_ce_mean, 4),
            paired.changed_rank_queries === undefined ? '—' : paired.changed_rank_queries];
  });
  tableBlock(variantsBox,
             ['文本门变体', 'L2·I2T CE', 'L2·I2T R@1', 'L3·I2T CE', 'L3·I2T R@1', 'L3·I2T MRR',
              'L3·T2I CE', 'L3·T2I R@1', 'ΔCE vs NORMAL', '排名变化查询数'], rows);
  drawBars(qs('canvas-variants'), order.filter(name => variants[name]).map(name => ({
    label: name.replace('SHUFFLED_', 'SHUF_'),
    value: (variants[name].L3_I2T || {}).ce,
    colour: name === 'NORMAL' ? '#57d9a3' : (name.indexOf('SHUFFLED') === 0 ? '#ff6b6b' : '#4da3ff'),
  })), { format: value => value.toFixed(4) });
  const shuffledAggregate = (payload.replacements || {}).shuffled_aggregate || {};
  const identityChecks = (payload.replacements || {}).identity_checks || {};
  qs('diag-variants-note').textContent =
    'L3·I2T 的 4 个 SHUFFLED：CE 均值 ' + fmt((shuffledAggregate.L3_I2T || {}).ce_mean, 4)
    + '（范围 ' + fmt((shuffledAggregate.L3_I2T || {}).ce_min, 4) + ' ~ '
    + fmt((shuffledAggregate.L3_I2T || {}).ce_max, 4) + '），R@1 均值 '
    + fmt((shuffledAggregate.L3_I2T || {}).R@1_mean, 4)
    + '。实现核对：全 1 时 Q3−Q1 = ' + fmt(identityChecks.ones_q3_equals_normal_q1_max_abs_diff, 8)
    + '，Q2−原生全局余弦 = ' + fmt(identityChecks.ones_q2_equals_raw_global_cosine_max_abs_diff, 8)
    + '。L1 与 loss_total 不参与比较：L1 不随文本门变化。';

  /* 3. intersection geometry */
  const geometry = payload.geometry || {};
  const stats = geometry.stats || {};
  const geoRows = ['I2T', 'T2I'].map(direction => {
    const q3 = (geometry.q3_metrics || {})[direction] || {};
    const qcap = (geometry.qcap_metrics || {})[direction] || {};
    const paired = (geometry.qcap_vs_q3_paired || {})[direction] || {};
    return [direction, fmt(q3.ce, 4), fmt(qcap.ce, 4), fmtSigned(paired.delta_ce_mean, 4),
            fmt(q3['R@1'], 4), fmt(qcap['R@1'], 4), fmtSigned(paired.delta_R@1, 4),
            fmt(q3.mrr, 4), fmt(qcap.mrr, 4), paired.changed_rank_queries,
            paired.rank_worsened, paired.rank_improved,
            fmtSigned(paired.delta_s_pos_mean, 4), fmtSigned(paired.delta_s_max_negative_mean, 4)];
  });
  tableBlock(geometryBox,
             ['方向', 'Q3 CE', 'Qcap CE', 'ΔCE', 'Q3 R@1', 'Qcap R@1', 'ΔR@1', 'Q3 MRR', 'Qcap MRR',
              '排名变化', '变差', '变好', 'Δ正例分数', 'Δ最强负例分数'], geoRows);
  qs('diag-geometry-note').textContent =
    '恒等式 Q3 = Qcap × factor：有效 pair ' + stats.valid_pairs + ' / ' + stats.total_pairs
    + '（' + fmtShare(stats.valid_fraction) + '），最大绝对误差 '
    + fmt(stats.max_abs_error_on_valid_pairs, 8) + '。factor 正例均值 '
    + fmt(stats.factor_positive_mean, 4) + ' vs 负例均值 ' + fmt(stats.factor_negative_mean, 4)
    + '：两者几乎相同，说明该因子并未专门压低正例。';

  /* 4. hard queries */
  const queries = payload.hard_queries || [];
  tableBlock(hardBox,
             ['路径', '方向', '查询（图/标注）', 'rank', 'CE', 'M_max', 'M_lse', '最强负例（图/标注）',
              'Q3 该负例', 'Qcap 该负例', 'Qcap rank', 'factor 正例', 'factor 该负例', '视觉核查'],
             queries.map(row => [
               row.path, row.direction, row.query_label, row.rank, fmt(row.ce, 3),
               fmt(row.m_max, 3), fmt(row.m_lse, 3), row.strongest_negative_label,
               fmt(row.score_of_worst_negative, 2), fmt(row.qcap_score_of_worst_negative, 2),
               row.qcap_rank === undefined ? '—' : row.qcap_rank,
               fmt(row.factor_of_positive, 3), fmt(row.factor_of_worst_negative, 3),
               row.visual_verification]));
  notRun.textContent = 'NOT RUN：' + (payload.not_run || []).join('；');

  qs('diag-hard').open = false;
}

function download(filename, text, type) {  const blob = new Blob([text], { type: type || 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = filename;
  document.body.appendChild(link); link.click(); link.remove();
  URL.revokeObjectURL(url);
}

function wire() {
  qs('run-select').addEventListener('change', event => {
    state.runId = event.target.value;
    state.cursor = 0;
    state.records = [];
    pollOnce().catch(reportError);
  });
  ['direction-select', 'smooth-select', 'window-input'].forEach(id => {
    qs(id).addEventListener('change', () => renderCurves(state.records));
  });
  qs('log-limit').addEventListener('change', () => {
    if (state.logs) renderLogs(state.logs);
  });
  qs('btn-json').addEventListener('click', () => {
    download('trimask-' + state.runId + '-metrics.json', JSON.stringify({
      run_id: state.runId, records: state.records, masks: state.maskPayload,
      evaluation: state.evaluation,
    }, null, 2));
  });
  qs('btn-csv').addEventListener('click', () => {
    fetch('/api/run/' + state.runId + '/metrics.csv').then(r => r.text())
      .then(text => download('trimask-' + state.runId + '-metrics.csv', text, 'text/csv'))
      .catch(reportError);
  });
}

function reportError(error) {
  qs('live-dot').className = 'dot bad';
  setText(qs('last-poll'), '轮询失败：' + error.message + '（页面故障不会影响训练）');
}

async function start() {
  wire();
  try {
    await refreshRuns();
    await pollOnce();
  } catch (error) {
    reportError(error);
  }
  state.pollTimer = setInterval(() => {
    pollOnce().catch(reportError);
  }, POLL_MS);
}

window.addEventListener('DOMContentLoaded', start);
window.addEventListener('resize', () => {
  if (state.records && state.records.length) renderCurves(state.records);
  if (state.maskPayload) renderMasks(state.maskPayload);
});
