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
  nuisance: null,
  clip512: null,
  pgclip: null,
  cgclip: null,
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
    ['损失 profile', status.loss_profile === 'balanced' ? 'balanced（均衡权重）' : 'default（冻结权重）'],
    ['对齐权重 λ1/λ2/λ3', [status.lambda_1, status.lambda_2, status.lambda_3].join(' / ')],
    ['稀疏权重 λS_I / λS_T', (status.lambda_sparse_i === null || status.lambda_sparse_i === undefined
      ? '暂无' : status.lambda_sparse_i) + ' / ' + (status.lambda_sparse_t === null
      || status.lambda_sparse_t === undefined ? '暂无' : status.lambda_sparse_t)],
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
    getJSON('/api/run/' + runId + '/text-nuisance')
      .then(renderTextNuisance)
      .catch(reportError);
    getJSON('/api/run/' + runId + '/clip512')
      .then(renderClip512)
      .catch(reportError);
    getJSON('/api/run/' + runId + '/pgclip')
      .then(renderPgClip)
      .catch(reportError);
    getJSON('/api/run/' + runId + '/cgclip')
      .then(renderCgClip)
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
    + fmt((shuffledAggregate.L3_I2T || {})['R@1_mean'], 4)
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
            fmt(q3['R@1'], 4), fmt(qcap['R@1'], 4), fmtSigned(paired['delta_R@1'], 4),
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

/* ---------------------------------------------------------------- G: text nuisance
 *
 * Reads one offline file (diagnostics/text_nuisance/clip_text_nuisance_ui.json). The page never
 * runs a forward pass: the numbers were produced by tools/diag/clip_text_nuisance_probe.py and the
 * endpoint only serves that file. Everything is labelled by dimension index -- no coordinate is
 * ever given a semantic name.
 */

const NUISANCE_VARIANTS = [
  ['base', '原句 C'], ['R1', 'C + R1'], ['R2', 'C + R2'], ['R3', 'C + R3'], ['R4', 'C + R4'],
  ['paraphrase', '改述'], ['visual_change', '视觉要素改变'],
];

function drawDimensionStrip(canvas, values, options) {
  const opts = options || {};
  const height = Number(canvas.getAttribute('height')) || 120;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!values || !values.length) {
    return;
  }
  const peak = opts.maxAbs || Math.max.apply(null, values.map(v => Math.abs(v)).concat([1e-6]));
  const slot = width / values.length;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const ratio = peak ? Math.abs(value) / peak : 0;
    if (opts.diverging) {
      const shade = value >= 0 ? '87,217,163' : '255,107,107';
      ctx.fillStyle = 'rgba(' + shade + ',' + (0.12 + 0.88 * ratio).toFixed(3) + ')';
    } else {
      ctx.fillStyle = 'rgba(77,163,255,' + (0.12 + 0.88 * ratio).toFixed(3) + ')';
    }
    ctx.fillRect(index * slot, 0, Math.max(1, slot - 0.4), height);
  }
  canvas.dataset.values = JSON.stringify(values);
  canvas.dataset.peak = String(peak);
}

function wireDimensionHover(canvas, label) {
  if (canvas.dataset.hoverWired === '1') {
    return;
  }
  canvas.dataset.hoverWired = '1';
  canvas.addEventListener('mousemove', event => {
    const values = JSON.parse(canvas.dataset.values || '[]');
    if (!values.length) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const index = Math.max(0, Math.min(values.length - 1,
      Math.floor((event.clientX - rect.left) / (rect.width / values.length))));
    setText(qs('nuisance-hover'), label + ' · 维度 ' + index + ' = ' + Number(values[index]).toFixed(5)
      + '（共 ' + values.length + ' 维，峰值 ' + Number(canvas.dataset.peak).toFixed(5) + '）');
  });
}

function renderTextNuisance(payload) {
  const banner = qs('nuisance-banner');
  setText(qs('nuisance-reminders'), (payload.reminders || []).join('   '));
  state.nuisance = payload;
  const modelSelect = qs('nuisance-model');
  const readoutSelect = qs('nuisance-readout');
  const baseSelect = qs('nuisance-base');
  const variantSelect = qs('nuisance-variant');

  if (!payload.available) {
    banner.textContent = '未运行：' + (payload.error || ('未找到 ' + payload.directory + '/' + payload.file))
      + '。运行 tools/diag/clip_text_nuisance_probe.py 后本区域才会出现数字。';
    ['nuisance-scope', 'nuisance-hand', 'nuisance-coordinates', 'nuisance-retrieval',
     'nuisance-not-run'].forEach(id => { qs(id).textContent = ''; });
    ['canvas-nuisance-vector', 'canvas-nuisance-delta', 'canvas-nuisance-zoom'].forEach(id => {
      const canvas = qs(id);
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      canvas.dataset.values = '[]';
    });
    modelSelect.textContent = ''; readoutSelect.textContent = ''; baseSelect.textContent = '';
    variantSelect.textContent = '';
    return;
  }

  banner.textContent = '已运行（只读离线）· 文件 ' + payload.file + ' · 写入时间 '
    + (payload.updated_at_iso || '暂无') + ' · 新增 optimizer updates = '
    + payload.new_optimizer_updates + '（必须为 0）· 128×128 只读诊断，不参与晋级门';

  const coordinates = payload.coordinates || {};
  const labels = payload.model_labels || {};
  const modelKeys = Object.keys(coordinates);
  if (modelSelect.dataset.filled !== modelKeys.join(',')) {
    modelSelect.textContent = '';
    modelKeys.forEach(key => {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = labels[key] || key;
      modelSelect.appendChild(option);
    });
    if (modelSelect.dataset.lastRun !== payload.run_id) {
      modelSelect.dataset.lastRun = payload.run_id;
      const preferred = modelKeys.indexOf('hs_500');
      modelSelect.selectedIndex = preferred >= 0 ? preferred : 0;
    }
    modelSelect.dataset.filled = modelKeys.join(',');
  }
  const model = modelSelect.value || modelKeys[0];
  const readouts = Object.keys(coordinates[model] || {});
  if (readoutSelect.dataset.filled !== readouts.join(',')) {
    readoutSelect.textContent = '';
    readouts.forEach(key => {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = key === 'NATIVE' ? '原生文本读出' : '文本 masked 读出（z = normalize(t_raw · mT)）';
      readoutSelect.appendChild(option);
    });
    readoutSelect.dataset.filled = readouts.join(',');
  }
  const readout = readoutSelect.value || readouts[0] || 'NATIVE';
  const handLabels = (coordinates[model] || {})[readout] ?
    ((coordinates[model][readout] || {}).hand_labels || {}) : {};

  const variantEntry = NUISANCE_VARIANTS.filter(item => item[0] !== 'base');
  if (variantSelect.dataset.filled !== '1') {
    variantSelect.textContent = '';
    variantEntry.forEach(item => {
      const option = document.createElement('option');
      option.value = item[0];
      option.textContent = item[1];
      variantSelect.appendChild(option);
    });
    variantSelect.dataset.filled = '1';
  }
  const variant = variantSelect.value || 'R1';

  const vectors = (coordinates[model] && coordinates[model][readout] &&
    coordinates[model][readout].hand_vectors) || {};
  const handGroups = (payload.hand_groups || {})[model] || {};
  const group = handGroups[readout] || {};
  const groupRows = [['附加句（4 条）', group.appended_suffix], ['改述', group.paraphrase],
                     ['视觉要素改变', group.visual_change]];
  tableBlock(qs('nuisance-hand'),
             ['对照', 'L2 中位', 'L2 中位以外的分位', 'cos 中位', 'cos 范围'],
             groupRows.map(row => {
               const body = row[1] || {};
               const l2 = body.l2 || {};
               const cos = body.cos || {};
               return [row[0],
                       l2['q0.5'] === undefined ? '暂无' : Number(l2['q0.5']).toFixed(4),
                       l2['q0.05'] === undefined ? '暂无'
                         : Number(l2['q0.05']).toFixed(3) + ' ~ ' + Number(l2['q0.95']).toFixed(3),
                       cos['q0.5'] === undefined ? '暂无' : Number(cos['q0.5']).toFixed(4),
                       cos['q0'] === undefined ? '暂无'
                         : Number(cos['q0']).toFixed(3) + ' ~ ' + Number(cos['q1']).toFixed(3)];
             }));
  const likeVsDislike = (group.length_matched_pairs || {})['R1_vs_R2'];
  if (likeVsDislike) {
    const footnote = document.createElement('p');
    footnote.className = 'muted';
    footnote.textContent = '等长对照 R1（like）vs R2（dislike）：长度差 '
      + likeVsDislike.length_delta + '，cos 中位 '
      + Number((likeVsDislike.cos || {})['q0.5']).toFixed(4)
      + '。长度近似只排除长度因素，不排除位置/上下文因素。';
    qs('nuisance-hand').appendChild(footnote);
  }

  // 512-d heatmaps for the selected base caption
  const baseKeys = Object.keys(vectors).sort((a, b) => Number(a) - Number(b));
  if (baseSelect.dataset.filled !== baseKeys.join(',')) {
    baseSelect.textContent = '';
    baseKeys.forEach(key => {
      const option = document.createElement('option');
      option.value = key;
      const text = (handLabels.base || [])[Number(key)] || '';
      option.textContent = Number(key) + ' · ' + (text.length > 42 ? text.slice(0, 42) + '…' : text);
      baseSelect.appendChild(option);
    });
    baseSelect.dataset.filled = baseKeys.join(',');
    baseSelect.selectedIndex = 0;
  }
  const baseKey = baseSelect.value || baseKeys[0];
  const baseValues = baseKey ? (vectors[baseKey] || {}).base : null;
  const variantValues = baseKey ? (vectors[baseKey] || {})[variant] : null;
  const peak = baseValues ? Math.max.apply(null, baseValues.map(v => Math.abs(v))) : 1;
  drawDimensionStrip(qs('canvas-nuisance-vector'), baseValues, { maxAbs: peak });
  wireDimensionHover(qs('canvas-nuisance-vector'), '原句 C 的 512 维');
  const delta = (baseValues && variantValues)
    ? variantValues.map((value, index) => value - baseValues[index]) : null;
  drawDimensionStrip(qs('canvas-nuisance-delta'), delta, { diverging: true });
  wireDimensionHover(qs('canvas-nuisance-delta'),
                     'delta（' + variant + ' − base）的 512 维');

  // top-32 zoom: coordinates with the largest delta energy over the whole real pool
  const energy = ((coordinates[model] || {})[readout] || {}).pool_delta_energy || [];
  const order = energy.map((value, index) => [index, value])
    .sort((a, b) => b[1] - a[1]).slice(0, 32).map(item => item[0]);
  if (delta && order.length) {
    drawDimensionStrip(qs('canvas-nuisance-zoom'), order.map(index => delta[index]),
                       { diverging: true });
    setText(qs('nuisance-zoom-label'),
            '放大维度（按真实池 delta 能量排序的前 32 个索引）：' + order.join(', '));
  }

  const summary = ((payload.coordinate_summary || {})[model] || {})[readout] || {};
  const topk = summary.topk_share || {};
  const split = summary.split_half || {};
  const svd = summary.svd || {};
  const visual = summary.hand_visual_change || {};
  tableBlock(qs('nuisance-coordinates'), ['量', '数值'], [
    ['top1 / top4 / top8 能量占比',
      ['top1_share', 'top4_share', 'top8_share']
        .map(key => topk[key] === undefined ? '暂无'
          : (100 * topk[key]).toFixed(2) + '%').join(' / ')],
    ['top16 / top32 / top64 能量占比',
      ['top16_share', 'top32_share', 'top64_share']
        .map(key => topk[key] === undefined ? '暂无'
          : (100 * topk[key]).toFixed(2) + '%').join(' / ')],
    ['折半（前半定序 → 后半计算）top1 / top32',
      [(split.second_half_topk_share_using_first_half_order || {}).top1_share,
       (split.second_half_topk_share_using_first_half_order || {}).top32_share]
        .map(value => value === undefined ? '暂无' : (100 * value).toFixed(2) + '%').join(' / ')],
    ['两半 top1 / top32 / top64 Jaccard',
      ['top1_jaccard', 'top32_jaccard', 'top64_jaccard']
        .map(key => (split.topk_jaccard_between_halves || {})[key] === undefined ? '暂无'
          : Number(split.topk_jaccard_between_halves[key]).toFixed(3)).join(' / ')],
    ['两半能量轮廓相关', split.energy_profile_correlation === undefined ? '暂无'
      : Number(split.energy_profile_correlation).toFixed(4)],
    ['未中心化 SVD：pc1 / pc4 / pc8 / pc16',
      ['pc1', 'pc4', 'pc8', 'pc16'].map(key => (svd.component_share || {})[key] === undefined
        ? '暂无' : (100 * svd.component_share[key]).toFixed(1) + '%').join(' / ')],
    ['平均 delta 能量占比（共同偏移）', svd.mean_delta_energy_share === undefined ? '暂无'
      : (100 * svd.mean_delta_energy_share).toFixed(1) + '%'],
    ['手写视觉改变 top32 与 R 敏感 top32 的重合',
      (visual.overlap_with_R_sensitive_top32 || {}).jaccard === undefined ? '暂无'
        : 'Jaccard ' + Number(visual.overlap_with_R_sensitive_top32.jaccard).toFixed(3)],
  ]);

  const poolMetrics = ((payload.pool_metrics || {})[model]) || {};
  const vsBase = ((payload.pool_vs_base || {})[model]) || {};
  const retrievalRows = [];
  Object.keys(poolMetrics).forEach(readoutKey => {
    const conditions = poolMetrics[readoutKey] || {};
    const base = conditions.BASE || {};
    const baseVs = (vsBase[readoutKey] || {}) || {};
    retrievalRows.push([readoutKey, 'BASE（干净）', fmt(base['R@1'], 4), fmt(base.ce, 4),
                        fmt(base.mrr, 4), '—', '—', '—', '—']);
    ['R1', 'R2', 'R3', 'R4', 'REPEAT'].forEach(condition => {
      const metrics = conditions[condition] || {};
      const delta = baseVs[condition] || {};
      const sign = delta.paired_sign_test || {};
      retrievalRows.push([
        readoutKey, condition === 'REPEAT' ? '重复一次（对照）' : condition,
        fmt(metrics['R@1'], 4), fmt(metrics.ce, 4), fmt(metrics.mrr, 4),
        fmtPoints(delta['delta_R@1']), fmtSigned(delta.delta_ce, 4),
        (sign.worse === undefined ? '—' : sign.worse + ' / ' + sign.better),
        (sign.p_value_two_sided === undefined ? '—' : Number(sign.p_value_two_sided).toFixed(3))]);
    });
  });
  tableBlock(qs('nuisance-retrieval'),
             ['读出', '条件', 'R@1', 'CE', 'MRR', 'ΔR@1(pp)', 'ΔCE', '变差/变好查询数', '符号检验 p'],
             retrievalRows);
  setText(qs('nuisance-not-run'), 'NOT RUN：' + (payload.not_run || []).join('；'));
}

/* ---------------------------------------------------------------- H: 512-d functional analysis
 *
 * Reads one offline JSON (diagnostics/clip512_functional_probe/clip512_functional_probe.json) and,
 * for phase B, the contact-sheet PNG of the selected scene through its own endpoint. The page runs
 * no forward pass, imports no model and reads no checkpoint. Every coordinate is named by index
 * only, and a number is shown only when the probe measured it.
 */

const C512_VIEWS = ['I_AB', 'I_A', 'I_B', 'I_control'];
const C512_VIEW_LABELS = {
  I_AB: 'I_AB · 完整图',
  I_A: 'I_A · 只留 A（删未述 B）',
  I_B: 'I_B · 只留 B（删已述 A）',
  I_control: 'I_control · 同尺寸控制裁剪',
};
const C512_RETENTION_KEY = {
  I_AB: 'retained_in_AB',
  I_A: 'retained_in_A_crop',
  I_B: 'retained_in_B_crop',
  I_control: 'retained_in_control',
};
/* the four noisy-suffix conditions and the repeatability condition, as the probe names them */
const C512_CONDITION_NOTE = 'BASE 为原始 caption；R1–R4 为在 caption 后追加一句与图像内容无关的'
  + '非视觉句子；REPEAT 为同一输入的重复前向（用于报重复误差，不是第五种条件）。';

function c512Mean(values) {
  const clean = (values || []).filter(value => typeof value === 'number' && Number.isFinite(value));
  if (!clean.length) return null;
  return clean.reduce((total, value) => total + value, 0) / clean.length;
}

function c512AbsMean(values) {
  return c512Mean((values || []).map(value => Math.abs(value)));
}

function c512Range(values) {
  const clean = (values || []).filter(value => typeof value === 'number' && Number.isFinite(value));
  if (!clean.length) return null;
  return [Math.min.apply(null, clean), Math.max.apply(null, clean)];
}

function c512Count(flags) {
  return (flags || []).filter(Boolean).length;
}

function c512Num(value, digits) {
  return typeof value === 'number' && Number.isFinite(value) ? fmt(value, digits) : '暂无';
}

/* Pearson correlation over two per-coordinate columns; null when either side is constant */
function c512Pearson(left, right) {
  const pairs = [];
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const a = left[index], b = right[index];
    if (typeof a === 'number' && Number.isFinite(a) && typeof b === 'number' && Number.isFinite(b)) {
      pairs.push([a, b]);
    }
  }
  if (pairs.length < 3) return null;
  const meanA = c512Mean(pairs.map(pair => pair[0]));
  const meanB = c512Mean(pairs.map(pair => pair[1]));
  let cov = 0, varA = 0, varB = 0;
  pairs.forEach(pair => {
    cov += (pair[0] - meanA) * (pair[1] - meanB);
    varA += (pair[0] - meanA) ** 2;
    varB += (pair[1] - meanB) ** 2;
  });
  if (varA <= 0 || varB <= 0) return null;
  return cov / Math.sqrt(varA * varB);
}

/* the per-scene masked-selection statistics, recomputed from the exported per-scene records; every
 * definition is named next to its number, and the two magnitude conventions are never merged */
function c512SelectionAggregate(records) {
  const rows = records || [];
  const said = rows.map(row => ((row.smartclip_selection || {}).T_A_delete_said_A_distance));
  const unsaid = rows.map(row => ((row.smartclip_selection || {}).T_A_delete_unsaid_B_distance));
  const nativeSaid = rows.map(row => ((row.smartclip_selection || {}).native_T_A_delete_said_A));
  const nativeUnsaid = rows.map(row => ((row.smartclip_selection || {}).native_T_A_delete_unsaid_B));
  const kept = [];
  rows.forEach(row => {
    Object.keys(row.smartclip_masked || {}).forEach(name => {
      kept.push(row.smartclip_masked[name].mask_kept);
    });
  });
  return {
    said: said,
    unsaid: unsaid,
    saidSignedMean: c512Mean(said),
    unsaidSignedMean: c512Mean(unsaid),
    saidAbsMean: c512AbsMean(said),
    unsaidAbsMean: c512AbsMean(unsaid),
    signedOrderCount: c512Count(rows.map((row, index) => unsaid[index] > said[index])),
    absOrderCount: c512Count(rows.map((row, index) => Math.abs(unsaid[index]) < Math.abs(said[index]))),
    nativeSignedOrderCount: c512Count(rows.map((row, index) => nativeUnsaid[index] > nativeSaid[index])),
    nativeAbsOrderCount:
      c512Count(rows.map((row, index) => Math.abs(nativeUnsaid[index]) < Math.abs(nativeSaid[index]))),
    nativePreferredCount: c512Count(rows.map(row => row.A_preferred_on_A_view)),
    maskedPreferredCount:
      c512Count(rows.map(row => (row.smartclip_selection || {}).masked_prefers_A_view_for_T_A)),
    nativeBPreferredCount: c512Count(rows.map(row => row.B_preferred_on_B_view)),
    bothMarginsPositive: c512Count(rows.map(row => row.A_margin > 0 && row.B_margin > 0)),
    positiveKCount: c512Count(rows.map(row => row.four_score_identity.K > 0)),
    cosValues: rows.map(row => row.cos_dv_dt),
    identityMax: Math.max.apply(null, rows.map(row => row.four_score_identity.max_abs_diff)
      .concat([0])),
    maskKeptRange: c512Range(kept),
    sceneCount: rows.length,
  };
}

function renderClip512(payload) {
  state.clip512 = payload;
  const modelSelect = qs('c512-model');
  const sceneSelect = qs('c512-scene');
  const containers = ['c512-scope', 'c512-dimensions', 'c512-dimension-note', 'c512-capture',
    'c512-capture-note', 'c512-queries', 'c512-scene', 'c512-scores', 'c512-masked',
    'c512-scene-detail', 'c512-dataset', 'c512-per-condition', 'c512-paired', 'c512-summary'];
  if (!payload.available) {
    setText(qs('c512-banner'), '未运行：' + (payload.error || ('未找到 ' + payload.directory))
      + '（' + payload.directory + '）。运行 tools/diag/clip512_functional_probe.py 后本区域才会出现数字。');
    setText(qs('c512-reminders'), '本区域不显示任何推测值：文件不存在时只显示"未运行"。');
    containers.forEach(id => { qs(id).textContent = ''; });
    qs('c512-sheet').style.display = 'none';
    modelSelect.textContent = '';
    sceneSelect.textContent = '';
    setText(qs('c512-not-run'), '');
    return;
  }
  setText(qs('c512-reminders'), (payload.reminders || []).join('   '));

  const models = payload.models || {};
  const keys = Object.keys(models);
  if (modelSelect.dataset.filled !== keys.join(',') || modelSelect.dataset.run !== payload.run_id) {
    modelSelect.textContent = '';
    keys.forEach(key => {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = key + ' · ' + (models[key].label || '');
      modelSelect.appendChild(option);
    });
    modelSelect.dataset.filled = keys.join(',');
    modelSelect.dataset.run = payload.run_id;
    const preferred = keys.indexOf('s0_500');
    modelSelect.selectedIndex = preferred >= 0 ? preferred : 0;
  }
  const model = modelSelect.value || keys[0];
  const body = models[model] || {};
  const phaseB = payload.phase_b || {};
  const scenes = phaseB.scenes || [];
  const perScene = ((phaseB.models || {})[model] || {}).per_scene || [];

  setText(qs('c512-banner'),
    '已运行（只读离线）· ' + payload.file + ' · ' + (payload.updated_at_iso || '暂无')
    + ' · 新增 optimizer updates = ' + payload.new_optimizer_updates
    + ' · phase A ' + (payload.phase_a_status || '?') + ' / phase B ' + (payload.phase_b_status || '?')
    + ' · 128 图文池 + ' + scenes.length + ' 个裁剪场景，都不参与晋级门');

  const pool = payload.pool || {};
  const truncation = payload.truncation || {};
  const templateAudit = payload.template_audit || [];
  kvGrid(qs('c512-scope'), [
    ['模型标签', (payload.models_used || []).map(key => key + '=' + ((models[key] || {}).label || '?')).join('  |  ')],
    ['选中模型的编码量', c512Num(body.texts_encoded, 0) + ' 条文本 / ' + c512Num(body.views_encoded, 0) + ' 个视图'],
    ['dtype / autocast', (body.dtype || '?') + ' / ' + (body.autocast || '?')],
    ['参数摘要前后一致', body.parameter_state_unchanged === true ? '是' : '否'],
    ['重复前向最大绝对误差', c512Num((body.repeat_check || {}).forward_error_max_abs, 8)],
    ['run_status.json 未被改动', (payload.run_status || {}).unchanged === true ? '是（前后 SHA 相同）' : '否'],
    ['pool manifest SHA256', String(pool.manifest_sha256 || '').slice(0, 16) + '…（' + c512Num(pool.images, 0) + ' 图）'],
    ['改述审计', payload.paraphrase_audit_ok === true
      ? '通过（' + templateAudit.length + ' 组，颜色/数量/对象/关系不变）' : '未通过'],
    ['截断检查', '最大有效长度 ' + c512Num(truncation.max_effective_length, 0) + '（EOT 规则）；'
      + '未进入编码器的后缀 ' + ((truncation.suffix_not_entered || []).length) + ' 个'],
    ['探针耗时', c512Num((payload.timing || {}).wall_seconds, 1) + ' s（0 次 optimizer update）'],
  ]);

  /* ---------------------------------------------------------------- A */
  const chart = body.chart || {};
  const unified = (payload.unified_dimension_table || {})[model] || [];
  const byDimension = {};
  unified.forEach(row => { byDimension[row.dimension] = row; });
  const topMargin = chart.top32_abs_margin || [];
  const topEnergy = chart.top32_raw_energy || [];
  const absSum = (chart.abs_margin_contribution || []).reduce((total, value) => total + value, 0);
  const rows = topMargin.slice(0, 16).map((dimension, rank) => {
    const row = byDimension[dimension] || {};
    return [
      rank + 1, '#' + dimension,
      c512Num((chart.raw_delta_energy || [])[dimension], 6),
      c512Num((chart.abs_margin_contribution || [])[dimension], 4),
      c512Num((chart.signed_margin_contribution || [])[dimension], 4),
      c512Num(row.mask_keep_frequency, 3),
      c512Num(row.mean_gap_squared, 6),
      topEnergy.indexOf(dimension) >= 0 ? '是（第 ' + (topEnergy.indexOf(dimension) + 1) + ' 名）' : '否',
    ];
  });
  tableBlock(qs('c512-dimensions'),
    ['按 |margin| 排名', '坐标', '平均 ΔT²（后缀造成的文本变化量）', '平均 |margin 贡献|',
     '平均有符号 margin 贡献', 'mask 保留频率', 'mean_gap²', '是否也在变化量前 32'],
    rows.length ? rows : [['暂无']]);
  const topShare = absSum > 0 ? (chart.abs_margin_contribution || [])[topMargin[0]] / absSum : null;
  const keepColumn = unified.map(row => row.mask_keep_frequency);
  const marginColumn = unified.map(row => row.margin_contribution_abs_mean);
  const gapColumn = unified.map(row => row.mean_gap_squared);
  const maskMarginR = c512Pearson(keepColumn, marginColumn);
  const maskGapR = c512Pearson(keepColumn, gapColumn);
  setText(qs('c512-dimension-note'),
    '变化量前 32 与 |margin| 前 32 重合 ' + c512Num(chart.top32_overlap_at_32, 0)
    + '/32（Jaccard ' + c512Num(chart.top32_jaccard, 3) + '）—— 变化大不等于影响判别。'
    + ' 第 1 名 #' + topMargin[0] + ' 占全部 |margin| 贡献的 '
    + (topShare === null ? '暂无' : (topShare * 100).toFixed(1) + '%')
    + '（单点驱动，须与"典型坐标关系很弱"一起读）。'
    + ' mask 保留频率与 |margin| 的 Pearson 相关为 ' + c512Num(maskMarginR, 3)
    + '、与 gap² 为 ' + c512Num(maskGapR, 3)
    + '（本页按当前 512 行表就地计算）——mask 保留率高的坐标并不因此更影响判别。');

  const cvs = (body.coordinate_vs_subspace || {}).uncentered || {};
  const cvsCentered = (body.coordinate_vs_subspace || {}).centered || {};
  const captureRow = (label, block) => [
    label,
    c512Num((block.coordinate_topk_capture || {}).k4, 4),
    c512Num((block.coordinate_topk_capture || {}).k8, 4),
    c512Num((block.coordinate_topk_capture || {}).k16, 4),
    c512Num((block.subspace_topk_capture || {}).k4, 4),
    c512Num((block.subspace_topk_capture || {}).k8, 4),
    c512Num((block.subspace_topk_capture || {}).k16, 4),
  ];
  tableBlock(qs('c512-capture'),
    ['能量捕获（后半样本，同等秩预算）', '坐标 k=4', '坐标 k=8', '坐标 k=16',
     '方向 k=4', '方向 k=8', '方向 k=16'],
    [captureRow('未中心化', cvs), captureRow('中心化后', cvsCentered)]);
  setText(qs('c512-capture-note'),
    '同等秩预算下，k 个原始坐标捕获的能量远少于 k 个方向：'
    + '这只说明"变化不是沿少数原始坐标分布"，**不能**推出"换成子空间投影就会涨分"（本轮未训练投影、未做删方向后的检索对照）。'
    + ' 前半/后半样本量 ' + c512Num(cvs.n_pairs_first, 0) + ' / ' + c512Num(cvs.n_pairs_second, 0)
    + '，同秩预算 = ' + (cvs.same_rank_budget === true ? '是' : '否')
    + '；未中心化方向含共同偏移，中心化行给出对照。');

  const queryRow = (kind, item) => [
    kind, '#' + item.query_index, item.image_id, item.annotation_id, item.base_rank, item.r1_rank,
    (item.base_caption || item.suffixed_caption || '').slice(0, 40) || '—（探针只导出排名）',
    typeof item.base_m_lse === 'number'
      ? c512Num(item.base_m_lse, 3) + ' / ' + c512Num(item.r1_m_lse, 3) : '—',
  ];
  tableBlock(qs('c512-queries'),
    ['I2T 排名变化（R1 相对 BASE）', '查询下标', '图像 ID', '标注 ID', 'BASE 排名', 'R1 排名',
     'caption（截断）', 'BASE/R1 m_lse'],
    (chart.best_queries || []).map(item => queryRow('改善', item))
      .concat((chart.worst_queries || []).map(item => queryRow('变差', item))));
  setText(qs('c512-queries-note'),
    '只列改善/变差最大的各 5 个查询（128 池中）。它们是"非视觉后缀改变了排序"的具体样本，'
    + '不构成对后缀类型的普遍结论；完整逐查询数组在探针 JSON 的 per_condition 里。');

  /* ---------------------------------------------------------------- B */
  if (!scenes.length) {
    setText(qs('c512-scene'), '阶段B 未产出场景：' + (phaseB.reason || '未运行'));
    ['c512-scores', 'c512-masked', 'c512-scene-detail', 'c512-dataset'].forEach(id => {
      qs(id).textContent = '';
    });
    qs('c512-sheet').style.display = 'none';
  } else {
    if (sceneSelect.dataset.filled !== String(scenes.length) || sceneSelect.dataset.run !== payload.run_id) {
      sceneSelect.textContent = '';
      scenes.forEach((scene, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = index + ' · ' + scene.category_pair + '（图 ' + scene.image_id + '）';
        sceneSelect.appendChild(option);
      });
      sceneSelect.dataset.filled = String(scenes.length);
      sceneSelect.dataset.run = payload.run_id;
      sceneSelect.selectedIndex = 0;
    }
    const index = Number(sceneSelect.value || 0);
    const scene = scenes[index] || {};
    const record = perScene[index] || {};
    const scores = record.scores || {};
    const masked = record.smartclip_masked || {};
    const selection = record.smartclip_selection || {};

    tableBlock(qs('c512-scene'),
      ['视图', 'A（' + (record.A || '?') + '）框内占比', 'B（' + (record.B || '?') + '）框内占比'],
      C512_VIEWS.map(view => [
        C512_VIEW_LABELS[view],
        c512Num((scene.A || {})[C512_RETENTION_KEY[view]], 4),
        c512Num((scene.B || {})[C512_RETENTION_KEY[view]], 4),
      ]).concat([['A 分割点多边形保留（只留 A 的裁剪内）',
        c512Num((scene.A || {}).segmentation_point_retention_in_A_crop, 4), '—'],
      ['B 分割点多边形保留（只留 B 的裁剪内）', '—',
        c512Num((scene.B || {}).segmentation_point_retention_in_B_crop, 4)]]));

    tableBlock(qs('c512-scores'),
      ['原生读出（未加 mask）', 'T_A', 'T_AB', 'T_B'],
      C512_VIEWS.map(view => [
        C512_VIEW_LABELS[view], c512Num((scores[view] || {}).T_A, 3),
        c512Num((scores[view] || {}).T_AB, 3), c512Num((scores[view] || {}).T_B, 3),
      ]));

    const maskLabel = name => name + '（mask 保留 '
      + c512Num((masked[name] || {}).mask_kept, 0) + '/512，范数 '
      + c512Num((masked[name] || {}).mask_norm, 1) + '）';
    tableBlock(qs('c512-masked'),
      ['SmartCLIP masked 读出（同一文本的同一个 mask 用于四个视图）',
       maskLabel('T_A'), maskLabel('T_AB'), maskLabel('T_B')],
      C512_VIEWS.map(view => [
        C512_VIEW_LABELS[view],
        c512Num((((masked.T_A || {}).views || {})[view] || {}).score, 3),
        c512Num((((masked.T_AB || {}).views || {})[view] || {}).score, 3),
        c512Num((((masked.T_B || {}).views || {})[view] || {}).score, 3),
      ]));

    const kIdentity = record.four_score_identity || {};
    kvGrid(qs('c512-scene-detail'), [
      ['K = 100·dv·dt（四分数恒等式）', c512Num(kIdentity.K, 4)
        + ' vs ' + c512Num(kIdentity.K_from_four_scores, 4)
        + '（差 ' + c512Num(kIdentity.max_abs_diff, 8) + '）'],
      ['cos(dv, dt)', c512Num(record.cos_dv_dt, 3)],
      ['A / B 视图上的 margin', c512Num(record.A_margin, 3) + ' / ' + c512Num(record.B_margin, 3)],
      ['A 视图偏好 A / B 视图偏好 B', String(record.A_preferred_on_A_view) + ' / '
        + String(record.B_preferred_on_B_view)],
      ['K 的逐维正/负能量', c512Num(record.K_dim_positive_energy, 2) + ' / '
        + c512Num(record.K_dim_negative_energy, 2)],
      ['|K_d| 最大的 8 个坐标', (record.K_top_coordinates || []).map(dim => '#' + dim).join(' ')],
      ['masked 偏好 A 视图（T_A）/ 原生', String(selection.masked_prefers_A_view_for_T_A) + ' / '
        + String(selection.native_prefers_A_view_for_T_A)],
      ['Δscore：删未述 B / 删已述 A（masked）', fmtSigned(selection.T_A_delete_unsaid_B_distance, 3)
        + ' / ' + fmtSigned(selection.T_A_delete_said_A_distance, 3)],
      ['Δscore：删未述 B / 删已述 A（原生）', fmtSigned(selection.native_T_A_delete_unsaid_B, 3)
        + ' / ' + fmtSigned(selection.native_T_A_delete_said_A, 3)],
      ['Δscore：同尺寸控制裁剪（masked）', fmtSigned(selection.T_A_control_distance, 3)],
      ['四视图裁剪框（x0, y0, x1, y1）', JSON.stringify(scene.crops || {})],
      ['官方视图映射核对', (scene.official_view || {}).analytic_mapping_matches_pipeline === true
        ? '一致（缩放 ' + c512Num((scene.official_view || {}).scale, 4) + '，偏移 '
          + JSON.stringify((scene.official_view || {}).center_crop_offset) + '）' : '不一致'],
      ['隔离级别', String(scene.isolation || '?') + '（框级；分割点保留率见上表，非像素级纯净）'],
    ]);

    const aggregate = c512SelectionAggregate(perScene);
    tableBlock(qs('c512-dataset'),
      ['当前模型的 5 场景汇总（每个数字都写明口径）', '结果'],
      [
        ['场景数', c512Num(aggregate.sceneCount, 0)],
        ['K > 0 的场景', aggregate.positiveKCount + ' / ' + aggregate.sceneCount],
        ['四分数恒等式最大绝对差', c512Num(aggregate.identityMax, 8)],
        ['cos(dv,dt) 均值（范围）', c512Num(c512Mean(aggregate.cosValues), 3) + '（'
          + (c512Range(aggregate.cosValues) || []).map(value => value.toFixed(2)).join(' ~ ') + '）'],
        ['masked / 原生 偏好 A 视图（T_A）', aggregate.maskedPreferredCount + ' / '
          + aggregate.nativePreferredCount + ' / ' + aggregate.sceneCount],
        ['B 视图偏好 B', aggregate.nativeBPreferredCount + ' / ' + aggregate.sceneCount],
        ['A、B 两项 margin 同时为正', aggregate.bothMarginsPositive + ' / ' + aggregate.sceneCount],
        ['Δscore 有符号均值：删未述 / 删已述', fmtSigned(aggregate.unsaidSignedMean, 2) + ' / '
          + fmtSigned(aggregate.saidSignedMean, 2)],
        ['Δscore 绝对均值：删未述 / 删已述', c512Num(aggregate.unsaidAbsMean, 2) + ' / '
          + c512Num(aggregate.saidAbsMean, 2)],
        ['"删未述更稳"的场景数：有符号口径 / 绝对口径',
          aggregate.signedOrderCount + ' / ' + aggregate.absOrderCount + ' / ' + aggregate.sceneCount],
        ['同上的原生读出：有符号 / 绝对', aggregate.nativeSignedOrderCount + ' / '
          + aggregate.nativeAbsOrderCount + ' / ' + aggregate.sceneCount],
        ['mask 保留坐标数（三模板 × 场景，范围）',
          (aggregate.maskKeptRange || []).map(value => value.toFixed(0)).join(' ~ ') + ' / 512'],
      ]);
    const sheet = qs('c512-sheet');
    sheet.src = '/api/run/' + encodeURIComponent(payload.run_id) + '/clip512-sheet?scene=' + index;
    sheet.style.display = '';
  }

  /* ---------------------------------------------------------------- C */
  const perCondition = body.per_condition || {};
  const conditions = Object.keys(perCondition);
  tableBlock(qs('c512-per-condition'),
    ['条件', 'I2T R@1', 'I2T R@5', 'I2T R@10', 'T2I R@1', 'T2I R@5', 'T2I R@10',
     'I2T ce', 'I2T entropy', 'I2T mrr'],
    conditions.map(name => {
      const i2t = perCondition[name].I2T || {};
      const t2i = perCondition[name].T2I || {};
      return [name, c512Num(i2t['R@1'], 4), c512Num(i2t['R@5'], 4), c512Num(i2t['R@10'], 4),
        c512Num(t2i['R@1'], 4), c512Num(t2i['R@5'], 4), c512Num(t2i['R@10'], 4),
        c512Num(i2t.ce, 4), c512Num(i2t.entropy, 4), c512Num(i2t.mrr, 4)];
    }));
  setText(qs('c512-queries-note'), '只列改善/变差最大的各 5 个查询（128 池中）。它们是"非视觉后缀改变了排序"的具体样本，'
    + '不构成对后缀类型的普遍结论；完整逐查询数组在探针 JSON 的 per_condition 里。');
  const paired = body.paired_outcome || {};
  const pairedConditions = Object.keys(paired);
  tableBlock(qs('c512-paired'),
    ['与 BASE 配对的离散结果', '方向', 'R@1 命中变化', '命中增加', '命中丢失',
     '排名改善', '排名不变', '排名变差', '最大排名恶化'],
    pairedConditions.reduce((accumulator, name) => {
      ['I2T', 'T2I'].forEach(direction => {
        const block = paired[name][direction] || {};
        accumulator.push([name, direction, c512Num(block['R@1_hit_count_change'], 0),
          c512Num(block['R@1_hits_gained'], 0), c512Num(block['R@1_hits_lost'], 0),
          c512Num(block.rank_improved, 0), c512Num(block.rank_unchanged, 0),
          c512Num(block.rank_worsened, 0), c512Num(block.max_rank_worsening, 0)]);
      });
      return accumulator;
    }, []));

  const distribution = body.distribution || {};
  const scale = body.scale_control || {};
  const deltaQ = body.deltaQ_identity || {};
  const margin = body.margin_contribution || {};
  const covariance = body.candidate_covariance || {};
  const offset = body.common_offset || {};
  const identityConditions = Object.keys(deltaQ);
  /* every row names its own conditions: the probe exports REPEAT for some identities and not for
   * others, and a missing condition must not be printed as 暂无 next to a measured one */
  const conditionRow = (label, block, pick, digits) => [label, Object.keys(block)
    .map(name => name + ':' + c512Num(pick(block[name] || {}), digits)).join('  ')];
  tableBlock(qs('c512-summary'),
    ['恒等式 / 控制量', '数值'],
    [
      conditionRow('ΔQ（ΔT 造成的逐查询分差）恒等式最大绝对差', deltaQ,
        block => block.Q_R_minus_Q_max_abs_diff, 8),
      conditionRow('固定负例下的 margin 归因恒等式最大绝对差', margin,
        block => block.identity_error_max_abs, 8),
      conditionRow('候选池二次型（协方差）恒等式最大绝对差', covariance,
        block => block.max_abs_diff, 12),
      conditionRow('最强负例被切换的查询数', margin, block => block.negative_switched_queries, 0),
      conditionRow('I2T 行共同偏移下逐查询排名不变', offset,
        block => String(block.per_query_rank_identical)),
      conditionRow('I2T 行共同偏移的常数均值 / 范数', offset,
        block => c512Num(block.row_constant_mean, 3) + ' / ' + c512Num(block.mu_norm, 3)),
      conditionRow('尺度控制：Q×0.5 / ×1 / ×2 排名不变', scale,
        block => String(block.ranking_identical_to_x1)),
      conditionRow('尺度控制 I2T ce', scale, block => block.ce, 4),
      conditionRow('尺度控制 I2T entropy', scale, block => block.entropy, 4),
      ['mean/centroid gap（简单均值间隙）', c512Num(distribution.centroid_gap, 4)],
      ['paired alignment', c512Num(distribution.paired_alignment, 4)],
      ['image / text uniformity',
        c512Num(distribution.image_uniformity, 4) + ' / ' + c512Num(distribution.text_uniformity, 4)],
      ['阶段B 场景数 / 拒绝原因',
        scenes.length + ' / ' + Object.keys(phaseB.rejection_reason_counts || {})
          .map(reason => reason + '=' + phaseB.rejection_reason_counts[reason]).join('，')],
    ]);
  const identityNote = (Object.keys(offset).length
    ? ((offset[Object.keys(offset)[0]] || {}).note || '') : '');
  setText(qs('c512-not-run'), '口径说明：' + C512_CONDITION_NOTE
    + ' 共同偏移不变性只对 I2T 成立（' + identityNote + '）。'
    + ' 集合指标（uniformity、gap、alignment）是样本集合统计量，不是任何单一坐标的语义标签。'
    + ' ' + ((payload.sources || {}).note || '')
    + ' NOT RUN：' + (payload.not_run || []).join('；'));
}

/* ---------------------------------------------------------------- I: PG-CLIP v0.1
 *
 * Reads one read-only JSON (`/api/run/<id>/pgclip`) built from the run's own small files: the
 * trainer configuration, the scalar log, the status file, the frozen evaluation rows and -- when the
 * post-hoc snapshot exists -- per-coordinate mask values. The page runs no forward pass, imports no
 * model and reads no checkpoint, and it never plots a field the log does not contain.
 */

const PG_SERIES_LABELS = {
  loss_global: 'LG = 原生对齐（I2T + T2I）',
  loss_preproj: 'LP = 投影前条件对齐（I2T + T2I）',
  loss_sparse: 'LS = mean|mask|',
  loss_total: 'L_total = 5·LG + 5·LP + LS',
  weighted_loss_global: '5·LG',
  weighted_loss_preproj: '5·LP',
  path_global_i2t_top1: '原生 I2T top1',
  path_preproj_i2t_top1: '投影前条件 I2T top1',
  path_global_t2i_top1: '原生 T2I top1',
  path_preproj_t2i_top1: '投影前条件 T2I top1',
  path_global_i2t_max_margin_mean: '原生 I2T max margin 均值',
  path_preproj_i2t_max_margin_mean: '条件 I2T max margin 均值',
  mask_kept_mean: 'mask 平均保留坐标数（/768）',
  mask_all_on_fraction: '全开 caption 比例',
  gate_probability_mean: 'gate 概率均值',
  gate_probability_min: 'gate 概率最小值',
  preproj_retained_energy_mean: 'preproj_retained_energy（投影前能量保留）',
  projected_output_energy_ratio_mean: 'projected_output_energy_ratio（可 > 1）',
  projected_output_energy_ratio_max: 'projected ratio 最大值',
  sec_per_step: '每步秒数',
  lr: 'CLIP 学习率',
  gate_lr: 'gate 学习率',
};

function pgValue(value) {
  return (value === null || value === undefined || value === '') ? '暂无' : String(value);
}

function pgTable(container, headers, rows) {
  return tableBlock(container, headers, rows.length ? rows : [['暂无']]);
}

/* a 768-cell grid whose layout encodes nothing but the coordinate index */
function renderPgMaskGrid(container, payload) {
  container.textContent = '';
  const mask = payload.mask || {};
  if (!mask.available || !mask.mask_groups) {
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent = mask.not_run || ('未运行：' + (mask.error || '没有逐坐标 mask 快照文件'));
    container.appendChild(note);
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'mask-groups';
  (mask.mask_groups || []).forEach(group => {
    const caption = document.createElement('div');
    caption.className = 'mask-caption';
    const label = document.createElement('div');
    label.className = 'k';
    label.textContent = (group.label || ('caption ' + group.index)) + '（保留 '
      + (group.kept === undefined ? '?' : group.kept) + '/768）';
    caption.appendChild(label);
    const row = document.createElement('div');
    row.className = 'cells';
    (group.mask || []).forEach((value, coordinate) => {
      const cell = document.createElement('span');
      cell.className = 'cell ' + (value >= 0.5 ? 'on' : 'off');
      cell.title = '#' + coordinate + ' = ' + value;
      row.appendChild(cell);
    });
    caption.appendChild(row);
    grid.appendChild(caption);
  });
  container.appendChild(grid);
}

/* ---------------------------------------------- I.2: auxiliary-branch retrieval (diagnostic)
 *
 * The frozen promotion protocol never uses the gate. This block shows the second, explicitly
 * diagnostic protocol: retrieval scored with the pre-projection conditional path
 *   QP[i,j] = 100 * <Norm((h_i * mask_j) @ W), t_j>
 * where every candidate caption brings its own gate mask. The numbers come from a small read-only
 * JSON written by tools/diag/pgclip_aux_retrieval.py next to the run; the page never runs a forward
 * pass and never treats these numbers as a gate candidate.
 */

const PG_AUX_METRICS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10'];

function pgAuxDeltaPoints(value, reference) {
  return (typeof value === 'number' && typeof reference === 'number')
    ? ((value - reference) * 100).toFixed(2) + ' pp' : '暂无';
}

function renderPgAuxiliary(payload, containerId, noteId) {
  const container = qs(containerId);
  const block = payload.auxiliary_retrieval || {};
  if (!block.available) {
    pgTable(container, ['辅助分支图文检索（诊断，不参与冻结门）', '状态'],
            [['未运行', block.not_run || '没有 pgclip_aux_retrieval.json']]);
    setText(qs(noteId), block.not_run || '');
    return;
  }
  const protocol = block.protocol || {};
  const rows = [];
  Object.keys(block.results || {}).forEach(name => {
    const entry = block.results[name] || {};
    const native = entry.native || {};
    const auxiliary = entry.auxiliary || {};
    rows.push([name + '：池 ' + entry.images + ' 图 × ' + entry.texts + ' 文本', '原生', '辅助',
               '差值', '']);
    PG_AUX_METRICS.forEach(metric => {
      rows.push(['　' + metric.toUpperCase().replace('_', ' '), fmt(native[metric], 4),
                 fmt(auxiliary[metric], 4),
                 pgAuxDeltaPoints(auxiliary[metric], native[metric]), '']);
    });
    const paired = ((entry.paired_native_vs_auxiliary) || {}).I2T || {};
    rows.push(['　I2T 配对：命中 +' + pgValue(paired.hits_gained_r1) + ' / −'
               + pgValue(paired.hits_lost_r1) + '，排名改善 ' + pgValue(paired.rank_improved)
               + ' / 不变 ' + pgValue(paired.rank_unchanged) + ' / 变差 '
               + pgValue(paired.rank_worsened), '', '', '',
               '最大变差 ' + pgValue(paired.max_rank_worsening)]);
    const stats = entry.mask_statistics_over_evaluated_texts || {};
    rows.push(['　被测文本的 mask 平均保留坐标数（/768）', fmt(stats.mask_kept_mean, 1), '', '',
               '全开比例 ' + fmt(stats.mask_all_on_fraction, 4)]);
  });
  pgTable(container, ['辅助分支图文检索（诊断，不参与冻结门）', '原生（CLS/EOS）',
                      '辅助（投影前条件）', '差值', '备注'], rows);
  setText(qs(noteId),
    '口径：' + pgValue(protocol.native) + '　vs　' + pgValue(protocol.auxiliary)
    + '；mask 来源 ' + pgValue(protocol.mask_source)
    + '；判定规则 ' + pgValue(protocol.tie_rule)
    + '；精度 ' + pgValue(protocol.precision)
    + (protocol.subset ? '；**本轮用的是诊断子集（limit_images=' + pgValue(protocol.limit_images)
      + '），不是冻结协议的全池**' : '；池与冻结协议一致')
    + '。checkpoint ' + String(block.checkpoint_sha256 || '').slice(0, 12)
    + '（第 ' + pgValue(block.completed_steps) + ' 步，新增 optimizer update = '
    + pgValue(block.new_optimizer_updates) + '）。'
    + '这两个数只是"把辅助分支当读出用"的检索效果，**不是**晋级门指标，也不代表两路融合的上限。');
}

function renderPgClip(payload) {
  state.pgclip = payload;
  const ids = ['pg-scope', 'pg-paths', 'pg-curves', 'pg-curve-note', 'pg-mask-stats',
    'pg-mask-grid', 'pg-mask-note', 'pg-energy', 'pg-progress', 'pg-eval'];
  if (!payload.available) {
    setText(qs('pg-banner'), '未运行：这个 run 不是 PG-CLIP（objective=' + pgValue(payload.objective)
      + '）。只有 objective = clip_native_preproj_mask 的 run 才会在这里显示两路损失。');
    setText(qs('pg-reminders'), '本区域不显示推测值：objective 不匹配时只显示"未运行"。');
    ids.forEach(id => { qs(id).textContent = ''; });
    ['pg-aux', 'pg-aux-note'].forEach(id => { qs(id).textContent = ''; });
    setText(qs('pg-conclusion'), '');
    return;
  }
  setText(qs('pg-reminders'), (payload.reminders || []).join('   '));
  const config = payload.config || {};
  const progress = payload.progress || {};
  const weights = payload.loss_weights || {};
  setText(qs('pg-banner'),
    '已运行（只读）· objective ' + config.objective + ' · arm ' + config.arm
    + ' · ' + pgValue(config.gate_mode) + ' · 当前阶段 ' + pgValue(progress.path)
    + ' · 实现 SHA ' + String(progress.implementation_sha || '').slice(0, 12)
    + ' · 本页不加载 checkpoint、不触发 GPU 前向');

  kvGrid(qs('pg-scope'), [
    ['两路定义', 'native: Norm(h @ W)；preproj: Norm((h * mask_j) @ W)，W = clip.visual.proj 共享'],
    ['h 取点', config.h_source],
    ['文本侧', config.text_source],
    ['gate', ((config.gate || {}).stem || '?') + ' → ' + ((config.gate || {}).output || '?')
      + '（bias = log 8，输出 weight = 0；' + pgValue((config.gate || {}).mode) + '）'],
    ['损失组合', config.loss_combination],
    ['梯度职责', config.grader],
    ['候选规则', config.candidate_rule],
    ['精度', config.precision],
    ['分块', 'image_chunk=' + pgValue((config.chunking || {}).image_chunk)
      + '，text_chunk=' + pgValue((config.chunking || {}).text_chunk)
      + '，qp_checkpoint=' + pgValue((config.chunking || {}).qp_checkpoint)],
    ['DDP 路线', pgValue(config.ddp_route) + '（world_size 倍率：'
      + (config.no_world_size_factor ? '无' : '有') + '）'],
    ['数据', pgValue(config.view) + '；' + pgValue(config.caption_stream)],
    ['初始化文件 SHA256', String(config.init_file_sha256 || '').slice(0, 32) + '…'],
    ['初始化 state 摘要', String(config.initial_state_digest || '').slice(0, 32) + '…'],
    ['模型形状', JSON.stringify(config.model_shapes || {})],
  ]);

  const latest = payload.latest || {};
  const weightedSum = (typeof latest.weighted_loss_global === 'number'
    && typeof latest.weighted_loss_preproj === 'number')
    ? latest.weighted_loss_global + latest.weighted_loss_preproj : null;
  pgTable(qs('pg-paths'), ['量', '最新一步的值', '说明'], [
    ['LG（原生）', fmt(latest.loss_global, 4), 'LG = CE(QG, y) + CE(QGᵀ, y)，双向相加不取平均'],
    ['LP（投影前条件）', fmt(latest.loss_preproj, 4), 'LP = CE(QP, y) + CE(QPᵀ, y)，双向相加不取平均'],
    ['LS（稀疏）', fmt(latest.loss_sparse, 4), 'mean(|mask|)，本地 caption × 768，每步只算一次'],
    ['L_total', fmt(latest.loss_total, 4), '权重 ' + weights.global + ' / ' + weights.preproj
      + ' / ' + weights.sparse],
    ['5·LG + 5·LP', fmt(weightedSum, 4), '两项对齐的加权和（另加 1·LS）'],
    ['原生 I2T / T2I top1', fmt(latest.path_global_i2t_top1, 4) + ' / '
      + fmt(latest.path_global_t2i_top1, 4), '本地 batch × 全局候选'],
    ['条件 I2T / T2I top1', fmt(latest.path_preproj_i2t_top1, 4) + ' / '
      + fmt(latest.path_preproj_t2i_top1, 4), '同一候选池，换成投影前条件分数'],
    ['原生 / 条件 I2T max margin 均值',
      fmt(latest.path_global_i2t_max_margin_mean, 3) + ' / '
      + fmt(latest.path_preproj_i2t_max_margin_mean, 3), '正配分数 − 最强负配分数'],
    ['原生 / 条件 I2T LSE margin 均值',
      fmt(latest.path_global_i2t_lse_margin_mean, 3) + ' / '
      + fmt(latest.path_preproj_i2t_lse_margin_mean, 3), 'logsumexp 口径的难负例余量'],
    ['h 范数 / 条件输出范数 / 原生输出范数',
      fmt(latest.h_norm_mean, 3) + ' / ' + fmt(latest.conditioned_output_norm_mean, 3) + ' / '
      + fmt(latest.native_output_norm_mean, 3), '同一隐藏状态，条件路径按 mask 选坐标后过同一个 W'],
    ['原生与条件输出余弦（均值）', fmt(latest.native_vs_conditioned_cosine_mean, 6),
      '全开 mask 时定义上应为 1'],
    ['梯度范数：clip / gate / gate 输出层 / gate stem',
      fmt(latest.clip_grad_norm, 2) + ' / ' + fmt(latest.gate_grad_norm, 3) + ' / '
      + fmt(latest.gate_output_grad_norm, 3) + ' / ' + fmt(latest.gate_stem_grad_norm, 4),
      '初始时输出层权重为 0，stem 梯度为 0 属预期'],
  ]);

  const series = payload.series || {};
  const steps = payload.steps || [];
  const curveFields = Object.keys(PG_SERIES_LABELS).filter(field => (series[field] || [])
    .some(value => typeof value === 'number'));
  const curveTable = curveFields.map(field => {
    const values = (series[field] || []).filter(value => typeof value === 'number');
    const first = values.length ? values[0] : null;
    const last = values.length ? values[values.length - 1] : null;
    const min = values.length ? Math.min.apply(null, values) : null;
    const max = values.length ? Math.max.apply(null, values) : null;
    return [PG_SERIES_LABELS[field], fmt(first, 5), fmt(last, 5), fmt(min, 5), fmt(max, 5)];
  });
  pgTable(qs('pg-curves'), ['曲线（' + steps.length + ' 个记录点）', '首', '末', '最小', '最大'],
          curveTable);
  setText(qs('pg-curve-note'),
    '曲线点来自 run 自己的 salu_log.jsonl（每 10 步一条标量记录、每 25 步一条较重统计）。'
    + ' 只列出日志里真实存在的字段，缺失字段不会以 0 出现；记录条数 ' + pgValue(payload.record_count)
    + (payload.log_error ? '（读取告警：' + payload.log_error + '）' : ''));

  const mask = payload.mask || {};
  const stats = mask.statistics || {};
  pgTable(qs('pg-mask-stats'),
    ['mask / gate 统计（快照于 ' + pgValue(mask.completed_steps) + ' 步）', '数值'], [
      ['保留坐标数：均值 / 最小 / 最大',
        fmt(stats.mask_kept_mean, 1) + ' / ' + fmt(stats.mask_kept_min, 0) + ' / '
        + fmt(stats.mask_kept_max, 0)],
      ['保留比例均值 / 坐标级平均', fmt(stats.mask_keep_fraction_mean, 4) + ' / '
        + fmt(stats.mask_coordinate_mean, 4)],
      ['全开 / 全关 caption 比例', fmt(stats.mask_all_on_fraction, 4) + ' / '
        + fmt(stats.mask_all_off_fraction, 4)],
      ['gate 概率：均值 / 标准差 / 最小 / 最大',
        fmt(stats.gate_probability_mean, 4) + ' / ' + fmt(stats.gate_probability_std, 4) + ' / '
        + fmt(stats.gate_probability_min, 4) + ' / ' + fmt(stats.gate_probability_max, 4)],
      ['阈值附近（|p−0.5| < 0.05）比例', fmt(stats.gate_probability_near_threshold_fraction, 6)],
      ['跨 caption 的逐坐标概率变化（均值）',
        fmt(stats.gate_coordinate_variation_across_captions, 5)],
      ['gate 概率分位数 p5/p25/p50/p75/p95',
        (stats.gate_probability_quantiles || []).map(value => Number(value).toFixed(3)).join(' / ')],
    ]);
  renderPgMaskGrid(qs('pg-mask-grid'), payload);
  setText(qs('pg-mask-note'),
    '网格每一格是一个维度下标（0…767），换行只为了排版，不代表图像空间位置：'
    + ' 绿色 = 保留（mask = 1），灰色 = 关闭（mask = 0）。'
    + ' 快照来源：' + pgValue(mask.source || (mask.available
      ? 'run 目录内的 pgclip_mask_snapshot.json' : '未运行'))
    + (mask.available ? '；新增 optimizer update = ' + pgValue(mask.new_optimizer_updates)
      + '；checkpoint SHA ' + String(mask.checkpoint_sha256 || '').slice(0, 12) : ''));

  renderPgAuxiliary(payload, 'pg-aux', 'pg-aux-note');

  const energy = mask.energy || {};
  pgTable(qs('pg-energy'), ['两个能量指标（不要混为一谈）', '数值'], [
    ['preproj_retained_energy = ||h·m||² / ||h||²',
      '均值 ' + fmt(energy.preproj_retained_energy_mean, 4) + '（范围 '
      + fmt(energy.preproj_retained_energy_min, 4) + ' ~ '
      + fmt(energy.preproj_retained_energy_max, 4) + '）；硬 0/1 mask 下应在 [0, 1]'],
    ['projected_output_energy_ratio = ||(h·m)@W||² / ||h@W||²',
      '均值 ' + fmt(energy.projected_output_energy_ratio_mean, 4) + '，最大 '
      + fmt(energy.projected_output_energy_ratio_max, 4) + '，> 1 的比例 '
      + fmt(energy.projected_output_energy_ratio_above_one_fraction, 4)
      + '（可能 > 1，从不被截断）'],
    ['条件 / 原生输出余弦（快照均值）', fmt(energy.native_vs_conditioned_cosine_mean, 5)],
  ]);

  pgTable(qs('pg-progress'), ['训练进度', '数值'], [
    ['completed_steps / max_steps', pgValue(progress.completed_steps) + ' / '
      + pgValue(progress.max_steps)],
    ['世界大小 × 每卡 batch = 全局 batch', pgValue(progress.world_size) + ' × '
      + pgValue(progress.batch_size_per_gpu) + ' = ' + pgValue(progress.global_batch)],
    ['每 epoch 步数 / LR horizon', pgValue(progress.loader_batches) + ' / '
      + pgValue(progress.lr_horizon_steps)],
    ['CLIP lr / gate lr / warmup / weight decay', pgValue(progress.lr) + ' / '
      + pgValue(progress.gate_lr) + ' / ' + pgValue(progress.warmup_length) + ' / '
      + pgValue(progress.weight_decay)],
    ['每步秒数 / 吞吐 / 峰值显存（最新记录）',
      fmt(latest.sec_per_step, 2) + ' s / ' + fmt(latest.samples_per_sec, 1)
      + ' 样本每秒 / ' + fmt(latest.peak_memory_gb, 1) + ' GB'],
    ['日志统计口径', pgValue(config.statistics_scope)],
  ]);

  const evaluation = payload.evaluation || {};
  const coco = evaluation.coco || {};
  const urban = evaluation.urban1k || {};
  const base = evaluation.baseline_s0_500 || {};
  const baseCoco = base.coco || {};
  const baseUrban = base.urban1k || {};
  const delta = (value, reference) => (typeof value === 'number' && typeof reference === 'number')
    ? ((value - reference) * 100).toFixed(2) + ' pp' : '暂无';
  pgTable(qs('pg-eval'),
    ['数据集（原生 CLS/EOS；不用 gate、不做两路融合、不做 reranking）', 'I2T R@1 / R@5 / R@10',
     'T2I R@1 / R@5 / R@10', '与 S0@500 的 R@1 差（I2T / T2I）'],
    [
      ['COCO canonical ' + pgValue(coco.name || '（未产出）'),
        fmt(coco.i2t_r1, 4) + ' / ' + fmt(coco.i2t_r5, 4) + ' / ' + fmt(coco.i2t_r10, 4),
        fmt(coco.t2i_r1, 4) + ' / ' + fmt(coco.t2i_r5, 4) + ' / ' + fmt(coco.t2i_r10, 4),
        delta(coco.i2t_r1, baseCoco.i2t_r1) + ' / ' + delta(coco.t2i_r1, baseCoco.t2i_r1)],
      ['S0@500（冻结参照）',
        fmt(baseCoco.i2t_r1, 4) + ' / ' + fmt(baseCoco.i2t_r5, 4) + ' / '
        + fmt(baseCoco.i2t_r10, 4),
        fmt(baseCoco.t2i_r1, 4) + ' / ' + fmt(baseCoco.t2i_r5, 4) + ' / '
        + fmt(baseCoco.t2i_r10, 4), '—'],
      ['Urban-1k ' + pgValue(urban.name || '（未产出）'),
        fmt(urban.i2t_r1, 4) + ' / ' + fmt(urban.i2t_r5, 4) + ' / ' + fmt(urban.i2t_r10, 4),
        fmt(urban.t2i_r1, 4) + ' / ' + fmt(urban.t2i_r5, 4) + ' / ' + fmt(urban.t2i_r10, 4),
        delta(urban.i2t_r1, baseUrban.i2t_r1) + ' / ' + delta(urban.t2i_r1, baseUrban.t2i_r1)],
      ['S0@500（冻结参照）',
        fmt(baseUrban.i2t_r1, 4) + ' / ' + fmt(baseUrban.i2t_r5, 4) + ' / '
        + fmt(baseUrban.i2t_r10, 4),
        fmt(baseUrban.t2i_r1, 4) + ' / ' + fmt(baseUrban.t2i_r5, 4) + ' / '
        + fmt(baseUrban.t2i_r10, 4), '—'],
    ]);
  const detail = evaluation.verdict_detail || {};
  setText(qs('pg-conclusion'),
    '冻结晋级门（只在 500 步、只看 COCO 原始精度）：I2T R@1 ≥ 0.6058 且 T2I R@1 ≥ 0.41236，'
    + '且至少一项严格更高 → PROMISING_AT_500，否则 FAIL。本 run 判定：'
    + pgValue(evaluation.verdict || '未产出')
    + (typeof detail.i2t_delta_points === 'number'
      ? '（I2T ' + detail.i2t_delta_points.toFixed(2) + ' pp，T2I '
        + detail.t2i_delta_points.toFixed(2) + ' pp）' : '')
    + '。Urban-1k 单列，不参与判定。'
    + ' 本轮没有 native-only / native+post-projection 控制臂，因此即使提升也只能归因于 PG-CLIP 这一整套组合。');
}

/* ---------------------------------------------------------------- J: CG-CLIP v0.1
 *
 * Native alignment plus a text-gated final-block CLS attention. Everything here comes from one
 * read-only JSON (`/api/run/<id>/cgclip`) built from the run's own small files. The 4-GPU job is
 * still training, so most artifacts (evaluation rows, student export, the optional attention
 * snapshot) do not exist yet: every absent field renders as 暂无/未产出 and never as 0.
 */

const CG_PATH_LABELS = { path_global: '原生（native）path_global_*', path_attention: '条件（gate）path_attention_*' };
const CG_DIRECTION_LABELS = { i2t: 'I2T', t2i: 'T2I' };
const CG_DIRECTION_KEYS = ['i2t', 't2i'];
const CG_STAT_LABELS = [
  ['top1', 'top1', 4],
  ['positive_win_fraction', '正例胜出比例', 4],
  ['positive_mean', 'positive_mean（正配分数）', 3],
  ['strongest_negative_mean', 'strongest_negative_mean（最强负配）', 3],
  ['max_margin_mean', 'max_margin_mean', 3],
  ['lse_margin_mean', 'lse_margin_mean', 3],
  ['ce_mean', 'ce_mean', 5],
  ['lse_margin_min', 'lse_margin_min（最差）', 3],
  ['max_margin_min', 'max_margin_min（最差）', 3],
  ['ce_from_lse_margin_max_abs_diff', 'CE 与 LSE margin 的自洽差（绝对值上限）', 8],
];

function cgNum(value, digits) {
  return (typeof value === 'number' && Number.isFinite(value)) ? value.toFixed(digits) : '暂无';
}

function cgTable(container, headers, rows) {
  return tableBlock(container, headers, rows.length ? rows : [['暂无']]);
}

/* one table per (path, direction) pair, so the two paths and the two directions are never mixed */
function cgDirectionTable(container, payload, path, direction) {
  container.textContent = '';
  const curves = (payload.curves || {})[path] || {};
  const block = curves[direction] || {};
  const prefix = path + '_' + direction + '_';
  const caption = document.createElement('p');
  caption.className = 'muted';
  caption.textContent = CG_PATH_LABELS[path] + ' · ' + CG_DIRECTION_LABELS[direction]
    + '（字段前缀 ' + prefix + '*）';
  container.appendChild(caption);
  tableBlock(container,
    ['统计量（字段 ' + prefix + '…）', '最新值'],
    CG_STAT_LABELS.map(entry => [entry[1], cgNum(block[prefix + entry[0]], entry[2])]));
}

function cgRange(values) {
  const numbers = (values || []).filter(value => typeof value === 'number' && Number.isFinite(value));
  if (!numbers.length) return ['暂无', '暂无', '暂无', '暂无'];
  return [cgNum(numbers[0], 5), cgNum(numbers[numbers.length - 1], 5),
          cgNum(Math.min.apply(null, numbers), 5), cgNum(Math.max.apply(null, numbers), 5)];
}

/* the 14x14 patch grid; the CLS slot is a separate box and is never a 197th cell in the grid */
function renderCgGrid(container, payload) {
  container.textContent = '';
  const grid = payload.grid || {};
  const samples = grid.samples || [];
  if (!grid.available || !samples.length) {
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent = grid.not_run || ('未产出：' + (grid.error || '日志里没有 gate_grid_samples'));
    container.appendChild(note);
    return;
  }
  samples.forEach(sample => {
    const wrap = document.createElement('div');
    wrap.className = 'gate-sample';
    const head = document.createElement('div');
    head.className = 'mask-caption';
    head.textContent = '样本 #' + sample.sample + '（第 ' + sample.completed_steps + ' 步）· 保留 '
      + sample.kept + '/' + sample.cells + ' 格';
    wrap.appendChild(head);
    const row = document.createElement('div');
    row.className = 'gridrow';
    const board = document.createElement('div');
    board.className = 'gate-grid';
    board.style.gridTemplateColumns = 'repeat(' + sample.side + ', 12px)';
    (sample.grid || []).forEach((value, index) => {
      const cell = document.createElement('span');
      cell.className = 'gcell ' + (value >= 0.5 ? 'on' : 'off');
      cell.title = 'token #' + index + ' = ' + value;
      board.appendChild(cell);
    });
    row.appendChild(board);
    const cls = document.createElement('div');
    cls.className = 'cls-slot';
    const clsCell = document.createElement('span');
    clsCell.className = 'gcell on';
    clsCell.title = 'CLS 槽位固定 gate = ' + sample.cls_slot_gate_value;
    cls.appendChild(clsCell);
    const label = document.createElement('div');
    label.className = 'k';
    label.textContent = 'CLS 槽位（固定 gate = ' + sample.cls_slot_gate_value + '，不进 14×14 网格）';
    cls.appendChild(label);
    row.appendChild(cls);
    wrap.appendChild(row);
    container.appendChild(wrap);
  });
}

function renderCgClip(payload) {
  state.cgclip = payload;
  const ids = ['cg-scope', 'cg-identity', 'cg-gate', 'cg-precision', 'cg-losses', 'cg-curves',
    'cg-curve-note', 'cg-direction-i2t', 'cg-direction-t2i', 'cg-direction-note',
    'cg-gate-stats', 'cg-gate-note',
    'cg-tile-stats', 'cg-tile-note', 'cg-grid', 'cg-grid-note', 'cg-cls', 'cg-heads',
    'cg-cls-note', 'cg-snapshot', 'cg-cost', 'cg-export', 'cg-eval'];
  if (!payload.available) {
    setText(qs('cg-banner'), '未运行：这个 run 不是 CG-CLIP v0.1（objective='
      + pgValue(payload.objective) + '，期望 ' + pgValue(payload.objective_expected) + '）。'
      + '只有 objective = clip_native_caption_gated_cls 的 run 才会在这里显示两路损失与门控网格。');
    setText(qs('cg-reminders'), '本区域不显示推测值：objective 不匹配时只显示"未运行"'
      + (payload.error ? '（' + payload.error + '）' : ''));
    ids.forEach(id => { qs(id).textContent = ''; });
    setText(qs('cg-conclusion'), '');
    return;
  }
  setText(qs('cg-reminders'), (payload.reminders || []).join('   '));
  const identity = payload.identity || {};
  const gate = payload.gate || {};
  const weights = identity.loss_weights || {};
  const cost = payload.cost || {};
  setText(qs('cg-banner'),
    '已运行（只读，训练进行中）· objective ' + pgValue(identity.objective) + ' · arm '
    + pgValue(identity.arm) + ' · phase ' + pgValue(identity.phase) + ' · 记录 '
    + pgValue(payload.record_count) + ' 条 · 实现 SHA '
    + String(cost.implementation_sha || '').slice(0, 12)
    + ' · 本页不加载 checkpoint、不触发 GPU 前向、不写入任何文件');

  kvGrid(qs('cg-scope'), [
    ['run id', payload.run_id],
    ['状态口径', '这份数据来自正在运行的 4 卡训练：config 与日志先出现，'
      + 'evaluation/ 与 student_export/ 在导出与评估阶段才写出'],
    ['日志统计口径', pgValue(identity.statistics_scope)],
    ['每步记录', '标量记录每 10 步、重统计每 25 步；缺失字段显示"暂无"，不会以 0 出现'],
    ['CLS 槽位', pgValue(gate.cls_self_gate)],
    ['gate patch 数', pgValue(gate.patches) + ' 个，12 个视觉头共享同一份'],
  ]);

  cgTable(qs('cg-identity'), ['身份项', '值'], [
    ['arm', pgValue(identity.arm)],
    ['objective', pgValue(identity.objective)],
    ['phase', pgValue(identity.phase)],
    ['gate kind', pgValue(identity.gate_kind)],
    ['损失权重（global / attention / sparse）', pgValue(weights.global) + ' / '
      + pgValue(weights.attention) + ' / ' + pgValue(weights.sparse)],
    ['损失组合', pgValue(identity.loss_combination)],
    ['固定分数尺度 fixed_scale', pgValue(identity.fixed_scale) + '（分数 = 100 × 余弦，从不缩放）'],
    ['原生路径定义', pgValue((identity.two_paths || {}).native)],
    ['条件路径定义', pgValue((identity.two_paths || {}).attention)],
    ['条件路径使用范围', pgValue((identity.attention_route || {}).conditional_path_use)],
    ['候选规则', pgValue(identity.candidate_rule)],
    ['梯度职责', pgValue(identity.grader)],
    ['caption 流', pgValue(identity.caption_stream)],
    ['视图', pgValue(identity.view)],
    ['DDP 路线', pgValue(identity.ddp_route) + '（world_size 倍率：'
      + (identity.no_world_size_factor ? '无' : '有') + '）'],
    ['分块', 'image_chunk=' + pgValue((identity.chunking || {}).image_chunk)
      + '，text_chunk=' + pgValue((identity.chunking || {}).text_chunk)
      + '，cond_checkpoint=' + pgValue((identity.chunking || {}).cond_checkpoint)],
    ['x11 取点', pgValue(identity.x11_source)],
    ['文本侧', pgValue(identity.text_source)],
    ['种子 / gate 种子', pgValue(identity.seed) + ' / ' + pgValue(identity.gate_seed)],
    ['max_steps / epochs / 每 epoch 步数', pgValue(identity.max_steps) + ' / '
      + pgValue(identity.epochs) + ' / ' + pgValue(identity.loader_batches)],
    ['CLIP lr / gate lr / warmup / weight decay', pgValue(identity.lr) + ' / '
      + pgValue(identity.gate_lr) + ' / ' + pgValue(identity.warmup_length) + ' / '
      + pgValue(identity.weight_decay)],
    ['初始化 state', pgValue(identity.init_state)],
  ]);

  cgTable(qs('cg-gate'), ['gate 结构（' + pgValue(gate.kind) + '）', '值'], [
    ['gate 描述', pgValue(gate.description)],
    ['gate patch 数', pgValue(gate.patches) + '（期望 ' + pgValue(gate.patches_expected) + '）'],
    ['key 维度', pgValue(gate.key_dim)],
    ['A（query 侧）初始化', pgValue(gate.query_init)],
    ['B（key 侧）初始化', pgValue(gate.key_init)],
    ['可训练标量 bias 初始化', 'log 8 = ' + pgValue(gate.bias_init_log) + '（记录值 '
      + pgValue(gate.bias_init) + '）；当前值 '
      + cgNum((payload.latest || {}).gate_bias_value, 6)],
    ['前向', pgValue(gate.forward)],
    ['A 的输入', pgValue(gate.query_input)],
    ['B 的输入', pgValue(gate.key_input)],
    ['CLS 槽位', pgValue(gate.cls_self_gate)],
    ['soft_floor / top_k', pgValue(gate.soft_floor) + ' / ' + pgValue(gate.top_k)],
    ['初始化实测（logits / 概率）', 'logits ' + pgValue((gate.init || {}).logits_min) + ' … '
      + pgValue((gate.init || {}).logits_max) + '，概率均值 '
      + pgValue((gate.init || {}).probability_mean) + '（全开 ' + pgValue((gate.init || {}).mask_all_one)
      + '）'],
  ]);
  pgTable(qs('cg-precision'), ['精度', '配置值'], [
    ['核心精度', pgValue(payload.precision)],
    ['TF32（matmul / cudnn）', 'matmul_allow_tf32 = ' + pgValue((payload.tf32 || {}).matmul_allow_tf32)
      + '，cudnn_allow_tf32 = ' + pgValue((payload.tf32 || {}).cudnn_allow_tf32)],
    ['视觉末块形状', JSON.stringify((gate.visual_spec || {}).last_num_heads || '') + ' 头 × dim '
      + pgValue((gate.visual_spec || {}).last_head_dim) + '，embed '
      + pgValue((gate.visual_spec || {}).last_embed_dim) + '，context '
      + pgValue((gate.visual_spec || {}).context_length)],
  ]);

  const series = payload.series || {};
  const steps = payload.steps || [];
  const latest = payload.latest || {};
  cgTable(qs('cg-losses'), ['损失项（' + steps.length + ' 个记录点）', '首', '末', '最小', '最大'], [
    ['LG = 原生对齐（I2T + T2I）'].concat(cgRange(series.loss_global_sum)),
    ['LA = 条件对齐（I2T + T2I）'].concat(cgRange(series.loss_attention_sum)),
    ['LS = 正例对 196 个 patch gate 的均值'].concat(cgRange(series.loss_sparse)),
    ['5×LG / 5×LA / 1×LS（末值）',
      cgNum(latest.weighted_loss_global, 5) + ' / ' + cgNum(latest.weighted_loss_attention, 5)
      + ' / ' + cgNum(latest.weighted_loss_sparse, 5), '', '', ''],
    ['L_total = 5·LG + 5·LA + LS（末值）', cgNum(latest.loss_total, 5), '', '',
      '方向内双向相加、不取平均'],
  ]);
  const curveFields = [
    'loss_global_i2t', 'loss_global_t2i', 'loss_attention_i2t', 'loss_attention_t2i',
    'loss_sparse', 'loss_total', 'sec_per_step', 'gate_bias_value', 'gate_grad_norm',
    'gate_query_weight_norm', 'gate_key_weight_norm', 'clip_grad_norm', 'last_block_grad_norm',
    'lr', 'gate_lr', 'samples_per_sec', 'peak_memory_gb', 'epoch', 'captions_seen',
    'synchronized_pair_presentations', 'effective_length_mean',
  ];
  cgTable(qs('cg-curves'), ['曲线（' + steps.length + ' 个记录点）', '首', '末', '最小', '最大'],
    curveFields.filter(field => (series[field] || []).some(v => typeof v === 'number'))
      .map(field => {
        const values = (series[field] || []).filter(v => typeof v === 'number');
        return [field, cgNum(values[0], 5), cgNum(values[values.length - 1], 5),
                cgNum(Math.min.apply(null, values), 5), cgNum(Math.max.apply(null, values), 5)];
      }));
  setText(qs('cg-curve-note'),
    '曲线点来自 run 自己的 salu_log.jsonl（completed_steps = ' + pgValue(payload.record_count)
    + ' 条记录）。两路各自独立成表：原生 path_global_* 与条件 path_attention_* 不会混在同一列里；'
    + '缺失字段显示"暂无"，不会填 0。'
    + (payload.log_error ? '读取告警：' + payload.log_error : ''));

  CG_DIRECTION_KEYS.forEach(direction => {
    const container = qs('cg-direction-' + direction);
    container.textContent = '';
    const heading = document.createElement('h4');
    heading.textContent = CG_DIRECTION_LABELS[direction] + ' 方向：原生路径与条件路径（分开成表）';
    container.appendChild(heading);
    const nativeBox = document.createElement('div');
    const attentionBox = document.createElement('div');
    container.appendChild(nativeBox);
    container.appendChild(attentionBox);
    cgDirectionTable(nativeBox, payload, 'path_global', direction);
    cgDirectionTable(attentionBox, payload, 'path_attention', direction);
  });
  setText(qs('cg-direction-note'),
    '每一格都来自重统计行的对应字段，原生与条件两路各自一表，正例/负例/margin/top1 全部同池同规则可比；'
    + '缺失字段显示"暂无"，不会用另一路的值代填。');

  const heavy = payload.latest_heavy || {};
  cgTable(qs('cg-gate-stats'),
    ['正例对 gate 统计（重统计行，第 ' + pgValue(heavy.completed_steps) + ' 步）', '数值'], [
      ['保留 patch 数：均值 / 最小 / 最大（共 196）',
        cgNum(heavy.positive_pairs_gate_kept_mean, 3) + ' / '
        + cgNum(heavy.positive_pairs_gate_kept_min, 0) + ' / '
        + cgNum(heavy.positive_pairs_gate_kept_max, 0)],
      ['保留比例均值 / 坐标级平均',
        cgNum(heavy.positive_pairs_gate_keep_fraction_mean, 4) + ' / '
        + cgNum(heavy.positive_pairs_gate_coordinate_mean, 4)],
      ['全关 / 全开比例', cgNum(heavy.positive_pairs_gate_all_off_fraction, 4) + ' / '
        + cgNum(heavy.positive_pairs_gate_all_on_fraction, 4)],
      ['软概率：均值 / 标准差 / 最小 / 最大',
        cgNum(heavy.positive_pairs_gate_probability_mean, 4) + ' / '
        + cgNum(heavy.positive_pairs_gate_probability_std, 4) + ' / '
        + cgNum(heavy.positive_pairs_gate_probability_min, 4) + ' / '
        + cgNum(heavy.positive_pairs_gate_probability_max, 4)],
      ['阈值附近（软概率）比例',
        cgNum(heavy.positive_pairs_gate_probability_near_threshold_fraction, 6)],
      ['软概率分位数',
        (heavy.positive_pairs_gate_probability_quantiles || []).map(v => Number(v).toFixed(4)).join(' / ')],
      ['作用范围', pgValue(heavy.positive_pairs_gate_scope)],
    ]);
  setText(qs('cg-gate-note'),
    '口径：保留数/保留比例来自硬门（mask = 1 的 patch 数），软概率分布来自同一批正例对的 gate 概率；'
    + '两者口径不同，不可互相替代。正例对只统计 rank0 本地 batch 的真实 (image, caption) 正例，'
    + 'CLS 槽位被排除在外。');

  cgTable(qs('cg-tile-stats'), ['tile 统计（text_chunk × image_chunk 分块）', '数值'], [
    ['保留 patch 数：均值 / 最小 / 最大（共 196）',
      cgNum(heavy.tile_gate_kept_mean, 3) + ' / ' + cgNum(heavy.tile_gate_kept_min, 0) + ' / '
      + cgNum(heavy.tile_gate_kept_max, 0)],
    ['保留比例均值 / 坐标级平均',
      cgNum(heavy.tile_gate_keep_fraction_mean, 4) + ' / ' + cgNum(heavy.tile_gate_coordinate_mean, 4)],
    ['全关 / 全开比例',
      cgNum(heavy.tile_gate_all_off_fraction, 4) + ' / ' + cgNum(heavy.tile_gate_all_on_fraction, 4)],
    ['软概率：均值 / 标准差 / 最小 / 最大',
      cgNum(heavy.tile_gate_probability_mean, 4) + ' / ' + cgNum(heavy.tile_gate_probability_std, 4)
      + ' / ' + cgNum(heavy.tile_gate_probability_min, 6) + ' / '
      + cgNum(heavy.tile_gate_probability_max, 4)],
    ['阈值附近比例', cgNum(heavy.tile_gate_probability_near_threshold_fraction, 6)],
    ['软概率分位数',
      (heavy.tile_gate_probability_quantiles || []).map(v => Number(v).toFixed(4)).join(' / ')],
    ['跨图同文本变化量（sample = ' + pgValue(heavy.tile_gate_pair_sample) + '）',
      cgNum(heavy.tile_gate_variation_across_images_same_text, 5)],
    ['跨文本同图变化量（sample = ' + pgValue(heavy.tile_gate_pair_sample) + '）',
      cgNum(heavy.tile_gate_variation_across_texts_same_image, 5)],
    ['变化量样本数（tile_gate_pair_sample / gate_pair_variation_sample）',
      pgValue(heavy.tile_gate_pair_sample) + ' / ' + pgValue(heavy.gate_pair_variation_sample)],
    ['tile 范围', pgValue(heavy.tile_scope)],
  ]);
  setText(qs('cg-tile-note'),
    '两个变化量只有在 tile_gate_pair_sample 这么大的样本上才算出来，不是全量统计，'
    + '也不能当成语义信息比例：它们只描述 gate 取值在"换图"和"换文本"两个方向上的分布。');

  renderCgGrid(qs('cg-grid'), payload);
  const grid = payload.grid || {};
  const firstSample = (grid.samples || [])[0] || {};
  setText(qs('cg-grid-note'),
    '网格每一格是一个 patch 的硬 gate 取值（' + pgValue(grid.side) + '×' + pgValue(grid.side) + ' = '
    + pgValue(grid.cells) + ' 格，期望 ' + pgValue(grid.expected_cells) + '）：绿色 = 保留（gate = 1），'
    + '灰色 = 关闭（gate = 0）。网格说明：' + pgValue(firstSample.note || grid.note)
    + '。样本序号：' + pgValue(firstSample.sample)
    + '（第 ' + pgValue(firstSample.completed_steps) + ' 步）。CLS 槽位是固定的 gate = 1，'
    + '在右侧单独显示，永远不折进这 196 格。'
    + (grid.error ? ' 解析告警：' + grid.error : ''));

  const cls = payload.cls_read || {};
  cgTable(qs('cg-cls'), ['CLS 读出诊断（原生 vs 条件）', '数值'], [
    ['原生 CLS 自注意力质量均值', cgNum(cls.native_cls_self_mass_mean, 6)],
    ['条件 CLS 自注意力质量均值', cgNum(cls.conditional_cls_self_mass_mean, 6)],
    ['原生 / 条件 patch 质量均值',
      cgNum(cls.native_patch_mass_mean, 6) + ' / ' + cgNum(cls.conditional_patch_mass_mean, 6)],
    ['原生自质量逐头最小 / 最大',
      cgNum(cls.native_cls_self_mass_per_head_min, 6) + ' / '
      + cgNum(cls.native_cls_self_mass_per_head_max, 6)],
    ['native_out_norm_ratio_mean', cgNum(cls.native_out_norm_ratio_mean, 6)],
    ['条件 vs 原生 CLS 余弦（末块输出）', cgNum(cls.conditional_vs_native_cls_cosine_mean, 6)],
    ['条件 vs 原生投影后余弦', cgNum(cls.conditional_vs_native_projected_cosine_mean, 6)],
    ['单位', pgValue(cls.cls_self_mass_unit)],
    ['原生注意力形状', JSON.stringify(cls.native_attention_shape || null)],
    ['统计范围', pgValue(cls.scope)],
  ]);
  const nativeHeads = cls.native_attention_cls_self_mass || [];
  const conditionalHeads = cls.conditional_attention_cls_self_mass || [];
  const headCount = Math.max(nativeHeads.length, conditionalHeads.length);
  const headRows = [];
  for (let index = 0; index < headCount; index += 1) {
    headRows.push(['head ' + index, cgNum(nativeHeads[index], 6), cgNum(conditionalHeads[index], 6)]);
  }
  cgTable(qs('cg-heads'), ['逐头显示值（12 个视觉头，按头顺序）', '原生 CLS 自质量',
    '条件 CLS 自质量'], headRows);
  setText(qs('cg-cls-note'),
    '逐头两张列表是"每头显示值"，不是 12 个可独立训练的门：gate 只有一份 196 维，'
    + '被 12 个头共享（' + pgValue(cls.attention_head_axis_note) + '）。'
    + '余弦高只说明条件路径与原生路径的读出接近，不等于检索更好。');

  const snapshotInfo = payload.attention_snapshot || {};
  cgTable(qs('cg-snapshot'), ['可选注意力快照诊断', '状态'], [
    ['cgclip_v01_diag/cgclip_attention_snapshot.json',
      snapshotInfo.available ? '已读取' : '未产出'],
    ['说明', snapshotInfo.available ? JSON.stringify(snapshotInfo.payload).slice(0, 400)
      : pgValue(snapshotInfo.not_run || snapshotInfo.error)],
  ]);

  cgTable(qs('cg-cost'), ['溯源与成本', '值'], [
    ['implementation_sha', pgValue(cost.implementation_sha)],
    ['run 状态 / 阶段 / 尝试次数', pgValue(cost.phase) + ' / ' + pgValue(cost.stage) + ' / '
      + pgValue(cost.attempt)],
    ['阶段列表与退出码', (cost.stages || []).join(' → ') + '　退出码 '
      + JSON.stringify(cost.exit_codes || {})],
    ['GPU 数（world_size）', pgValue(cost.world_size) + '（' + pgValue(cost.gpu_check) + '）'],
    ['每卡 batch × 全局 batch',
      pgValue(cost.batch_size_per_gpu) + ' × ' + pgValue(cost.global_batch)],
    ['峰值显存（GiB，1024³ 字节）',
      cgNum(cost.peak_memory_gi_b !== null && cost.peak_memory_gi_b !== undefined
        ? cost.peak_memory_gi_b : cost.peak_memory_gb, 3) + ' GiB'
      + '（记录字段 peak_memory_gb 的实际单位：' + pgValue(cost.peak_memory_note) + '）'],
    ['sec/step / samples/sec', cgNum(cost.sec_per_step, 3) + ' s / '
      + cgNum(cost.samples_per_sec, 3)],
    ['单卡累计 captions_seen / 全局累计 pair 表示数',
      pgValue(cost.captions_seen) + '（每一步单卡 ' + pgValue(cost.batch_size_per_gpu)
      + ' 条文本，rank 本地累计） / ' + pgValue(cost.synchronized_pair_presentations)
      + '（同步计数，约等于 completed_steps × global_batch，与单卡数不同口径）'],
    ['单步全局 pair 数（global_pairs / global_batch）', pgValue(cost.global_pairs)
      + '（每步全局同步的 (图, 文本) 对）'],
    ['500 步 × 1024 的全局 pair 表示数', pgValue(cost.global_pair_presentations_total)
      + '（' + pgValue(cost.global_pair_presentations_note) + '）'],
    ['pair 表示速率 / 样本速率',
      cgNum(cost.pair_presentations_per_sec, 3) + ' pair/s · ' + cgNum(cost.samples_per_sec, 3)
      + ' 样本/s'],
    ['统计范围 statistics_scope', pgValue(cost.statistics_scope)],
    ['初始化文件 SHA256', pgValue(cost.init_file_sha256)],
    ['初始化 state 摘要', pgValue(cost.initial_state_digest)],
    ['gate 参数摘要 / clip 状态摘要', pgValue(cost.gate_param_digest) + ' / '
      + pgValue(cost.clip_state_digest_prefix)],
    ['rank 本地流摘要', JSON.stringify(cost.rank_local_stream_digests || null).slice(0, 300)],
    ['5/5/1 权重的日志记录', JSON.stringify(cost.weights_5_5_1 || weights)],
    ['base model / clip lr', pgValue(cost.base_model) + ' / ' + pgValue(cost.clip_lr)],
    ['run 目录 / 仓库', pgValue(cost.run_dir) + ' / ' + pgValue(cost.repo)],
    ['启动时间 / 保存步', pgValue(cost.started_at_iso) + ' / ' + pgValue(cost.save_steps)],
  ]);
  const checkpoint = cost.checkpoint || {};
  const student = cost.student || {};
  const checkpointRow = ck => (ck.available
    ? '已产出（' + (ck.bytes ? (ck.bytes / (1024 * 1024)).toFixed(1) + ' MiB' : '大小未知') + '）'
    : '未产出');
  cgTable(qs('cg-export'), ['产物', '状态', 'SHA256'], [
    ['CG_CLIP_V01_step000500.pt', checkpointRow(checkpoint), pgValue(checkpoint.sha256 || '未计算（不打开 checkpoint）')],
    ['student_export/cgclip_v01_student.pt', checkpointRow(student), pgValue(student.sha256 || '未计算（不打开 checkpoint）')],
    ['student_metadata.json',
      (cost.student_metadata ? '已读取' : '未产出'),
      JSON.stringify(cost.student_metadata || {}).slice(0, 500)],
  ]);

  const evaluation = payload.evaluation || {};
  const coco = evaluation.coco || {};
  const urban = evaluation.urban1k || {};
  const floors = evaluation.gate || {};
  const cocoFloor = `冻结地板 0.6058 / 0.41236`;
  cgTable(qs('cg-eval'),
    ['检索结果（原生 CLS/EOS，不用 gate、不融合、不 rerank）', 'I2T R@1 / R@5 / R@10',
     'T2I R@1 / R@5 / R@10', '对照'], [
      ['COCO canonical ' + pgValue(coco.name || '（未产出）'),
        fmt(coco.i2t_r1, 5) + ' / ' + fmt(coco.i2t_r5, 5) + ' / ' + fmt(coco.i2t_r10, 5),
        fmt(coco.t2i_r1, 5) + ' / ' + fmt(coco.t2i_r5, 5) + ' / ' + fmt(coco.t2i_r10, 5),
        cocoFloor],
      ['Urban-1k ' + pgValue(urban.name || '（未产出）'),
        fmt(urban.i2t_r1, 5) + ' / ' + fmt(urban.i2t_r5, 5) + ' / ' + fmt(urban.i2t_r10, 5),
        fmt(urban.t2i_r1, 5) + ' / ' + fmt(urban.t2i_r5, 5) + ' / ' + fmt(urban.t2i_r10, 5),
        '不参与冻结门'],
      ['S0@500 冻结参照（COCO）',
        fmt((evaluation.baseline_s0_500 || {}).i2t_r1, 5) + ' / '
        + fmt((evaluation.baseline_s0_500 || {}).i2t_r5, 5) + ' / '
        + fmt((evaluation.baseline_s0_500 || {}).i2t_r10, 5),
        fmt((evaluation.baseline_s0_500 || {}).t2i_r1, 5) + ' / '
        + fmt((evaluation.baseline_s0_500 || {}).t2i_r5, 5) + ' / '
        + fmt((evaluation.baseline_s0_500 || {}).t2i_r10, 5), '—'],
    ]);
  setText(qs('cg-conclusion'),
    '冻结晋级门（只在 500 步、只看 COCO 原始精度）：' + pgValue(floors.rule)
    + '。本 run 的两个检索结果文件出现之前，这里只显示"未产出"，不会预填任何数字：'
    + (evaluation.not_run || []).join('；')
    + '。页面按文件原样报告（不做四舍五入到地板位数之外的加工），也不把 Urban-1k 混进判定。');
}

function download(filename, text, type) {
  const blob = new Blob([text], { type: type || 'application/json' });
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
  ['nuisance-model', 'nuisance-readout', 'nuisance-base', 'nuisance-variant'].forEach(id => {
    qs(id).addEventListener('change', () => {
      if (state.nuisance) renderTextNuisance(state.nuisance);
    });
  });
  ['c512-model', 'c512-scene'].forEach(id => {
    qs(id).addEventListener('change', () => {
      if (state.clip512) renderClip512(state.clip512);
    });
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
