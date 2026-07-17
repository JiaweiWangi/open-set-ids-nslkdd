// Tab 切换
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

const API = '';

// ===== Tab 1: 数据集概览 =====
async function loadOverview() {
  const res = await fetch(`${API}/api/overview`);
  const d = await res.json();

  // 统计卡片
  const stats = [
    {label: '训练集样本', value: d.train_size.toLocaleString(), cls: 'blue'},
    {label: '测试集样本', value: d.test_size.toLocaleString(), cls: 'blue'},
    {label: '已知类数', value: d.n_known, cls: 'green'},
    {label: '未知攻击种数', value: d.n_unknown, cls: 'orange'},
    {label: '特征维度', value: d.feature_dim, cls: ''},
  ];
  document.getElementById('overview-stats').innerHTML = stats.map(s =>
    `<div class="stat-item"><div class="label">${s.label}</div><div class="value ${s.cls}">${s.value}</div></div>`
  ).join('');

  // 训练集分布柱状图
  const trainChart = echarts.init(document.getElementById('chart-train-dist'));
  const td = d.train_dist.sort((a, b) => b.n - a.n);
  trainChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, bottom: 90 },
    xAxis: { type: 'category', data: td.map(x => x.attack), axisLabel: { rotate: 45 } },
    yAxis: { type: 'log', name: '样本数(log)' },
    series: [{ type: 'bar', data: td.map(x => x.n),
      itemStyle: { color: '#3498db' } }],
  });

  // 未知攻击表
  const unk = d.unknown_attacks.sort((a, b) => b.n - a.n);
  document.getElementById('unknown-table').innerHTML = `
    <table>
      <thead><tr><th>未知攻击</th><th>样本数</th><th>是否数据极限</th></tr></thead>
      <tbody>${unk.map(u => `
        <tr>
          <td>${u.attack}</td>
          <td>${u.n}</td>
          <td>${u.is_overlap ? '<span class="tag overlap">重叠(无法检出)</span>' : '<span class="tag ok">可检出</span>'}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;

  // 数据极限列表
  document.getElementById('overlap-list').innerHTML =
    `<div class="unknown-grid">${d.overlap_attacks.map(a =>
      `<div class="unknown-item overlap"><span class="name">${a}</span></div>`).join('')}</div>`;
}

// ===== Tab 2: 批量评估 =====
async function loadTestFiles() {
  const res = await fetch(`${API}/api/test-files`);
  const files = await res.json();
  const sel = document.getElementById('test-file-select');
  sel.innerHTML = files.map(f => `<option value="${f.path}">${f.name}</option>`).join('');
}

document.getElementById('btn-evaluate').addEventListener('click', async () => {
  const btn = document.getElementById('btn-evaluate');
  const status = document.getElementById('eval-status');
  const sel = document.getElementById('test-file-select');
  btn.disabled = true;
  status.className = 'status running';
  status.textContent = '⏳ 加载模型并评估中(约20秒)...';
  document.getElementById('eval-results').style.display = 'none';

  try {
    const res = await fetch(`${API}/api/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test_file: sel.value }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || '评估失败');
    }
    const d = await res.json();
    renderEval(d);
    status.className = 'status done';
    status.textContent = `✅ 完成 (${d.n_test} 条测试样本)`;
  } catch (e) {
    status.className = 'status error';
    status.textContent = '❌ ' + e.message;
    console.error(e);
  } finally {
    btn.disabled = false;
  }
});

function renderEval(d) {
  document.getElementById('eval-results').style.display = 'block';
  const u = d.unknown_detection, k = d.known_classification, o = d.overall;
  const stats = [
    {label: '未知检测 F1', value: u.F1, cls: 'green'},
    {label: '未知 P', value: u.P, cls: 'blue'},
    {label: '未知 R', value: u.R, cls: 'blue'},
    {label: '已知类 TNR', value: u.TNR, cls: 'green'},
    {label: '已知分类 acc', value: k.acc, cls: 'orange'},
    {label: '整体 acc', value: o.acc, cls: 'orange'},
  ];
  document.getElementById('eval-stats').innerHTML = stats.map(s =>
    `<div class="stat-item"><div class="label">${s.label}</div><div class="value ${s.cls}">${s.value}</div></div>`
  ).join('');

  // 未知攻击检出率柱状图
  const unkChart = echarts.init(document.getElementById('chart-unknown-detect'));
  const unk = d.per_unknown.sort((a, b) => b.detect_rate - a.detect_rate);
  unkChart.setOption({
    tooltip: { trigger: 'axis', formatter: p => `${p[0].name}<br/>检出率: ${(p[0].value*100).toFixed(1)}%<br/>n=${p[0].data.n}` },
    grid: { left: 50, right: 20, bottom: 90 },
    xAxis: { type: 'category', data: unk.map(x => x.attack), axisLabel: { rotate: 45 } },
    yAxis: { type: 'value', max: 1, name: '检出率' },
    series: [{
      type: 'bar',
      data: unk.map(x => ({ value: x.detect_rate, n: x.n })),
      itemStyle: { color: (p) => p.value < 0.1 ? '#e74c3c' : (p.value < 0.5 ? '#e67e22' : '#27ae60') },
    }],
  });

  // 已知类表格
  const known = d.per_known;
  document.getElementById('known-table').innerHTML = `
    <table>
      <thead><tr><th>已知类</th><th>n</th><th>召回</th><th>接受率</th><th>精确率</th><th>F1</th></tr></thead>
      <tbody>${known.map(k => `
        <tr>
          <td>${k.attack}</td><td>${k.n}</td>
          <td>${k.n ? k.rec : '—'}</td>
          <td>${k.n ? k.accept : '—'}</td>
          <td>${k.n ? k.prec : '—'}</td>
          <td>${k.n ? k.f1 : '—'}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;

  // 混淆矩阵
  const cmChart = echarts.init(document.getElementById('chart-confusion'));
  const labels = d.labels;
  const data = [];
  for (let i = 0; i < d.confusion_matrix.length; i++)
    for (let j = 0; j < d.confusion_matrix[i].length; j++)
      data.push([j, i, d.confusion_matrix[i][j]]);
  const maxv = Math.max(...data.map(x => x[2]));
  cmChart.setOption({
    tooltip: { position: 'top', formatter: p => `真:${labels[p.value[1]]}<br/>预:${labels[p.value[0]]}<br/>${p.value[2]}` },
    grid: { left: 120, right: 20, bottom: 120 },
    xAxis: { type: 'category', data: labels, name: '预测', axisLabel: { rotate: 60 } },
    yAxis: { type: 'category', data: labels, name: '真实', axisLabel: { rotate: 30 } },
    visualMap: { min: 0, max: maxv, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
      inRange: { color: ['#f0f0f0', '#3498db', '#1a3a5c'] } },
    series: [{ type: 'heatmap', data: data,
      label: { show: false },
      emphasis: { itemStyle: { shadowBlur: 10 } } }],
  });
}

// 初始化
loadOverview();
loadTestFiles();
