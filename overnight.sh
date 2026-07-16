#!/bin/bash
# 整夜调参脚本（固定种子，可复现）
# 用法: nohup bash overnight.sh > overnight_run.log 2>&1 &
# 醒来看: cat overnight.md

set -e
cd "$(dirname "$0")"
SEED=42   # 固定种子，保证可复现

run() {
  # $1=配置名 $2=额外环境变量
  NAME="$1"; ENVV="$2"
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] 配置: $NAME  ($ENVV)"
  echo "=========================================="
  LOG="log_${NAME}.txt"
  env SEED=$SEED $ENVV uv run python -u train.py > "$LOG" 2>&1 || echo "  $NAME 失败见 $LOG"
  # 提取关键结果到报告
  {
    echo ""
    echo "## $NAME  ($ENVV)"
    echo ""
    echo '```'
    grep -E "未知检测|已知类正确接受率|已知类分类|整体|-> 扫描最优|mailbomb|httptunnel|saint|snmpguess|snmpgetattack|warezmaster|guess_passwd|macro平均|配置:" "$LOG" || echo "(无结果)"
    echo '```'
  } >> overnight.md
  echo "[$(date +%H:%M:%S)] $NAME 完成"
}

search() {
  # 对当前 sig_*.npy 跑权重搜索
  NAME="$1"
  echo "[$(date +%H:%M:%S)] $NAME 权重搜索..."
  uv run python -u search_fine.py > "log_search_${NAME}.txt" 2>&1 || echo "搜索失败"
  {
    echo ""
    echo "### $NAME 最优权重搜索"
    echo '```'
    tail -22 "log_search_${NAME}.txt"
    echo '```'
  } >> overnight.md
}

# 初始化报告
cat > overnight.md <<'EOF'
# 整夜调参结果（固定种子 SEED=42，可复现）

best 基线: 未知F1=0.642, 已知acc=0.892, mailbomb=0.061, TNR=0.910
目的: 在 best 基础上扫超参，找稳定可复现的最优配置

EOF

# 1. 基线复现(确认种子可复现)
run "01_baseline" ""
search "01_baseline"

# 2. label_smoothing 调整
run "02_ls0.02" "CLS_LS=0.02"
run "03_ls0" "CLS_LS=0"

# 3. dropout 调整
run "04_drop0.3" "CLS_DROP=0.3"

# 4. 学习率调整
run "05_lr1e-3" "CLS_LR=0.001"

# 5. 组合: ls0.02 + drop0.3
run "06_ls02_drop03" "CLS_LS=0.02 CLS_DROP=0.3"

echo ""
echo "=========================================="
echo "全部完成! 见 overnight.md"
echo "=========================================="
echo ""
echo "=== 汇总(各配置未知F1对比) ==="
grep -E "^## |未知检测" overnight.md
