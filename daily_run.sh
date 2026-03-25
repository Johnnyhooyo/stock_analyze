#!/bin/zsh
# daily_run.sh — 每日量化推荐引擎定时运行脚本
#
# 用法：
#   bash daily_run.sh                  # 手动运行
#   bash daily_run.sh --skip-sentiment # 跳过情感分析（更快）
#
# Cron 配置（每个交易日 18:30 HKT，即 UTC+8，cron 使用本地时间）：
#   30 18 * * 1-5 /path/to/stock_analyze/daily_run.sh >> /path/to/stock_analyze/data/logs/daily_cron.log 2>&1
#
# launchd 配置（macOS 推荐，放在 ~/Library/LaunchAgents/）：
#   见项目 docs/daily_run_launchd.plist.example
#
# 注意：脚本会自动激活 .venv 虚拟环境（如存在）

set -euo pipefail

# ── 工作目录 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 日志 ─────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_$(date +%Y%m%d).log"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  每日量化推荐引擎  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# ── 激活虚拟环境（如存在） ────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "  ✅ 已激活虚拟环境: .venv" | tee -a "$LOG_FILE"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "  ✅ 已激活虚拟环境: venv" | tee -a "$LOG_FILE"
else
    echo "  ℹ️  未找到虚拟环境，使用系统 Python" | tee -a "$LOG_FILE"
fi

# ── Python 版本检查 ───────────────────────────────────────────────
PYTHON_CMD="${PYTHON_CMD:-python3}"
echo "  Python: $($PYTHON_CMD --version 2>&1)" | tee -a "$LOG_FILE"

# ── 运行每日推荐引擎 ──────────────────────────────────────────────
ARGS="${@:---watchlist portfolio}"
echo "  参数: $ARGS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

$PYTHON_CMD "$SCRIPT_DIR/daily_run.py" $ARGS 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ 运行成功  $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
else
    echo "  ❌ 运行失败（退出码 $EXIT_CODE）  $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
fi
echo "============================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE

