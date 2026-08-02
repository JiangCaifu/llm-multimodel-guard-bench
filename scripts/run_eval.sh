#!/bin/bash
# =============================================================================
# 统一评测入口脚本
# =============================================================================
# 用法:
#   bash scripts/run_eval.sh smoke          # 快速评测（<1min）
#   bash scripts/run_eval.sh core           # 核心评测（<5min）
#   bash scripts/run_eval.sh full           # 全量评测（<30min）
#   bash scripts/run_eval.sh --only safety  # 仅跑安全评测
#   bash scripts/run_eval.sh --skip agent   # 跳过指定模块
#   bash scripts/run_eval.sh --model qwen-plus  # 指定模型
# =============================================================================

set -euo pipefail

# ---------- 配置 ----------
PYTHON=${PYTHON:-python}
LEVEL=${1:-core}
MODEL=${MODEL:-qwen-turbo}
RESULTS_DIR=${RESULTS_DIR:-data/results}
MAX_SAMPLES=${MAX_SAMPLES:-0}

# 模块开关
SKIP_MODULES=""
ONLY_MODULES=""

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --only)
            ONLY_MODULES="$2"
            shift 2
            ;;
        --skip)
            SKIP_MODULES="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        smoke|core|full)
            LEVEL="$1"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ---------- 日志函数 ----------
log_info() { echo "[INFO] $(date '+%H:%M:%S') $*"; }
log_warn() { echo "[WARN] $(date '+%H:%M:%S') $*"; }
log_error() { echo "[ERROR] $(date '+%H:%M:%S') $*"; }

# ---------- 检查模块是否应执行 ----------
should_run() {
    local module=$1
    if [[ -n "$ONLY_MODULES" ]]; then
        [[ "$ONLY_MODULES" == *"$module"* ]] && return 0 || return 1
    fi
    if [[ -n "$SKIP_MODULES" ]]; then
        [[ "$SKIP_MODULES" == *"$module"* ]] && return 1 || return 0
    fi
    return 0
}

# ---------- 构建命令 ----------
build_cmd() {
    local module=$1
    local extra_args=$2
    local cmd="$PYTHON -m llm_guard_bench.cli $module --model $MODEL --output $RESULTS_DIR/$LEVEL"

    if [[ $MAX_SAMPLES -gt 0 ]]; then
        cmd="$cmd --max-samples $MAX_SAMPLES"
    fi
    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi

    echo "$cmd"
}

# ---------- 执行评测 ----------
run_safety() {
    if ! should_run "safety"; then return; fi
    log_info "运行安全评测..."

    case $LEVEL in
        smoke)
            $PYTHON -m llm_guard_bench.cli safety --level 1 --max-samples 3 --model $MODEL --output $RESULTS_DIR/$LEVEL
            ;;
        core)
            $PYTHON -m llm_guard_bench.cli safety --level 1 --model $MODEL --output $RESULTS_DIR/$LEVEL
            $PYTHON -m llm_guard_bench.cli safety --level 2 --model $MODEL --output $RESULTS_DIR/$LEVEL
            ;;
        full)
            $PYTHON -m llm_guard_bench.cli safety --level 1 --model $MODEL --output $RESULTS_DIR/$LEVEL
            $PYTHON -m llm_guard_bench.cli safety --level 2 --model $MODEL --output $RESULTS_DIR/$LEVEL
            $PYTHON -m llm_guard_bench.cli safety --level 3 --model $MODEL --output $RESULTS_DIR/$LEVEL
            ;;
    esac
    log_info "安全评测完成"
}

run_capability() {
    if ! should_run "capability"; then return; fi
    log_info "运行能力评测..."

    local extra=""
    if [[ $MAX_SAMPLES -gt 0 ]]; then
        extra="--max-samples $MAX_SAMPLES"
    elif [[ "$LEVEL" == "smoke" ]]; then
        extra="--max-samples 5"
    fi

    $PYTHON -m llm_guard_bench.cli run \
        --config configs/evaluation/capability.yaml \
        --model $MODEL \
        --output $RESULTS_DIR/$LEVEL \
        $extra
    log_info "能力评测完成"
}

run_performance() {
    if ! should_run "perf"; then return; fi
    log_info "运行性能基准..."

    if [[ "$LEVEL" == "smoke" ]]; then
        log_info "Smoke模式跳过性能基准"
        return
    elif [[ "$LEVEL" == "core" ]]; then
        $PYTHON -m llm_guard_bench.cli perf --quick --model $MODEL --output $RESULTS_DIR/$LEVEL
    else
        $PYTHON -m llm_guard_bench.cli perf --model $MODEL --output $RESULTS_DIR/$LEVEL
    fi
    log_info "性能基准完成"
}

run_agent() {
    if ! should_run "agent"; then return; fi
    if [[ "$LEVEL" == "smoke" ]]; then
        log_info "Smoke模式跳过Agent评测"
        return
    fi
    log_info "运行Agent评测..."
    $PYTHON -m llm_guard_bench.cli agent --task full --model $MODEL --output $RESULTS_DIR/$LEVEL
    log_info "Agent评测完成"
}

run_experience() {
    if ! should_run "experience"; then return; fi
    if [[ "$LEVEL" == "smoke" ]]; then
        log_info "Smoke模式跳过体验评测"
        return
    fi
    log_info "运行体验评测..."
    $PYTHON -m llm_guard_bench.cli experience --task full --model $MODEL --output $RESULTS_DIR/$LEVEL
    log_info "体验评测完成"
}

run_aitest() {
    if ! should_run "aitest"; then return; fi
    if [[ "$LEVEL" == "smoke" ]]; then
        log_info "Smoke模式跳过AI测试工具"
        return
    fi
    log_info "运行AI测试工具..."
    $PYTHON -m llm_guard_bench.cli aitest --task full --model $MODEL --output $RESULTS_DIR/$LEVEL
    log_info "AI测试工具完成"
}

run_alignment() {
    if ! should_run "alignment"; then return; fi
    log_info "运行Judge对齐验证..."

    local samples=10
    if [[ "$LEVEL" == "full" ]]; then
        samples=30
    fi

    $PYTHON -m llm_guard_bench.cli alignment --task demo --samples $samples --output $RESULTS_DIR/$LEVEL
    log_info "Judge对齐验证完成"
}

run_tests() {
    log_info "运行单元测试..."
    $PYTHON -m pytest tests/ -v --tb=short
    log_info "单元测试完成"
}

# ---------- 主流程 ----------
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     LLM Guard Bench - 统一评测入口                        ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  级别: $LEVEL"
    echo "║  模型: $MODEL"
    echo "║  输出: $RESULTS_DIR/$LEVEL"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    mkdir -p "$RESULTS_DIR/$LEVEL"

    # 1. 单元测试
    run_tests

    # 2. 安全评测
    run_safety

    # 3. 能力评测
    run_capability

    # 4. 性能基准
    run_performance

    # 5. Agent评测
    run_agent

    # 6. 体验评测
    run_experience

    # 7. AI测试工具
    run_aitest

    # 8. Judge对齐
    run_alignment

    # ---------- 汇总 ----------
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ✅ 评测完成"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  报告目录: $RESULTS_DIR/$LEVEL"
    echo "║  模型: $MODEL"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # 生成汇总JSON
    $PYTHON -c "
import json, os, glob

summary = {
    'level': '$LEVEL',
    'model': '$MODEL',
    'timestamp': '$(date -Iseconds)',
    'results': {}
}

# 统计各模块结果
for module in ['safety', 'capability', 'perf', 'agent', 'experience', 'aitest', 'alignment']:
    files = glob.glob('$RESULTS_DIR/$LEVEL/' + module + '_*.json')
    if files:
        summary['results'][module] = len(files)

with open('$RESULTS_DIR/$LEVEL/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('汇总已生成: $RESULTS_DIR/$LEVEL/summary.json')
"
}

# 执行
main "$@"
