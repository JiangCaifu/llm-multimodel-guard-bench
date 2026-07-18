# ========== 大模型全链路评测平台启动脚本 ==========
# 使用方式: powershell -ExecutionPolicy Bypass -File start.ps1

$env:PYTHONPATH = "$PWD\src"
$env:LLM_GUARD_BENCH_ENV = "dev"

Write-Host "`n=== 大模型全链路评测平台 ===" -ForegroundColor Cyan
Write-Host "Version: 0.1.0" -ForegroundColor Gray

# 激活虚拟环境
Write-Host "`n[1/3] 激活虚拟环境..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1
Write-Host "  ✓ 已激活 (.venv)" -ForegroundColor Green

# 验证安装
Write-Host "`n[2/3] 验证安装..." -ForegroundColor Yellow
python -c "import llm_guard_bench; print(f'  ✓ 核心包: llm_guard_bench {llm_guard_bench.__version__}')"
python -c "from llm_guard_bench.adapters import build_adapter, load_model_config; print('  ✓ 模型接入层: OK')"
python -c "from llm_guard_bench.engine import EvaluationConfig, EvaluationRunner; print('  ✓ 评测引擎: OK')"
python -c "import pytest; print('  ✓ 测试框架: pytest OK')"

# 列出配置
Write-Host "`n[3/3] 配置概览:" -ForegroundColor Yellow
Write-Host "  可用模型:" -ForegroundColor White
python -m llm_guard_bench.cli list-models
Write-Host "`n  可用数据集:" -ForegroundColor White
python -m llm_guard_bench.cli list-datasets

Write-Host "`n=== 就绪 ===" -ForegroundColor Cyan
Write-Host "`n常用命令:" -ForegroundColor Gray
Write-Host "  python -m pytest tests/ -v                     # 运行所有测试"
Write-Host "  python -m llm_guard_bench.cli run -c configs/evaluation/capability.yaml  # 运行能力评测"
Write-Host "  python -m llm_guard_bench.cli list-models      # 查看可用模型"
Write-Host "  python -m llm_guard_bench.cli list-datasets    # 查看可用数据集"
Write-Host "`n"
