# =============================================================================
# 统一评测入口脚本 (Windows PowerShell 5 兼容版本)
# =============================================================================
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1 -Level smoke
#   powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1 -Level core
#   powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1 -Level full
#   powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1 -Only safety
#   powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1 -Skip agent
#   powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1 -Model qwen-plus
# =============================================================================

param(
    [string]$Level = "core",
    [string]$Model = "qwen-turbo",
    [string]$OutputDir = "data/results",
    [int]$MaxSamples = 0,
    [string]$OnlyModules = "",
    [string]$SkipModules = ""
)

$ErrorActionPreference = "Stop"

# 兼容PS5: 用if替代??
if ($env:PYTHON) {
    $PyCommand = $env:PYTHON
} else {
    $PyCommand = "python"
}

$ResultsDir = Join-Path $OutputDir $Level

function Write-Log {
    param([string]$MsgLevel, [string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $color = "White"
    switch ($MsgLevel) {
        "INFO" { $color = "Cyan" }
        "WARN" { $color = "Yellow" }
        "ERROR" { $color = "Red" }
    }
    Write-Host "[$MsgLevel] $timestamp $Message" -ForegroundColor $color
}

function Should-Run {
    param([string]$Module)
    if ($OnlyModules) {
        $mods = $OnlyModules -split ","
        return $mods | ForEach-Object { $_.Trim() } | Where-Object { $_ -eq $Module }
    }
    if ($SkipModules) {
        $skipped = $SkipModules -split ","
        $skippedTrimmed = @()
        foreach ($s in $skipped) { $skippedTrimmed += $s.Trim() }
        return ($skippedTrimmed -notcontains $Module)
    }
    return $true
}

function Run-Safety {
    if (-not (Should-Run "safety")) { return }
    Write-Log "INFO" "运行安全评测..."

    switch ($Level) {
        "smoke" {
            & $PyCommand -m llm_guard_bench.cli safety --level 1 --max-samples 3 --model $Model --output $ResultsDir
        }
        "core" {
            & $PyCommand -m llm_guard_bench.cli safety --level 1 --model $Model --output $ResultsDir
            & $PyCommand -m llm_guard_bench.cli safety --level 2 --model $Model --output $ResultsDir
        }
        "full" {
            & $PyCommand -m llm_guard_bench.cli safety --level 1 --model $Model --output $ResultsDir
            & $PyCommand -m llm_guard_bench.cli safety --level 2 --model $Model --output $ResultsDir
            & $PyCommand -m llm_guard_bench.cli safety --level 3 --model $Model --output $ResultsDir
        }
    }
    Write-Log "INFO" "安全评测完成"
}

function Run-Capability {
    if (-not (Should-Run "capability")) { return }
    Write-Log "INFO" "运行能力评测..."

    $extraArgs = @()
    if ($MaxSamples -gt 0) {
        $extraArgs += "--max-samples"
        $extraArgs += $MaxSamples
    } elseif ($Level -eq "smoke") {
        $extraArgs += "--max-samples"
        $extraArgs += 5
    }

    & $PyCommand -m llm_guard_bench.cli run `
        --config configs/evaluation/capability.yaml `
        --model $Model `
        --output $ResultsDir `
        @extraArgs
    Write-Log "INFO" "能力评测完成"
}

function Run-Performance {
    if (-not (Should-Run "perf")) { return }
    if ($Level -eq "smoke") {
        Write-Log "INFO" "Smoke模式跳过性能基准"
        return
    }
    Write-Log "INFO" "运行性能基准..."

    if ($Level -eq "core") {
        & $PyCommand -m llm_guard_bench.cli perf --quick --model $Model --output $ResultsDir
    } else {
        & $PyCommand -m llm_guard_bench.cli perf --model $Model --output $ResultsDir
    }
    Write-Log "INFO" "性能基准完成"
}

function Run-Agent {
    if (-not (Should-Run "agent")) { return }
    if ($Level -eq "smoke") {
        Write-Log "INFO" "Smoke模式跳过Agent评测"
        return
    }
    Write-Log "INFO" "运行Agent评测..."
    & $PyCommand -m llm_guard_bench.cli agent --task full --model $Model --output $ResultsDir
    Write-Log "INFO" "Agent评测完成"
}

function Run-Experience {
    if (-not (Should-Run "experience")) { return }
    if ($Level -eq "smoke") {
        Write-Log "INFO" "Smoke模式跳过体验评测"
        return
    }
    Write-Log "INFO" "运行体验评测..."
    & $PyCommand -m llm_guard_bench.cli experience --task full --model $Model --output $ResultsDir
    Write-Log "INFO" "体验评测完成"
}

function Run-AITest {
    if (-not (Should-Run "aitest")) { return }
    if ($Level -eq "smoke") {
        Write-Log "INFO" "Smoke模式跳过AI测试工具"
        return
    }
    Write-Log "INFO" "运行AI测试工具..."
    & $PyCommand -m llm_guard_bench.cli aitest --task full --model $Model --output $ResultsDir
    Write-Log "INFO" "AI测试工具完成"
}

function Run-Alignment {
    if (-not (Should-Run "alignment")) { return }
    Write-Log "INFO" "运行Judge对齐验证..."

    $samples = 10
    if ($Level -eq "full") { $samples = 30 }

    & $PyCommand -m llm_guard_bench.cli alignment --task demo --samples $samples --output $ResultsDir
    Write-Log "INFO" "Judge对齐验证完成"
}

function Run-Tests {
    Write-Log "INFO" "运行单元测试..."
    & $PyCommand -m pytest tests/ -v --tb=short
    Write-Log "INFO" "单元测试完成"
}

function Save-Summary {
    $modules = @("safety", "capability", "perf", "agent", "experience", "aitest", "alignment")
    $results = @{}
    foreach ($mod in $modules) {
        $files = Get-ChildItem -Path $ResultsDir -Filter "$mod*.json" -ErrorAction SilentlyContinue
        if ($files) {
            $results[$mod] = $files.Count
        }
    }

    $summary = @{
        level     = $Level
        model     = $Model
        timestamp = (Get-Date).ToString("o")
        results   = $results
    }

    $summaryPath = Join-Path $ResultsDir "summary.json"
    $summary | ConvertTo-Json -Depth 3 | Out-File -FilePath $summaryPath -Encoding utf8
    Write-Log "INFO" "汇总已生成: $summaryPath"
}

# ========== 主流程 ==========
$banner = "`n╔══════════════════════════════════════════════════════════════╗"
$banner += "`n║     LLM Guard Bench - 统一评测入口                        ║"
$banner += "`n╠══════════════════════════════════════════════════════════════╣"
$banner += "`n║  级别: $Level"
$banner += "`n║  模型: $Model"
$banner += "`n║  输出: $ResultsDir"
$banner += "`n╚══════════════════════════════════════════════════════════════╝"
Write-Host $banner -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

Run-Tests
Run-Safety
Run-Capability
Run-Performance
Run-Agent
Run-Experience
Run-AITest
Run-Alignment
Save-Summary

$done = "`n╔══════════════════════════════════════════════════════════════╗"
$done += "`n║  ✅ 评测完成"
$done += "`n╠══════════════════════════════════════════════════════════════╣"
$done += "`n║  报告目录: $ResultsDir"
$done += "`n║  模型: $Model"
$done += "`n╚══════════════════════════════════════════════════════════════╝"
Write-Host $done -ForegroundColor Green
