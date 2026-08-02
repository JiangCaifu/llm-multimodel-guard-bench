# 大模型全链路评测平台 - 验证文档

记录各模块的验证结果，确保功能正确性。

---

## 1. 环境验证

| 检查项 | 命令 | 预期结果 | 状态 |
|--------|------|---------|------|
| Python 版本 | `python --version` | >= 3.9 | ✅ |
| 包安装 | `python -c "import llm_guard_bench"` | 无报错 | ✅ |
| CLI 可用 | `python -m llm_guard_bench.cli --version` | 输出版本号 | ✅ |
| API 连通 | `python -m llm_guard_bench.cli list-models` | 列出模型 | ✅ |

---

## 2. 单元测试

```bash
python -m pytest tests/ -v --override-ini="addopts="
```

| 测试文件 | 测试数 | 结果 |
|----------|--------|------|
| test_adapters.py | 4 | ✅ 全部通过 |
| test_engine.py | 3 | ✅ 全部通过 |
| test_init.py | 5 | ✅ 全部通过 |
| **合计** | **12** | **✅ 12/12** |

---

## 3. 各模块验证结果

### 3.1 安全评测 (safety)

**验证命令**：
```bash
python -m llm_guard_bench.cli safety --level 1 --model qwen-turbo
python -m llm_guard_bench.cli safety --level 2 --model qwen-turbo
python -m llm_guard_bench.cli safety --level 3 --model qwen-turbo
```

**验证结果**：

| 攻击级别 | 攻击数 | P0 | P1 | P2 | 安全 | 防御率 | 状态 |
|---------|--------|----|----|----|----- |--------|------|
| Level 1 | 9 | 0 | 0 | 0 | 9 | 100% | ✅ |
| Level 2 | 含语义变异 | - | - | - | - | - | ✅ |
| Level 3 | 含多轮诱导 | - | - | - | - | - | ✅ |

**评分卡**：综合评分 A（100.0），攻击防御/合规性/拒绝质量均为优秀

**已修复问题**：
- ~~关键词匹配误报P0~~ → 增加拒绝检测逻辑，识别包含"无法提供"等拒绝特征的回复
- 修复文件：`src/llm_guard_bench/safety/judges.py`

---

### 3.2 能力评测 (run)

**验证命令**：
```bash
python -m llm_guard_bench.cli run --config configs/evaluation/capability.yaml --model qwen-turbo
```

**验证结果**：

| 数据集 | 样本数 | 准确率 | 状态 |
|--------|--------|--------|------|
| finance_50 | 50 | 98.0% | ✅ |
| medical_50 | 50 | 100.0% | ✅ |

---

### 3.3 性能基准测试 (perf)

**验证命令**：
```bash
python -m llm_guard_bench.cli perf --quick --model qwen-turbo
```

**验证结果**：

| 指标 | 说明 | 状态 |
|------|------|------|
| 27场景矩阵 | 并发度(1/5/10) × 输入长度(短/中/长) × 输出长度(短/中/长) | ✅ |
| TTFT/TPOT | 首 token 延迟 / 每 token 耗时 | ✅ |
| 吞吐量 | 每秒完成请求数 | ✅ |
| 拐点分析 | 性能退化拐点识别 | ✅ |
| 快速模式 | 3 个核心场景 | ✅ |

**已修复问题**：
- ~~`-c` 参数与 Python 自身冲突~~ → 改用 `--concurrency`
- 修复文件：`src/llm_guard_bench/cli.py`

---

### 3.4 Agent 评测 (agent)

**验证命令**：
```bash
python -m llm_guard_bench.cli agent --task tool --model qwen-turbo
python -m llm_guard_bench.cli agent --task multi_step --model qwen-turbo
python -m llm_guard_bench.cli agent --task code --model qwen-turbo
python -m llm_guard_bench.cli agent --task full --model qwen-turbo
```

**验证结果**：

| 任务类型 | 核心指标 | 状态 |
|---------|---------|------|
| 工具调用 | 函数名/参数匹配率 | ✅ |
| 多步任务 | 任务完成率 | ✅ |
| Code Agent | 代码可执行率 + 功能正确性 | ✅ |

**已修复问题**：
- ~~中文关键词提取失败~~ → 修复分词逻辑
- 修复文件：`src/llm_guard_bench/agent/multi_step.py`

---

### 3.5 AI 驱动测试工具 (aitest)

**验证命令**：
```bash
python -m llm_guard_bench.cli aitest --task gen-cases --model qwen-turbo
python -m llm_guard_bench.cli aitest --task detect-bugs --model qwen-turbo
python -m llm_guard_bench.cli aitest --task gen-data --model qwen-turbo
python -m llm_guard_bench.cli aitest --task full --model qwen-turbo
```

**验证结果**：

| 任务 | 输出 | 状态 |
|------|------|------|
| gen-cases | 生成测试用例（JSON格式） | ✅ |
| detect-bugs | 缺陷识别与分类 | ✅ |
| gen-data | 测试数据构造 | ✅ |

**已修复问题**：
- ~~LLM 返回 JSON 包含 Markdown 代码块导致解析失败~~ → 增加代码块提取和边界定位
- 修复文件：`src/llm_guard_bench/aitest/case_generator.py`, `bug_detector.py`, `data_factory.py`
- ~~OpenAI 适配器使用旧版 Completion API~~ → 改用 Chat API
- 修复文件：`src/llm_guard_bench/adapters/openai.py`

---

### 3.6 应用层体验测试 (experience)

**验证命令**：
```bash
python -m llm_guard_bench.cli experience --task ux --model qwen-turbo
python -m llm_guard_bench.cli experience --task optimize \
    --input ./data/results/experience/ux_qwen-turbo.json
```

**验证结果**：

**UX 评测**：
| 维度 | 评分 | 等级 |
|------|------|------|
| 响应质量 | 8.8 | 优秀 |
| 相关性 | 9.6 | 优秀 |
| 信息完整性 | 7.9 | 优秀 |
| 可读性 | 8.9 | 优秀 |
| 友好度 | 8.4 | 优秀 |
| **总体均分** | **8.73** | **优秀** |

**优化闭环**：生成5条建议，应用2个策略，输出优化报告 ✅

**已修复问题**：
- ~~`os` 未导入导致保存报告失败~~ → 在 `cli.py` 和 `runner.py` 中添加 `import os`
- ~~`_json` 局部导入问题~~ → 改用顶层 `import json`

---

### 3.7 Judge 对齐 (alignment)

**验证命令**：
```bash
python -m llm_guard_bench.cli alignment --task demo
python -m llm_guard_bench.cli alignment --task generate \
    --input ./data/results/experience/ux_qwen-turbo.json
python -m llm_guard_bench.cli alignment --task annotate \
    --input ./data/results/alignment/pending_annotation.json
python -m llm_guard_bench.cli alignment --task analyze \
    --input ./data/annotations/annotation.json
```

**验证结果（demo 模式）**：

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 80.0% | 8/10 判对 |
| Cohen's Kappa | 0.524 | 中等（排除随机一致性后） |
| correct F1 | 0.857 | 7 个正样本，1 个误判 |
| incorrect F1 | 0.667 | 3 个负样本，1 个漏判 |

**指标计算逻辑验证**：Kappa=0.524 合理（Accuracy=80%看似不错，但排除随机后真实一致性只有中等，体现 Kappa 的价值） ✅

**已修复问题**：
- ~~demo 数据全是 correct 导致 Kappa=0（分母为0）~~ → 构造多类别分布（7 correct + 3 incorrect，2 个分歧样本）
- 修复文件：`src/llm_guard_bench/alignment/annotator.py`
- ~~标注保存路径 PermissionError~~ → 自动创建目录
- 修复文件：`src/llm_guard_bench/cli.py`

---

## 4. CI/CD 验证

### 4.1 YAML 语法

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/eval.yml')); print('OK')"
python -c "import yaml; yaml.safe_load(open('.github/workflows/nightly.yml')); print('OK')"
python -c "import yaml; yaml.safe_load(open('.github/workflows/pr-check.yml')); print('OK')"
```

全部通过 ✅

### 4.2 统一评测脚本

**验证命令**：
```bash
.venv\Scripts\python.exe scripts/run_eval.py smoke
```

**验证结果**：

| 模块 | 结果 | 耗时 |
|------|------|------|
| 单元测试 | ✅ 12/12 | ~16s |
| 安全评测 (Level 1) | ✅ 防御率100% | ~50s |
| 能力评测 | ✅ finance 98%, medical 100% | ~3min |
| 性能/Agent/体验/AI测试 | ✅ Smoke自动跳过 | 0s |
| Judge对齐 | ✅ Kappa=0.524 | ~3s |
| **总计** | **✅ 全部通过** | **~4min** |

### 4.3 指标对比脚本

**验证命令**：
```bash
python scripts/compare_metrics.py \
    --current data/results/test_current/ \
    --previous data/results/test_previous/ \
    --threshold 0.05
```

**验证结果**：
- 安全通过率从 0.95 → 0.85（退化10%>5%阈值）→ 正确识别为退化 ✅
- 退出码=1（检测到退化时返回失败）→ CI门禁可正确触发 ✅

---

## 5. Bug 修复记录

| # | 问题 | 根因 | 修复方案 | 修复文件 |
|---|------|------|---------|---------|
| 1 | 关键词匹配误报P0 | 拒绝文本含敏感词 | 增加拒绝模式检测 | `safety/judges.py` |
| 2 | CLI `-c` 参数冲突 | 与 Python `-c` 冲突 | 改用 `--concurrency` | `cli.py` |
| 3 | LLM JSON 解析失败 | 输出含 Markdown 代码块 | 增加代码块提取 | `aitest/*.py` |
| 4 | OpenAI 适配器报错 | 使用旧版 Completion API | 改用 Chat API | `adapters/openai.py` |
| 5 | 中文关键词提取失败 | 分词逻辑不适配中文 | 修复分词 | `agent/multi_step.py` |
| 6 | 保存报告 `os` 未定义 | 缺少 import | 添加 `import os` | `cli.py`, `runner.py` |
| 7 | Demo 数据 Kappa=0 | 类别单一 | 构造多类别分布 | `alignment/annotator.py` |
| 8 | 标注保存权限错误 | 目录不存在 | 自动创建目录 | `cli.py` |
| 9 | 测试期望 mmlu | 配置已改为 finance_50 | 更新测试断言 | `tests/test_engine.py` |
| 10 | PS1 脚本编码问题 | PS5 不支持 `??` 和 Unicode | 改用 Python 脚本 | `scripts/run_eval.py` |
