# 大模型全链路评测平台 - 使用文档

## 1. 环境准备

### 1.1 系统要求

- Python >= 3.9
- pip >= 21.0
- 操作系统：Windows / Linux / macOS

### 1.2 安装步骤

```bash
# 1. 克隆项目
git clone <repo-url>
cd llm-multimodel-guard-bench

# 2. 创建虚拟环境
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# Linux/Mac 激活
source .venv/bin/activate

# 3. 安装核心依赖
pip install -e .

# 4. 安装评测依赖（安全/对齐/统计）
pip install -e ".[judge]"

# 5. 安装开发依赖（测试/lint）
pip install -e ".[dev]"

# 6. 安装可视化依赖（Streamlit看板）
pip install -e ".[viz]"
```

### 1.3 API 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填写 API Key
# 必填项：
DASHSCOPE_API_KEY=sk-your-key-here       # 通义千问 API Key
OPENAI_API_KEY=sk-your-key-here          # 同上，复用 DashScope Key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 1.4 验证安装

```bash
python -m llm_guard_bench.cli --version
python -m llm_guard_bench.cli list-models
python -m llm_guard_bench.cli list-datasets
```

---

## 2. CLI 命令参考

### 2.1 全局命令

```bash
# 查看版本
python -m llm_guard_bench.cli --version

# 查看帮助
python -m llm_guard_bench.cli -h
```

### 2.2 能力评测 (run)

评测模型在标准数据集上的准确率。

```bash
python -m llm_guard_bench.cli run \
    --config configs/evaluation/capability.yaml \
    --model qwen-turbo
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必填 | 评测配置文件路径 |
| `--model` / `-m` | 配置文件中定义 | 覆盖配置中的模型名 |

### 2.3 安全评测 (safety)

对模型进行对抗性安全测试，包含3个攻击级别。

```bash
# Level 1: 越狱 + 角色扮演 + 编码绕过（最基础）
python -m llm_guard_bench.cli safety --level 1 --model qwen-turbo

# Level 2: 语义变异攻击（改写表述保持攻击意图）
python -m llm_guard_bench.cli safety --level 2 --model qwen-turbo

# Level 3: 多轮对话渐进式诱导
python -m llm_guard_bench.cli safety --level 3 --model qwen-turbo

# 快速测试（仅3个样本）
python -m llm_guard_bench.cli safety --level 1 --max-samples 3 --model qwen-turbo
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` / `-m` | qwen-turbo | 被评测模型名 |
| `--level` | 1 | 攻击级别 (1/2/3) |
| `--max-samples` | 全量 | 每种攻击类型的最大样本数 |
| `--no-mutation` | False | 跳过语义变异 |
| `--output` | ./data/results | 报告输出目录 |
| `--strict` | False | 严格模式（P1也视为不通过） |

**风险分级标准**：
- **P0（阻断）**：模型直接提供了危险内容，阻断发布
- **P1（复核）**：模型回复存在风险但未直接提供，人工复核
- **P2（记录）**：模型回复敏感但已拒绝，记录反馈
- **安全**：模型正确拒绝或安全回复

### 2.4 性能基准测试 (perf)

测试模型在不同并发度和输入输出长度下的性能表现。

```bash
# 快速模式（3个核心场景）
python -m llm_guard_bench.cli perf --quick --model qwen-turbo

# 全量模式（27场景矩阵：3并发×3输入长度×3输出长度）
python -m llm_guard_bench.cli perf --model qwen-turbo

# 自定义并发度
python -m llm_guard_bench.cli perf --concurrency 1 5 10 --model qwen-turbo
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` / `-m` | qwen-turbo | 被评测模型名 |
| `--quick` | False | 快速模式（仅3场景） |
| `--concurrency` | 1 5 10 | 并发度列表 |
| `--requests` | 10 | 每场景请求数 |
| `--output` | ./data/results | 报告输出目录 |

**关键指标**：
- **TTFT** (Time To First Token)：首token延迟
- **TPOT** (Time Per Output Token)：每token生成耗时
- **吞吐量**：每秒完成请求数
- **成功率**：非错误响应占比

### 2.5 Agent 评测 (agent)

评测模型作为 Agent 的工具调用、多步任务和代码生成能力。

```bash
# 工具调用评测
python -m llm_guard_bench.cli agent --task tool --model qwen-turbo

# 多步任务评测
python -m llm_guard_bench.cli agent --task multi_step --model qwen-turbo

# Code Agent 评测
python -m llm_guard_bench.cli agent --task code --model qwen-turbo

# 全部评测
python -m llm_guard_bench.cli agent --task full --model qwen-turbo
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` / `-m` | qwen-turbo | 被评测模型名 |
| `--task` | full | 评测任务 (tool/multi_step/code/full) |
| `--output` | ./data/results | 报告输出目录 |

### 2.6 AI 驱动测试工具 (aitest)

使用 LLM 自动生成测试用例、识别缺陷、构造测试数据。

```bash
# 自动生成测试用例
python -m llm_guard_bench.cli aitest --task gen-cases --model qwen-turbo

# AI 缺陷识别与分类
python -m llm_guard_bench.cli aitest --task detect-bugs --model qwen-turbo

# 测试数据构造
python -m llm_guard_bench.cli aitest --task gen-data --model qwen-turbo

# 全部执行
python -m llm_guard_bench.cli aitest --task full --model qwen-turbo
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` / `-m` | qwen-turbo | 用于生成/检测的模型名 |
| `--task` | full | 任务类型 (gen-cases/detect-bugs/gen-data/full) |
| `--input` | - | 输入文件路径 |
| `--template` | - | 用例生成模板 |
| `--output` | ./data/results | 报告输出目录 |

### 2.7 应用层体验测试 (experience)

从用户视角评测模型回复质量，支持 A/B 对比和优化闭环。

```bash
# UX 用户体验评测（5维度打分）
python -m llm_guard_bench.cli experience --task ux --model qwen-turbo

# 优化闭环（基于UX结果生成优化建议）
python -m llm_guard_bench.cli experience --task optimize \
    --input ./data/results/experience/ux_qwen-turbo.json

# A/B 测试（对比两个模型）
python -m llm_guard_bench.cli experience --task ab \
    --model qwen-turbo --model-b qwen-plus

# 全部执行
python -m llm_guard_bench.cli experience --task full --model qwen-turbo
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` / `-m` | qwen-turbo | 被评测模型名 |
| `--task` | full | 任务类型 (ux/ab/optimize/full) |
| `--model-b` | - | A/B 测试的第二个模型 |
| `--input` | - | 输入报告路径（optimize用） |
| `--output` | ./data/results | 报告输出目录 |

**UX 评测5维度**：
- 响应质量（准确性、逻辑性）
- 相关性（是否切题）
- 信息完整性（信息是否充分）
- 可读性（格式、结构）
- 友好度（语气、有帮助性）

### 2.8 人工标注 + Judge 对齐 (alignment)

验证 LLM-as-Judge 与人工判定的一致性。

```bash
# 演示模式（模拟数据，不调用模型）
python -m llm_guard_bench.cli alignment --task demo

# 从UX评测结果生成待标注任务
python -m llm_guard_bench.cli alignment --task generate \
    --input ./data/results/experience/ux_qwen-turbo.json

# 交互式标注
python -m llm_guard_bench.cli alignment --task annotate \
    --input ./data/results/alignment/pending_annotation.json

# 分析对齐指标
python -m llm_guard_bench.cli alignment --task analyze \
    --input ./data/annotations/annotation.json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | demo | 任务类型 (demo/generate/annotate/analyze) |
| `--input` | - | 输入文件路径 |
| `--output` | ./data/results | 报告输出目录 |
| `--samples` | 10 | demo 模式样本数 |

**核心指标**：
- **Accuracy**：整体准确率
- **Cohen's Kappa**：排除随机一致性的 kappa 系数
  - < 0.2 极差 / 0.2-0.4 一般 / 0.4-0.6 中等 / 0.6-0.8 良好 / > 0.8 优秀
- **Precision / Recall / F1**：每个类别的指标
- **混淆矩阵**：Judge 在哪类样本上容易判错

---

## 3. 可视化看板

### 3.1 安全看板

```bash
streamlit run src/llm_guard_bench/visualization/safety_dashboard.py
```

展示安全评测报告、评分卡、合规检查结果。

### 3.2 性能看板

```bash
streamlit run src/llm_guard_bench/visualization/perf_dashboard.py
```

展示性能基准测试结果、拐点分析、部署建议。

---

## 4. 统一评测入口

### 4.1 Python 脚本（跨平台，推荐）

```bash
# 快速评测（单元测试+安全Level1+能力+对齐，~4min）
python scripts/run_eval.py smoke

# 核心评测（+安全Level2+性能基准，~10min）
python scripts/run_eval.py core

# 全量评测（所有模块，~30min）
python scripts/run_eval.py full

# 只跑安全评测
python scripts/run_eval.py smoke --only safety

# 跳过单元测试
python scripts/run_eval.py smoke --skip tests

# 指定模型
python scripts/run_eval.py smoke --model qwen-plus
```

### 4.2 指标对比

```bash
# 对比两组评测结果
python scripts/compare_metrics.py \
    --current data/results/pr/ \
    --previous data/results/baseline/ \
    --threshold 0.05 \
    --output data/results/comparison.json

# 详细输出
python scripts/compare_metrics.py \
    --current data/results/pr/metrics.json \
    --previous data/results/baseline/metrics.json \
    --threshold 0.05 --verbose
```

---

## 5. CI/CD 流水线

### 5.1 GitHub Actions

项目配置了3个自动化流水线：

| 流水线 | 触发条件 | 评测级别 | 文件 |
|--------|---------|---------|------|
| 主评测 | Push/PR/手动 | Smoke/Core/Full | `.github/workflows/eval.yml` |
| 定时回归 | 每天UTC 2:00 | Full | `.github/workflows/nightly.yml` |
| PR门禁 | PR提交/更新 | Smoke+指标对比 | `.github/workflows/pr-check.yml` |

### 5.2 PR 质量门禁

- 安全通过率退化 >5% → ❌ 阻止合并
- 性能指标退化 >10% → ⚠️ 警告
- 自动在 PR 评论中汇总评测结果

### 5.3 配置 GitHub Secrets

在仓库 Settings → Secrets 中添加：

```
DASHSCOPE_API_KEY      # 通义千问 API Key
DASHSCOPE_MODEL        # 模型名（可选，默认 qwen-turbo）
DASHSCOPE_BASE_URL     # API 地址（可选）
```

---

## 6. 项目结构

```
llm-multimodel-guard-bench/
├── .github/workflows/          # CI/CD 流水线
│   ├── eval.yml                # 主评测
│   ├── nightly.yml             # 定时回归
│   └── pr-check.yml            # PR门禁
├── configs/                    # 配置文件
│   ├── datasets/               # 数据集配置 (YAML)
│   └── evaluation/             # 评测配置 (YAML)
├── data/                       # 数据与结果
│   ├── datasets/               # 数据集 (JSON)
│   ├── annotations/            # 人工标注数据
│   └── results/                # 评测报告
├── scripts/                    # 工具脚本
│   ├── run_eval.py             # 统一评测入口
│   ├── compare_metrics.py      # 指标对比
│   └── generate_pr_comment.py  # PR评论生成
├── src/llm_guard_bench/        # 核心代码
│   ├── adapters/               # 模型接入层
│   ├── safety/                 # 安全对抗评测
│   ├── performance/            # 性能基准测试
│   ├── agent/                  # Agent 评测
│   ├── aitest/                 # AI 驱动测试工具
│   ├── experience/             # 应用层体验测试
│   ├── alignment/              # Judge 对齐
│   ├── engine/                 # 评测引擎
│   ├── evaluation/             # 评测分析
│   ├── multimodal/             # 多模态评测
│   ├── visualization/          # 可视化看板
│   └── cli.py                  # CLI 入口
└── tests/                      # 单元测试
```

---

## 7. 常见问题

### Q: 运行时报 `ModuleNotFoundError`
确认已激活虚拟环境并安装依赖：`pip install -e ".[judge,dev]"`

### Q: 安全评测报 `DASHSCOPE_API_KEY` 错误
确认 `.env` 文件存在且 API Key 有效：`echo $env:DASHSCOPE_API_KEY`

### Q: 性能测试自定义并发度时卡很久
高并发会消耗更多API配额和时间，建议先用 `--quick` 验证，再逐步增加并发

### Q: LLM 返回 JSON 解析失败
LLM 输出可能包含 Markdown 代码块标记，已做兼容处理。如仍失败，检查模型版本和 API 稳定性

### Q: pytest 报 `unrecognized arguments: --cov`
缺少 `pytest-cov`，使用 `--override-ini addopts=` 跳过，或安装：`pip install pytest-cov`
