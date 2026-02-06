# Web3Quant 多 Agent 优化系统

基于 **Microsoft Agent Framework** 构建的 6-Agent 并行工作流，自动化分析与优化 Web3Quant 项目。

## 架构

```
用户请求
   │
   ▼
[Dispatcher] ──fan-out──> [Data Quality Agent]      ── 数据管道优化
                          [Factor Research Agent]   ── 因子研究改进
                          [ML Optimizer Agent]      ── 模型调优建议
                          [Code Review Agent]       ── 代码质量审查
                          [Backtest Agent]          ── 回测验证改进
                          [Risk Control Agent]      ── 风控与逻辑审查
                               │
                          ──fan-in──>
                               │
                          [Aggregator] ──> 优化报告（按优先级排序）
```

## 快速开始（3 分钟）

### 1. 安装依赖

```bash
pip install -r requirements-agenthq.txt
```

### 2. 配置 GitHub Token

```bash
copy agents\.env.github.example agents\.env
```

编辑 `agents/.env`，填入你的 GitHub Personal Access Token (classic)：

```env
LLM_BACKEND=github
GITHUB_TOKEN=ghp_你的Token
GITHUB_MODEL_ID=gpt-4o
```

> Token 获取：<https://github.com/settings/tokens> → Generate new token (classic)
> Pro+ 用户拥有更高配额（150 req/min, 最大 128K tokens/req）

### 3. 验证配置

```bash
python agents/test_config.py
```

### 4. 运行

```bash
# CLI 快速分析
python agents/main.py --cli

# 自定义查询
python agents/main.py --cli --query "分析 crypto_data_system 的缓存策略"

# HTTP Server 模式（支持 Agent Inspector 调试）
python agents/main.py --server
```

## 后端切换

系统支持 3 种 LLM 后端，通过 `.env` 中 `LLM_BACKEND` 切换：

| 后端 | 配置 | 费用 | 适合场景 |
|------|------|------|---------|
| **github** | `GITHUB_TOKEN` | 免费（有速率限制） | 开发测试、Pro+ 用户 |
| **foundry** | `AZURE_OPENAI_ENDPOINT` + `API_KEY` | 按量付费 | 生产环境、高并发 |
| **openai** | `OPENAI_API_KEY` | 按量付费 | 直连 OpenAI |

## 6 个 Agent 的职责

| Agent | 职责 | 扫描模块 |
|-------|------|---------|
| **Data Quality** | 数据完整性、API 效率、缓存策略、跨交易所标准化 | `crypto_data_system/` |
| **Factor Research** | DSL 因子表达式、RL 训练、IC/ICIR 评估、因子多样性 | `alphagen_style/`, `factor_research/` |
| **ML Optimizer** | 特征工程、时序泄露、Walk-Forward、超参优化、集成策略 | `machine_learning/` |
| **Code Review** | 代码异味、架构设计、错误处理、性能与安全 | `crypto_data_system/`, `alphagen_style/` |
| **Backtest** | 滑点建模、Walk-Forward 验证、过拟合检测、回测指标 | `quant_backtest/`, `alphagen_style/` |
| **Risk Control** | Look-ahead bias、生存偏差、止损/仓位、实盘可行性 | `quant_backtest/`, `machine_learning/` |

## 文件结构

```
agents/
  main.py              # 入口：工作流构建 + HTTP Server / CLI
  config.py            # 环境配置（支持 github/foundry/openai）
  prompts.py           # 6 个 Agent 的 System Prompt
  executors.py         # Dispatcher (fan-out) + Aggregator (fan-in)
  context_collector.py # 项目代码签名扫描器（精简上下文）
  progress.py          # 实时进度追踪器（终端可视化）
  test_config.py       # 配置验证脚本
  .env                 # 当前配置（不提交 Git）
  .env.github.example  # GitHub Models 配置模板
  REPORT.md            # 最近一次分析报告
```

## 进度可视化

运行 `--cli` 时会实时显示每个 Agent 的状态：

```
╔══════════════════════════════════════════════════════════════╗
║        Web3Quant Multi-Agent Optimization System            ║
║        6 Agent 并行分析 · Fan-out/Fan-in 架构               ║
╚══════════════════════════════════════════════════════════════╝

  🚀 工作流已启动 [0.0s]

    [数据质量  ] 📤 已分发请求
    [因子研究  ] 📤 已分发请求
    [ML 优化   ] 📤 已分发请求
    ...
  📊 Aggregator 正在汇总所有 Agent 报告... [32.5s]
    [数据质量  ] ✅ 完成 — 9 条发现 (28.3s)
    [因子研究  ] ✅ 完成 — 8 条发现 (30.1s)
    ...

┌──────────────────────────────────────────────────────────────┐
│                      执行摘要                                │
├──────────────┬───────────┬──────────┬────────────────────────┤
│ Agent        │ 状态      │ 耗时     │ 发现数                 │
├──────────────┼───────────┼──────────┼────────────────────────┤
│ 数据质量     │ ✅ 完成    │ 28.3s   │ 9                      │
│ 因子研究     │ ✅ 完成    │ 30.1s   │ 8                      │
│ ...          │           │          │                        │
└──────────────────────────────────────────────────────────────┘
```

## GitHub Actions 自动运行

项目已配置 GitHub Actions 工作流 (`.github/workflows/run-agents.yml`)，支持：

1. **手动触发**：在 GitHub → Actions → Run Multi-Agent Analysis → Run workflow
2. **定时运行**：每周一 UTC 08:00 自动执行
3. **自定义参数**：可选模型（gpt-4o / gpt-4o-mini）和自定义查询

### 配置步骤

1. 在 GitHub 仓库 → Settings → Secrets and variables → Actions
2. 添加 Repository Secret：`MODELS_TOKEN` = 你的 GitHub Classic PAT
3. 推送代码到 `main` 分支
4. 进入 Actions 页面，点击 "Run workflow"

报告会自动上传为 Artifact，也会显示在 Actions 的 Summary 中。

## VS Code 调试（F5）

按 **F5** 选择 `Debug Agent Optimization Server`，自动启动 Agent Server + Agent Inspector。

## 自定义

- **添加新 Agent**：在 `prompts.py` 的 `AGENT_PROMPTS` 中添加 prompt，`main.py` 自动识别
- **修改扫描范围**：编辑 `context_collector.py` 的 `domain_map`
- **切换模型**：修改 `.env` 中的 `GITHUB_MODEL_ID`（如 `gpt-4o-mini`, `o3-mini`）
