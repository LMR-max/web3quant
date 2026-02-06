# 快速开始

3 分钟完成配置并运行 6-Agent 分析系统。

---

## 步骤 1：安装依赖

```bash
cd d:\web3quant
pip install -r requirements-agenthq.txt
```

验证：看到 `Successfully installed agent-framework-core...` 即可（WARNING 可忽略）。

---

## 步骤 2：配置 GitHub Token

```bash
copy agents\.env.github.example agents\.env
```

编辑 `agents/.env`：

```env
LLM_BACKEND=github
GITHUB_TOKEN=ghp_你的Token粘贴到这里
GITHUB_MODEL_ID=gpt-4o
```

**获取 Token**：

1. 打开 <https://github.com/settings/tokens>
2. Generate new token → **classic**
3. 不需要勾选任何额外权限
4. 复制 Token 粘贴到 `.env`

> 你是 Pro+ 用户，将获得更高的速率配额。

---

## 步骤 3：验证

```bash
python agents/test_config.py
```

看到以下输出即成功：

```
配置检查  |  后端模式: GITHUB
✅ GITHUB_TOKEN: ghp_...xxxx
✅ 模型: gpt-4o
✅ Chat Client 创建成功：OpenAIChatClient
✅ 工作流构建成功：Workflow
🎉 所有测试通过！系统已就绪。
```

---

## 步骤 4：运行

```bash
# 全面分析（6 个 Agent 并行，约 50 秒）
python agents/main.py --cli

# 自定义分析
python agents/main.py --cli --query "分析 machine_learning 模块的过拟合风险"

# HTTP Server + Agent Inspector（调试用）
python agents/main.py --server
```

---

## 切换模型后端

修改 `agents/.env` 中的 `LLM_BACKEND`：

| 后端 | 设置 | 说明 |
|------|------|------|
| GitHub Models | `LLM_BACKEND=github` | 当前使用，免费 |
| Azure Foundry | `LLM_BACKEND=foundry` | 需要部署模型 |
| OpenAI 直连 | `LLM_BACKEND=openai` | 需要 API Key |

---

## 遇到问题？

| 错误 | 解决方法 |
|------|---------|
| `401 Unauthorized` | Token 无效或过期，重新生成 |
| `413 Payload Too Large` | 切换到 Pro+ 或减小 `MAX_CONTEXT_CHARS` |
| `GITHUB_TOKEN 未配置` | 检查 `.env` 文件是否在 `agents/` 目录下 |
