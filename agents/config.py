"""Web3Quant 多 Agent 优化系统 - 配置模块

支持三种后端模式（通过 LLM_BACKEND 环境变量切换）：
  github   - GitHub Models（免费、无需部署，推荐入门）
  foundry  - Azure AI Foundry（需要部署模型）
  openai   - OpenAI 直连（需要 OpenAI API Key）
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ─── 后端模式 ─────────────────────────────────────────────────
# 可选值: "github" | "foundry" | "openai"
LLM_BACKEND = os.getenv("LLM_BACKEND", "github").lower()

# ─── GitHub Models 配置（默认，最简单） ────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_MODEL_ID = os.getenv("GITHUB_MODEL_ID", "gpt-4o")

# ─── Azure AI Foundry 配置 ────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ─── OpenAI 直连配置 ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o")

# ─── 通用配置 ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
