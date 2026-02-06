"""Web3Quant 多 Agent 优化系统 - 主入口

使用 Microsoft Agent Framework 构建 6-Agent 并行工作流：
  Dispatcher ──fan-out──> [DataQuality, FactorResearch, MLOptimizer,
                           CodeReview, Backtest, RiskControl]
                               ──fan-in──> Aggregator ──> 优化报告

启动方式：
  # HTTP Server（推荐，支持 Agent Inspector 调试）
  python agents/main.py --server

  # CLI 快速测试
  python agents/main.py --cli
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv(override=True)

from agent_framework import (
    ChatAgent,
    ChatMessage,
    WorkflowBuilder,
    WorkflowOutputEvent,
)

from agents.config import (
    LLM_BACKEND,
    GITHUB_TOKEN,
    GITHUB_MODELS_ENDPOINT,
    GITHUB_MODEL_ID,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    OPENAI_API_KEY,
    OPENAI_MODEL_ID,
)
from agents.prompts import AGENT_PROMPTS
from agents.executors import (
    AggregatorExecutor,
    DispatchExecutor,
)
from agents.progress import init_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("web3quant.agents")

# ─── Agent 名称列表 ──────────────────────────────────────────
AGENT_NAMES = list(AGENT_PROMPTS.keys())


def create_chat_client():
    """创建 Chat Client，根据 LLM_BACKEND 环境变量自动选择后端
    
    支持:
      github  - GitHub Models（免费、推荐入门）
      foundry - Azure AI Foundry
      openai  - OpenAI 直连
    """
    if LLM_BACKEND == "github":
        # ── GitHub Models 模式 ──────────────────────────────
        from agent_framework.openai import OpenAIChatClient

        if not GITHUB_TOKEN:
            raise ValueError(
                "未配置 GITHUB_TOKEN。\n"
                "请在 https://github.com/settings/tokens 创建 Personal Access Token，\n"
                "然后在 agents/.env 中设置：GITHUB_TOKEN=ghp_xxxxx"
            )
        logger.info(f"使用 GitHub Models 模式 → 模型: {GITHUB_MODEL_ID}")
        return OpenAIChatClient(
            model_id=GITHUB_MODEL_ID,
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_MODELS_ENDPOINT,
        )

    elif LLM_BACKEND == "foundry":
        # ── Azure AI Foundry 模式 ───────────────────────────
        from agent_framework.azure import AzureOpenAIChatClient

        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "未配置 AZURE_OPENAI_ENDPOINT。\n"
                "请在 agents/.env 中设置你的 Foundry endpoint。"
            )
        if AZURE_OPENAI_API_KEY:
            logger.info(f"使用 Azure AI Foundry (API Key) → 部署: {AZURE_OPENAI_DEPLOYMENT}")
            return AzureOpenAIChatClient(
                api_key=AZURE_OPENAI_API_KEY,
                endpoint=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
            )
        else:
            from azure.identity import DefaultAzureCredential
            logger.info(f"使用 Azure AI Foundry (az login) → 部署: {AZURE_OPENAI_DEPLOYMENT}")
            return AzureOpenAIChatClient(
                credential=DefaultAzureCredential(),
                endpoint=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
            )

    elif LLM_BACKEND == "openai":
        # ── OpenAI 直连模式 ──────────────────────────────────
        from agent_framework.openai import OpenAIChatClient

        if not OPENAI_API_KEY:
            raise ValueError(
                "未配置 OPENAI_API_KEY。\n"
                "请在 https://platform.openai.com/api-keys 获取 API key，\n"
                "然后在 agents/.env 中设置：OPENAI_API_KEY=sk-xxxxx"
            )
        logger.info(f"使用 OpenAI 直连模式 → 模型: {OPENAI_MODEL_ID}")
        return OpenAIChatClient(
            model_id=OPENAI_MODEL_ID,
            api_key=OPENAI_API_KEY,
        )

    else:
        raise ValueError(
            f"未知的 LLM_BACKEND: {LLM_BACKEND}\n"
            f"支持的值: github, foundry, openai"
        )


def create_agent(name: str, chat_client: AzureOpenAIChatClient) -> ChatAgent:
    """创建指定名称的 ChatAgent"""
    instructions = AGENT_PROMPTS[name]
    return ChatAgent(
        chat_client=chat_client,
        name=name,
        instructions=instructions,
    )


def build_optimization_workflow():
    """构建 6-Agent Fan-out/Fan-in 优化工作流

    架构：
        Dispatcher → [6个专业Agent] → Aggregator → OptimizationReport
    """
    chat_client = create_chat_client()

    # 创建 6 个专业 Agent
    agents = {name: create_agent(name, chat_client) for name in AGENT_NAMES}

    # 创建 Executor
    dispatcher = DispatchExecutor(agent_names=AGENT_NAMES, id="dispatcher")
    aggregator = AggregatorExecutor(id="aggregator")

    # 构建工作流
    builder = WorkflowBuilder(
        name="Web3Quant Optimization Workflow",
        description="6-Agent 并行优化分析工作流",
    )

    # 注册 Agent 和 Executor
    for name, agent in agents.items():
        builder.register_agent(lambda a=agent: a, name=name)

    builder.register_executor(lambda: dispatcher, name="dispatcher")
    builder.register_executor(lambda: aggregator, name="aggregator")

    # 设置 Fan-out/Fan-in 边
    builder.set_start_executor("dispatcher")
    builder.add_fan_out_edges("dispatcher", AGENT_NAMES)
    builder.add_fan_in_edges(AGENT_NAMES, "aggregator")

    workflow = builder.build()
    logger.info(f"工作流已构建：Dispatcher → [{', '.join(AGENT_NAMES)}] → Aggregator")
    return workflow


async def run_cli(query: str | None = None) -> None:
    """CLI 模式：直接运行工作流并打印结果"""
    if query is None:
        query = "请对 Web3Quant 项目进行全面的优化分析，涵盖数据质量、因子研究、ML 模型、代码质量、回测与风控。"

    # 初始化进度追踪器
    tracker = init_tracker(AGENT_NAMES)
    tracker.workflow_started()

    logger.info(f"CLI 模式启动，查询: {query[:80]}...")
    workflow = build_optimization_workflow()

    result = await workflow.run(ChatMessage(role="user", text=query))

    # 从事件中提取输出
    outputs = result.get_outputs()
    if outputs:
        print("\n" + "=" * 80)
        for output in outputs:
            print(output)
        print("=" * 80)
    else:
        # 遍历所有事件查找输出
        found = False
        for event in result:
            if isinstance(event, WorkflowOutputEvent):
                if not found:
                    print("\n" + "=" * 80)
                    found = True
                print(event.data)
        if found:
            print("=" * 80)
        else:
            print("⚠️ 未收到优化报告，请检查 Agent 配置。")
            logger.info(f"收到 {len(result)} 个事件: {[type(e).__name__ for e in result]}")


async def run_server() -> None:
    """HTTP Server 模式：启动 Agent Server，支持 Agent Inspector 调试"""
    from azure.ai.agentserver.agentframework import from_agent_framework

    workflow = build_optimization_workflow()
    agent = workflow.as_agent()

    logger.info("启动 HTTP Server 模式...")
    await from_agent_framework(agent).run_async()


def main() -> None:
    parser = argparse.ArgumentParser(description="Web3Quant 多 Agent 优化系统")
    parser.add_argument("--server", action="store_true", help="以 HTTP Server 模式运行（推荐，支持调试）")
    parser.add_argument("--cli", action="store_true", help="以 CLI 模式运行（快速测试）")
    parser.add_argument("--query", type=str, default=None, help="自定义分析请求")
    args = parser.parse_args()

    if args.server or (not args.cli and not args.server):
        # 默认 HTTP Server 模式
        asyncio.run(run_server())
    else:
        asyncio.run(run_cli(args.query))


if __name__ == "__main__":
    main()
