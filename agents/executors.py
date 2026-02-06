"""Web3Quant 多 Agent 优化系统 - 自定义 Executor

实现 Fan-out/Fan-in 工作流中的 Dispatcher 和 Aggregator。
"""

import json
import logging
from typing import Any

from agent_framework import (
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    WorkflowContext,
    handler,
)

logger = logging.getLogger(__name__)


class DispatchExecutor(Executor):
    """Coordinator：将用户请求 Fan-out 到所有专业 Agent

    接收用户的 ChatMessage，为每个后续 Agent 生成
    AgentExecutorRequest，包含项目上下文。
    """

    def __init__(self, agent_names: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent_names = agent_names

    @handler
    async def dispatch(
        self, message: ChatMessage, ctx: WorkflowContext
    ) -> None:
        """将用户请求广播到所有 Agent"""
        # 延迟导入避免循环依赖
        from agents.context_collector import get_context_for_agent
        from agents.progress import get_tracker

        tracker = get_tracker()

        user_query = message.text if message.text else "请对项目进行全面优化分析"
        logger.info(f"Dispatcher: 收到请求 — {user_query[:80]}...")

        for agent_name in self._agent_names:
            # 为每个 Agent 构建针对性的上下文
            project_ctx = get_context_for_agent(agent_name)

            # 构造发送给 Agent 的消息
            prompt = (
                f"请基于以下项目代码进行分析，回答用户的问题。\n\n"
                f"## 用户请求\n{user_query}\n\n"
                f"## 项目代码上下文\n{project_ctx}"
            )

            if tracker:
                tracker.agent_dispatched(agent_name)

            await ctx.send_message(
                AgentExecutorRequest(
                    messages=[ChatMessage(role="user", text=prompt)],
                ),
                target_id=agent_name,
            )

        logger.info(f"Dispatcher: 已分发到 {len(self._agent_names)} 个 Agent")


class AggregatorExecutor(Executor):
    """Aggregator：Fan-in 汇总所有 Agent 的分析结果

    收集所有 AgentExecutorResponse，按优先级排序并生成综合报告。
    """

    @handler
    async def aggregate(
        self, responses: list[AgentExecutorResponse], ctx: WorkflowContext
    ) -> None:
        """汇总所有 Agent 的分析结果"""
        from agents.progress import get_tracker

        tracker = get_tracker()
        if tracker:
            tracker.aggregator_started()

        logger.info(f"Aggregator: 收到 {len(responses)} 个 Agent 的报告")

        agent_reports: dict[str, str] = {}
        all_findings: list[dict] = []

        for resp in responses:
            agent_name = getattr(resp, "executor_id", "unknown")
            agent_text = ""
            if hasattr(resp, "agent_run_response") and resp.agent_run_response:
                agent_text = resp.agent_run_response.text or ""
            agent_reports[agent_name] = agent_text

            # 尝试解析 JSON 格式的 findings
            findings_count = 0
            try:
                data = json.loads(agent_text)
                findings = data.get("findings", [])
                findings_count = len(findings)
                for f in findings:
                    f["source_agent"] = agent_name
                all_findings.extend(findings)
            except (json.JSONDecodeError, AttributeError):
                # 非 JSON 格式也正常处理
                all_findings.append({
                    "source_agent": agent_name,
                    "issue": agent_text[:500],
                    "priority": 3,
                })
                findings_count = 1

            # 更新进度
            if tracker:
                tracker.agent_done(agent_name, findings_count=findings_count)

        # 按优先级排序
        all_findings.sort(key=lambda x: x.get("priority", 5))

        # 生成优先行动列表
        priority_actions = []
        for f in all_findings[:10]:  # Top 10
            source = f.get("source_agent", "unknown")
            issue = f.get("issue", "N/A")
            suggestion = f.get("suggestion", "")
            priority_actions.append(f"[{source}] P{f.get('priority', '?')}: {issue}" + (f" → {suggestion}" if suggestion else ""))

        # 生成汇总
        summary_parts = [
            "# Web3Quant 项目优化报告\n",
            f"共收到 **{len(responses)}** 个 Agent 的分析结果，发现 **{len(all_findings)}** 个优化点。\n",
            "## 优先行动项（Top 10）\n",
        ]
        for i, action in enumerate(priority_actions, 1):
            summary_parts.append(f"{i}. {action}")

        summary_parts.append("\n## 各 Agent 详细报告\n")
        for agent_name, report in agent_reports.items():
            summary_parts.append(f"### {agent_name}\n{report[:2000]}\n")

        full_report = "\n".join(summary_parts)

        await ctx.yield_output(full_report)

        if tracker:
            tracker.aggregator_done()
            tracker.workflow_finished(total_findings=len(all_findings))

        logger.info("Aggregator: 优化报告已生成")
