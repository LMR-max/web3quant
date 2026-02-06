"""Web3Quant 多 Agent 优化系统 - 实时进度追踪器

在终端中实时显示每个 Agent 的运行状态、耗时和结果摘要。

状态流转:
  ⏳ QUEUED → 🔄 RUNNING → ✅ DONE / ❌ FAILED
"""

import time
import threading
import sys
from enum import Enum
from dataclasses import dataclass, field


class AgentStatus(Enum):
    QUEUED = "⏳ 排队中"
    DISPATCHED = "📤 已分发"
    RUNNING = "🔄 分析中"
    DONE = "✅ 完成"
    FAILED = "❌ 失败"


@dataclass
class AgentProgress:
    name: str
    status: AgentStatus = AgentStatus.QUEUED
    start_time: float = 0.0
    end_time: float = 0.0
    findings_count: int = 0
    summary: str = ""
    error: str = ""

    @property
    def elapsed(self) -> float:
        if self.start_time == 0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time

    @property
    def elapsed_str(self) -> str:
        e = self.elapsed
        if e == 0:
            return "--"
        return f"{e:.1f}s"


# Agent 中文名映射
AGENT_DISPLAY_NAMES = {
    "data_quality": "数据质量",
    "factor_research": "因子研究",
    "ml_optimizer": "ML 优化",
    "code_review": "代码审查",
    "backtest": "回测验证",
    "risk_control": "风险控制",
}


class ProgressTracker:
    """多 Agent 进度追踪器（线程安全）"""

    def __init__(self, agent_names: list[str]):
        self._lock = threading.Lock()
        self._agents: dict[str, AgentProgress] = {
            name: AgentProgress(name=name) for name in agent_names
        }
        self._workflow_start = 0.0
        self._workflow_end = 0.0
        self._aggregator_status = AgentStatus.QUEUED
        self._final_findings = 0

    # ─── 状态更新方法 ───────────────────────────────────────

    def workflow_started(self):
        self._workflow_start = time.time()
        self._print_header()

    def workflow_finished(self, total_findings: int = 0):
        self._workflow_end = time.time()
        self._final_findings = total_findings
        self._print_final_summary()

    def agent_dispatched(self, name: str):
        with self._lock:
            ap = self._agents.get(name)
            if ap:
                ap.status = AgentStatus.DISPATCHED
                ap.start_time = time.time()
        self._print_status_line(name, "📤 已分发请求")

    def agent_running(self, name: str):
        with self._lock:
            ap = self._agents.get(name)
            if ap:
                ap.status = AgentStatus.RUNNING
        self._print_status_line(name, "🔄 LLM 分析中...")

    def agent_done(self, name: str, findings_count: int = 0, summary: str = ""):
        with self._lock:
            ap = self._agents.get(name)
            if ap:
                ap.status = AgentStatus.DONE
                ap.end_time = time.time()
                ap.findings_count = findings_count
                ap.summary = summary
        self._print_status_line(
            name, f"✅ 完成 — {findings_count} 条发现 ({self._agents[name].elapsed_str})"
        )

    def agent_failed(self, name: str, error: str = ""):
        with self._lock:
            ap = self._agents.get(name)
            if ap:
                ap.status = AgentStatus.FAILED
                ap.end_time = time.time()
                ap.error = error
        self._print_status_line(name, f"❌ 失败: {error[:80]}")

    def aggregator_started(self):
        self._aggregator_status = AgentStatus.RUNNING
        self._print_phase("📊 Aggregator 正在汇总所有 Agent 报告...")

    def aggregator_done(self):
        self._aggregator_status = AgentStatus.DONE

    # ─── 输出方法 ───────────────────────────────────────────

    def _print_header(self):
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║        Web3Quant Multi-Agent Optimization System            ║")
        print("║        6 Agent 并行分析 · Fan-out/Fan-in 架构               ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        self._print_phase("🚀 工作流已启动")
        print()

    def _print_phase(self, msg: str):
        elapsed = ""
        if self._workflow_start > 0:
            e = time.time() - self._workflow_start
            elapsed = f" [{e:.1f}s]"
        print(f"  {msg}{elapsed}")

    def _print_status_line(self, agent_name: str, msg: str):
        display = AGENT_DISPLAY_NAMES.get(agent_name, agent_name)
        padded = f"{display:<8}"
        print(f"    [{padded}] {msg}")

    def _print_final_summary(self):
        total_elapsed = self._workflow_end - self._workflow_start

        print()
        print("┌──────────────────────────────────────────────────────────────┐")
        print("│                      执行摘要                                │")
        print("├──────────────┬───────────┬──────────┬────────────────────────┤")
        print("│ Agent        │ 状态      │ 耗时     │ 发现数                 │")
        print("├──────────────┼───────────┼──────────┼────────────────────────┤")

        for name, ap in self._agents.items():
            display = AGENT_DISPLAY_NAMES.get(name, name)
            status = ap.status.value
            elapsed = ap.elapsed_str
            findings = str(ap.findings_count) if ap.status == AgentStatus.DONE else "-"
            print(f"│ {display:<12} │ {status:<9} │ {elapsed:<8} │ {findings:<22} │")

        print("├──────────────┴───────────┴──────────┴────────────────────────┤")
        print(f"│ 总耗时: {total_elapsed:.1f}s | 总发现: {self._final_findings} 条               │")
        print("└──────────────────────────────────────────────────────────────┘")
        print()

    def get_dashboard(self) -> str:
        """返回当前状态的 dashboard 字符串（用于日志/API）"""
        lines = ["Agent Progress Dashboard:"]
        for name, ap in self._agents.items():
            display = AGENT_DISPLAY_NAMES.get(name, name)
            lines.append(
                f"  {display}: {ap.status.value} "
                f"| elapsed={ap.elapsed_str} "
                f"| findings={ap.findings_count}"
            )
        return "\n".join(lines)


# ─── 全局单例 ────────────────────────────────────────────────
_tracker: ProgressTracker | None = None


def init_tracker(agent_names: list[str]) -> ProgressTracker:
    global _tracker
    _tracker = ProgressTracker(agent_names)
    return _tracker


def get_tracker() -> ProgressTracker | None:
    return _tracker
