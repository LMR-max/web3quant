"""Web3Quant 多 Agent 优化系统 - 项目上下文收集器

扫描项目文件，构造 Agent 可用的精简上下文摘要。
针对 GitHub Models 的 8K token 限制进行优化：只提取文件结构 + 函数/类签名。
"""

import os
import re
from pathlib import Path
from agents.config import PROJECT_ROOT


# 需要扫描的核心模块
SCAN_TARGETS = {
    "crypto_data_system": {
        "description": "数据拉取、缓存与存储",
        "extensions": [".py"],
    },
    "alphagen_style": {
        "description": "DSL 因子表达式、RL 环境",
        "extensions": [".py"],
    },
    "factor_research": {
        "description": "因子研究与验证",
        "extensions": [".py"],
    },
    "machine_learning": {
        "description": "机器学习管线",
        "extensions": [".py"],
    },
    "quant_backtest": {
        "description": "回测引擎",
        "extensions": [".py"],
    },
}

# 顶层文件
TOP_LEVEL_FILES = [
    "FRAMEWORK.md",
]

# 跳过的目录
SKIP_DIRS = {"__pycache__", ".git", "node_modules", "data", "logs", "models", "exports", ".vscode", "agents"}

# 每个 agent 的上下文 token 预算（字符数，约 1 token ≈ 4 chars 中文 ≈ 2 chars）
MAX_CONTEXT_CHARS = 6000


def extract_signatures(filepath: str) -> str:
    """从 Python 文件中提取类名、函数签名和关键导入，不读完整代码"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"  [Error: {e}]"

    signatures = []
    docstring_next = False

    for i, line in enumerate(lines):
        stripped = line.rstrip()

        # 顶层导入（只保留 from xxx import）
        if re.match(r'^from\s+\S+\s+import\s+', stripped):
            signatures.append(stripped)
        # 类定义
        elif re.match(r'^class\s+', stripped):
            signatures.append(stripped)
            docstring_next = True
        # 函数/方法定义
        elif re.match(r'^(\s*)def\s+', stripped):
            signatures.append(stripped)
            docstring_next = True
        # 第一行 docstring（紧跟类/函数之后）
        elif docstring_next and ('"""' in stripped or "'''" in stripped):
            # 提取单行 docstring
            doc = stripped.strip().strip('"\'').strip()
            if doc:
                signatures.append(f"    # {doc}")
            docstring_next = False
        else:
            docstring_next = False

        # 全局变量赋值（常量）
        if re.match(r'^[A-Z_][A-Z0-9_]*\s*=', stripped) and 'import' not in stripped:
            signatures.append(stripped[:120])

    return "\n".join(signatures) if signatures else "  (empty file)"


def scan_module_compact(module_name: str, config: dict) -> str:
    """扫描单个模块，只返回文件列表 + 签名摘要"""
    module_path = Path(PROJECT_ROOT) / module_name
    if not module_path.exists():
        return f"[Module '{module_name}' not found]"

    result_parts = [f"\n## {module_name} — {config['description']}"]

    for root, dirs, files in os.walk(module_path):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS]

        py_files = [f for f in sorted(files) if any(f.endswith(ext) for ext in config["extensions"])]

        for fname in py_files:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, PROJECT_ROOT)
            sigs = extract_signatures(fpath)
            result_parts.append(f"\n### {rel_path}")
            result_parts.append(sigs)

    return "\n".join(result_parts)


def read_file_compact(filepath: str, max_lines: int = 40) -> str:
    """读取文件前 N 行"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
            return "".join(lines)
    except Exception:
        return ""


def collect_project_context(target_domains: list[str] | None = None) -> str:
    """收集精简的项目上下文，供 Agent 分析使用"""
    parts = ["# Web3Quant 项目代码摘要\n"]

    # FRAMEWORK.md 摘要
    fw_path = os.path.join(PROJECT_ROOT, "FRAMEWORK.md")
    if os.path.exists(fw_path):
        parts.append("## 项目架构")
        parts.append(read_file_compact(fw_path, max_lines=30))

    # 收集模块签名
    targets = SCAN_TARGETS if target_domains is None else {
        k: v for k, v in SCAN_TARGETS.items() if k in target_domains
    }

    for module_name, config in targets.items():
        module_ctx = scan_module_compact(module_name, config)
        parts.append(module_ctx)

    full = "\n".join(parts)

    # 截断保护
    if len(full) > MAX_CONTEXT_CHARS:
        full = full[:MAX_CONTEXT_CHARS] + "\n\n... (context truncated to fit token limit)"

    return full


def get_context_for_agent(agent_name: str) -> str:
    """根据 Agent 类型获取针对性上下文"""
    domain_map = {
        "data_quality": ["crypto_data_system"],
        "factor_research": ["alphagen_style", "factor_research"],
        "ml_optimizer": ["machine_learning"],
        "code_review": ["crypto_data_system", "alphagen_style"],
        "backtest": ["quant_backtest", "alphagen_style"],
        "risk_control": ["quant_backtest", "machine_learning"],
    }
    domains = domain_map.get(agent_name)
    return collect_project_context(domains)
