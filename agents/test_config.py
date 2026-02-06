"""验证 Agent 系统配置是否正确

支持三种后端：github / foundry / openai

用法：
    python agents/test_config.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.config import (
    LLM_BACKEND,
    GITHUB_TOKEN,
    GITHUB_MODEL_ID,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    OPENAI_API_KEY,
    OPENAI_MODEL_ID,
)


def check_config():
    """检查配置是否完整"""
    print("=" * 60)
    print(f"配置检查  |  后端模式: {LLM_BACKEND.upper()}")
    print("=" * 60)

    issues = []

    if LLM_BACKEND == "github":
        # ── GitHub Models ──────────────────────────────────
        if not GITHUB_TOKEN:
            issues.append("❌ GITHUB_TOKEN 未配置")
            issues.append("   → https://github.com/settings/personal-access-tokens/new 创建 Fine-grained Token")
        else:
            masked = GITHUB_TOKEN[:8] + "..." + GITHUB_TOKEN[-4:]
            print(f"✅ GITHUB_TOKEN: {masked}")
        print(f"✅ 模型: {GITHUB_MODEL_ID}")
        print(f"✅ 端点: https://models.inference.ai.azure.com")

    elif LLM_BACKEND == "foundry":
        # ── Azure AI Foundry ───────────────────────────────
        if not AZURE_OPENAI_ENDPOINT:
            issues.append("❌ AZURE_OPENAI_ENDPOINT 未配置")
        else:
            print(f"✅ AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT}")
            if not AZURE_OPENAI_ENDPOINT.endswith('/'):
                issues.append("⚠️  Endpoint 应该以 '/' 结尾")
        if not AZURE_OPENAI_API_KEY:
            print("⚠️  AZURE_OPENAI_API_KEY 未配置（将使用 az login 凭据）")
        else:
            print(f"✅ AZURE_OPENAI_API_KEY: {AZURE_OPENAI_API_KEY[:10]}...")
        print(f"✅ AZURE_OPENAI_DEPLOYMENT: {AZURE_OPENAI_DEPLOYMENT}")
        print(f"✅ AZURE_OPENAI_API_VERSION: {AZURE_OPENAI_API_VERSION}")

    elif LLM_BACKEND == "openai":
        # ── OpenAI 直连 ────────────────────────────────────
        if not OPENAI_API_KEY:
            issues.append("❌ OPENAI_API_KEY 未配置")
            issues.append("   → https://platform.openai.com/api-keys 获取")
        else:
            print(f"✅ OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...")
        print(f"✅ 模型: {OPENAI_MODEL_ID}")

    else:
        issues.append(f"❌ 未知的 LLM_BACKEND: {LLM_BACKEND}")
        issues.append("   → 支持的值: github, foundry, openai")

    print("\n" + "=" * 60)

    if issues:
        print("发现问题：")
        for issue in issues:
            print(f"  {issue}")
        print("\n请检查 agents/.env 文件配置")
        return False
    else:
        print("✅ 配置检查通过！")
        return True


def test_client():
    """测试 Chat Client 创建"""
    print("\n" + "=" * 60)
    print("测试 Chat Client 创建")
    print("=" * 60)
    
    try:
        from agents.main import create_chat_client
        client = create_chat_client()
        print(f"✅ Chat Client 创建成功：{type(client).__name__}")
        return True
    except Exception as e:
        print(f"❌ Chat Client 创建失败：{e}")
        return False


def test_workflow():
    """测试工作流构建"""
    print("\n" + "=" * 60)
    print("测试工作流构建")
    print("=" * 60)
    
    try:
        from agents.main import build_optimization_workflow
        workflow = build_optimization_workflow()
        print(f"✅ 工作流构建成功：{type(workflow).__name__}")
        print(f"   包含 6 个专业 Agent")
        return True
    except Exception as e:
        print(f"❌ 工作流构建失败：{e}")
        return False


def main():
    """运行所有测试"""
    print("\n🔍 Web3Quant Agent 系统配置验证\n")
    
    config_ok = check_config()
    if not config_ok:
        print("\n❌ 配置检查未通过，请先修复配置问题。")
        print("\n💡 提示：")
        if LLM_BACKEND == "github" or not LLM_BACKEND:
            print("   【推荐】GitHub Models（免费、无需部署）：")
            print("   1. 复制 agents/.env.github.example 为 agents/.env")
            print("   2. 去 https://github.com/settings/tokens 创建 Token")
            print("   3. 填入 GITHUB_TOKEN")
        else:
            print(f"   当前后端: {LLM_BACKEND}")
            print("   详细说明见 agents/FOUNDRY_SETUP.md")
        sys.exit(1)
    
    client_ok = test_client()
    if not client_ok:
        print("\n❌ Chat Client 创建失败")
        sys.exit(1)
    
    workflow_ok = test_workflow()
    if not workflow_ok:
        print("\n❌ 工作流构建失败")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！系统已就绪。")
    print("=" * 60)
    print("\n下一步：")
    print("  • CLI 模式：python agents/main.py --cli")
    print("  • Server 模式：python agents/main.py --server")
    print("  • VS Code 调试：按 F5 → 选择 'Debug Agent Optimization Server'")
    print()


if __name__ == "__main__":
    main()
