#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动加密货币数据系统 Web 应用（完整版）
"""

import os
import sys
import traceback

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == '__main__':
    print("=" * 60)
    print("加密货币数据系统 Web 应用（完整版）")
    print("=" * 60)
    print()
    print("📝 功能特性:")
    print("  ✅ 自由选择时间段 (日期范围选择器)")
    print("  ✅ 完整的数据类型选择 (现货、期货、期权等)")
    print("  ✅ 交易对/合约搜索和选择")
    print("  ✅ 批量数据获取")
    print("  ✅ 数据持久化保存")
    print("  ✅ 数据导出功能")
    print("  ✅ 系统监控和日志")
    print()
    print("🚀 启动服务器...")
    print()
    
    # 导入 Flask 应用
    try:
        from web_app import app, logger
        
        logger.info("=" * 60)
        logger.info("加密货币数据系统 Web 应用启动")
        logger.info("=" * 60)
        logger.info("访问地址: http://localhost:5000")
        logger.info("按 Ctrl+C 停止服务器")
        logger.info("=" * 60)
        
        # 运行应用
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        if isinstance(e, ModuleNotFoundError):
            missing = getattr(e, 'name', None) or str(e)
            print()
            print("可能缺少依赖包:")
            print(f"  - {missing}")
            print()
            print("你可以尝试安装依赖（任选其一）：")
            print("  - conda install flask flask-cors pandas")
            print("  - pip install flask flask-cors pandas")
            print()
        print("详细错误堆栈:")
        traceback.print_exc()
        sys.exit(1)
