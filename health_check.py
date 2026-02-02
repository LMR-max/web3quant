#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速健康检查脚本

一键检查系统是否正常运行
"""

import sys
import os

# Windows 编码修复
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_checks():
    """运行所有健康检查"""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║  加密货币数据系统 - 快速健康检查                              ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # 1. 模块导入检查
    print("1️⃣  检查模块导入...", end=" ")
    try:
        from crypto_data_system import (
            __version__, create_fetcher, create_data_manager,
            CacheManager, FileDataManager
        )
        print("✅")
        results.append(("模块导入", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("模块导入", False))
    
    # 2. Fetcher 创建检查
    print("2️⃣  检查 Fetcher 工厂...", end=" ")
    try:
        from crypto_data_system import create_fetcher
        spot = create_fetcher('binance', 'spot')
        assert spot is not None
        print("✅")
        results.append(("Fetcher 工厂", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("Fetcher 工厂", False))
    
    # 3. DataManager 创建检查
    print("3️⃣  检查 DataManager 工厂...", end=" ")
    try:
        from crypto_data_system import create_data_manager
        mgr = create_data_manager('spot', exchange='binance')
        assert mgr is not None
        print("✅")
        results.append(("DataManager 工厂", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("DataManager 工厂", False))
    
    # 4. FileDataManager 检查
    print("4️⃣  检查 FileDataManager...", end=" ")
    try:
        from crypto_data_system import FileDataManager
        import tempfile
        temp_dir = tempfile.mkdtemp()
        mgr = FileDataManager(root_dir=temp_dir, sub_dir='test')
        mgr.save_dict('test_key', {'test': 'value'})
        loaded = mgr.load_dict('test_key')
        assert loaded == {'test': 'value'}
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅")
        results.append(("FileDataManager", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("FileDataManager", False))
    
    # 5. CLI 检查
    print("5️⃣  检查 CLI 命令...", end=" ")
    try:
        from crypto_data_system.main import CryptoDataSystem
        system = CryptoDataSystem(None)
        assert system is not None
        print("✅")
        results.append(("CLI 系统", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("CLI 系统", False))
    
    # 6. 缓存检查
    print("6️⃣  检查缓存系统...", end=" ")
    try:
        from crypto_data_system import CacheManager, CacheConfig
        config = CacheConfig(enable_memory_cache=True, enable_disk_cache=False)
        cache = CacheManager(config)
        cache.set('test', 'value')
        val = cache.get('test')
        assert val == 'value'
        print("✅")
        results.append(("缓存系统", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("缓存系统", False))
    
    # 7. 数据模型检查
    print("7️⃣  检查数据模型...", end=" ")
    try:
        from crypto_data_system import OHLCVData, OrderBookData, TradeData
        ohlcv = OHLCVData(timestamp=0, symbol='BTC/USDT', timeframe='1h',
                         open=1, high=2, low=0.5, close=1.5, volume=100)
        assert ohlcv.symbol == 'BTC/USDT'
        print("✅")
        results.append(("数据模型", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("数据模型", False))
    
    # 8. 日志系统检查
    print("8️⃣  检查日志系统...", end=" ")
    try:
        from crypto_data_system import get_logger
        logger = get_logger('test')
        assert logger is not None
        print("✅")
        results.append(("日志系统", True))
    except Exception as e:
        print(f"❌ ({e})")
        results.append(("日志系统", False))
    
    # 总结
    print("\n" + "="*60)
    print("  检查结果总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name:<25} {'通过' if result else '失败'}")
    
    print(f"\n总体: {passed}/{total} 项通过", end="")
    
    if passed == total:
        print(" ✅ 系统正常")
        print("\n" + "="*60)
        print("🎉 所有检查通过！系统已就绪，可以开始使用了。")
        print("="*60)
        print("\n💡 建议阅读:")
        print("  1. README.md - 项目概览")
        print("  2. QUICKSTART.md - 快速开始")
        print("  3. python -m crypto_data_system.main info")
        print("  4. python demo.py")
        return 0
    else:
        print(" ⚠️  部分检查未通过")
        print("\n" + "="*60)
        print("❌ 系统存在问题，请检查错误信息")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit_code = run_checks()
    sys.exit(exit_code)
