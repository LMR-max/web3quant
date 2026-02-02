#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加密货币数据系统主程序 (Crypto Data System Main)

这是整个系统的主入口，提供了友好的 API 和使用示例。
支持多交易所、多市场类型的数据获取和分析。

Usage:
    python main.py --help
    python main.py fetch --exchange binance --market spot --symbol BTC/USDT
    python main.py analyze --market spot --period 30d
"""

import sys
import os
import argparse
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入模块
try:
    from crypto_data_system import (
        __version__,
        create_fetcher,
        create_data_manager,
        get_logger,
        CacheManager,
        CacheConfig,
    )
except ImportError:
    # 如果作为包内模块运行
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from __init__ import (
        __version__,
        create_fetcher,
        create_data_manager,
        get_logger,
        CacheManager,
        CacheConfig,
    )

# ==================== 配置 ====================

# 支持的交易所
SUPPORTED_EXCHANGES = [
    'binance', 'okx', 'bybit', 'kucoin', 'gate',
    'huobi', 'upbit', 'bithumb', 'kraken', 'coinbase'
]

# 支持的市场类型
SUPPORTED_MARKETS = [
    'spot', 'swap', 'future', 'option', 'margin', 'onchain', 'social'
]

# 默认配置
DEFAULT_CONFIG = {
    'cache_enabled': True,
    'cache_ttl': 300,  # 5分钟
    'max_workers': 10,
    'timeout': 30,
    'output_dir': './data',
}

# ==================== 日志配置 ====================

logger = get_logger('crypto_data_system.main')


# ==================== 主类 ====================

class CryptoDataSystem:
    """加密货币数据系统主类"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化系统
        
        参数:
            config: 配置字典
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.logger = get_logger('CryptoDataSystem')
        
        # 初始化缓存
        self.cache_manager = None
        if self.config['cache_enabled']:
            cache_config = CacheConfig(
                default_ttl=self.config['cache_ttl'],
                cache_dir=os.path.join(self.config['output_dir'], 'cache')
            )
            self.cache_manager = CacheManager(cache_config)
        
        # 缓存 Fetchers
        self.fetchers: Dict[str, Any] = {}
        self.data_managers: Dict[str, Any] = {}
        
        self.logger.info(f"加密货币数据系统初始化完成 (v{__version__})")

    def get_fetcher(self, exchange: str, market_type: str) -> Any:
        """
        获取或创建 Fetcher
        
        参数:
            exchange: 交易所名称
            market_type: 市场类型
            
        返回:
            Fetcher 实例
        """
        key = f"{exchange}_{market_type}"
        
        if key not in self.fetchers:
            self.logger.info(f"创建 Fetcher: {key}")
            fetcher = create_fetcher(
                exchange=exchange,
                market_type=market_type,
                config=self.config,
                cache_manager=self.cache_manager
            )
            self.fetchers[key] = fetcher
        
        return self.fetchers[key]

    def get_data_manager(self, market_type: str, **kwargs) -> Any:
        """
        获取或创建 DataManager
        
        参数:
            market_type: 市场类型
            **kwargs: 传递给 DataManager 的参数
            
        返回:
            DataManager 实例
        """
        key = f"{market_type}_{kwargs.get('exchange', 'default')}"
        
        if key not in self.data_managers:
            self.logger.info(f"创建 DataManager: {key}")
            manager = create_data_manager(
                market_type=market_type,
                cache_manager=self.cache_manager,
                **kwargs
            )
            self.data_managers[key] = manager
        
        return self.data_managers[key]

    def fetch_spot_data(self, 
                       exchange: str,
                       symbols: List[str],
                       timeframe: str = '1h',
                       limit: int = 100) -> Dict[str, List]:
        """
        获取现货数据
        
        参数:
            exchange: 交易所名称
            symbols: 交易对列表
            timeframe: 时间间隔
            limit: 数据条数
            
        返回:
            K线数据字典
        """
        try:
            fetcher = self.get_fetcher(exchange, 'spot')
            results = {}
            
            for symbol in symbols:
                self.logger.info(f"获取 {exchange} {symbol} {timeframe} K线数据")
                ohlcv = fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
                results[symbol] = ohlcv
            
            return results
            
        except Exception as e:
            self.logger.error(f"获取现货数据失败: {e}")
            return {}

    def fetch_spot_market_snapshot(
        self,
        exchange: str,
        symbol: str,
        timeframe: str = '1h',
        ohlcv_limit: int = 200,
        trades_limit: int = 200,
        orderbook_limit: int = 50,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """获取现货某交易对的“全量市场快照”。

        说明：这是对 fetcher 的聚合封装，便于一次性拿到现货公开市场数据。
        include 可选项见 spot_fetcher.CCXTSpotFetcher.fetch_market_snapshot。
        """
        try:
            fetcher = self.get_fetcher(exchange, 'spot')
            if hasattr(fetcher, 'fetch_market_snapshot'):
                return fetcher.fetch_market_snapshot(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv_limit=ohlcv_limit,
                    trades_limit=trades_limit,
                    orderbook_limit=orderbook_limit,
                    include=include,
                )

            # fallback：用现有能力拼一个最小快照
            snapshot: Dict[str, Any] = {
                'exchange': exchange,
                'market_type': 'spot',
                'symbol': symbol,
            }
            snapshot['ticker'] = fetcher.fetch_ticker(symbol)
            snapshot['orderbook'] = fetcher.fetch_orderbook(symbol, limit=orderbook_limit)
            snapshot['trades'] = fetcher.fetch_trades(symbol, limit=trades_limit)
            snapshot['ohlcv'] = fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=ohlcv_limit)
            return snapshot
        except Exception as e:
            self.logger.error(f"获取现货市场快照失败: {e}")
            return {}

    def fetch_swap_data(self,
                       exchange: str,
                       symbols: List[str]) -> Dict[str, Any]:
        """
        获取永续合约数据
        
        参数:
            exchange: 交易所名称
            symbols: 交易对列表
            
        返回:
            资金费率和未平仓合约数据
        """
        try:
            manager = self.get_data_manager(
                'swap',
                exchange=exchange,
                contract_type='linear'
            )
            manager.add_symbols(symbols)
            
            self.logger.info(f"获取 {exchange} 永续合约数据")
            
            results = {
                'funding_rates': manager.fetch_all_funding_rates(),
                'open_interest': manager.fetch_all_open_interest(),
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"获取永续合约数据失败: {e}")
            return {}

    def fetch_onchain_data(self,
                          networks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取链上数据
        
        参数:
            networks: 网络列表（默认为 ['ethereum', 'polygon']）
            
        返回:
            网络统计数据
        """
        try:
            networks = networks or ['ethereum', 'polygon']
            manager = self.get_data_manager('onchain', networks=networks)
            manager.init_fetchers()
            
            self.logger.info(f"获取链上数据: {networks}")
            results = manager.fetch_multi_network_stats()
            
            return results
            
        except Exception as e:
            self.logger.error(f"获取链上数据失败: {e}")
            return {}

    def analyze_market(self,
                      exchange: str,
                      market_type: str,
                      symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析市场数据
        
        参数:
            exchange: 交易所名称
            market_type: 市场类型
            symbols: 交易对列表
            
        返回:
            分析结果
        """
        try:
            manager = self.get_data_manager(market_type, exchange=exchange)
            
            if symbols:
                if hasattr(manager, 'add_symbols'):
                    manager.add_symbols(symbols)
                elif hasattr(manager, 'add_underlying_symbols'):
                    manager.add_underlying_symbols(symbols)
            
            # 获取市场摘要
            if hasattr(manager, 'get_market_summary'):
                summary = manager.get_market_summary()
                self.logger.info(f"{exchange} {market_type} 市场分析完成")
                return summary
            
            return {}
            
        except Exception as e:
            self.logger.error(f"市场分析失败: {e}")
            return {}

    def export_data(self, data: Dict[str, Any], output_file: str) -> bool:
        """
        导出数据到文件
        
        参数:
            data: 数据字典
            output_file: 输出文件路径
            
        返回:
            是否成功
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"数据已导出到: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
            return False


# ==================== CLI 命令 ====================

def cmd_fetch(args):
    """获取数据命令"""
    system = CryptoDataSystem()
    
    exchange = args.exchange.lower()
    market = args.market.lower()
    
    if exchange not in SUPPORTED_EXCHANGES:
        print(f"❌ 不支持的交易所: {exchange}")
        print(f"支持的交易所: {', '.join(SUPPORTED_EXCHANGES)}")
        return
    
    if market not in SUPPORTED_MARKETS:
        print(f"❌ 不支持的市场类型: {market}")
        print(f"支持的市场类型: {', '.join(SUPPORTED_MARKETS)}")
        return
    
    if market == 'spot':
        symbols = args.symbols.split(',')
        data = system.fetch_spot_data(
            exchange=exchange,
            symbols=symbols,
            timeframe=args.timeframe,
            limit=args.limit
        )
        print(f"[OK] 获取 {len(data)} 个交易对的数据")
        
        if args.output:
            system.export_data(
                {k: [str(v) for v in val] if val else [] for k, val in data.items()},
                args.output
            )
    
    elif market == 'swap':
        symbols = args.symbols.split(',')
        data = system.fetch_swap_data(exchange=exchange, symbols=symbols)
        print(f"✅ 获取永续合约数据")
        
        if args.output:
            system.export_data(data, args.output)
    
    elif market == 'onchain':
        data = system.fetch_onchain_data()
        print(f"✅ 获取链上数据")
        
        if args.output:
            system.export_data(data, args.output)
    
    else:
        print(f"⚠️  {market} 市场的数据获取正在开发中")


def cmd_analyze(args):
    """分析数据命令"""
    system = CryptoDataSystem()
    
    market = args.market.lower()
    exchange = args.exchange.lower()
    
    if market not in SUPPORTED_MARKETS:
        print(f"❌ 不支持的市场类型: {market}")
        return
    
    symbols = args.symbols.split(',') if args.symbols else None
    
    result = system.analyze_market(exchange=exchange, market_type=market, symbols=symbols)
    
    print(f"✅ {exchange} {market} 市场分析完成")
    print(json.dumps(result, indent=2, default=str))
    
    if args.output:
        system.export_data(result, args.output)


def cmd_info(args):
    """显示系统信息"""
    print(f"""
╔══════════════════════════════════════════╗
║  加密货币数据系统 (Crypto Data System)    ║
║  Version: {__version__}                          ║
╚══════════════════════════════════════════╝

📊 支持的交易所:
{', '.join(SUPPORTED_EXCHANGES)}

📈 支持的市场类型:
{', '.join(SUPPORTED_MARKETS)}

🔧 功能:
  • 多交易所数据获取
  • K线、行情、订单簿等数据
  • 资金费率、未平仓合约、波动率等衍生品数据
  • 链上交易、地址余额、合约信息等
  • 社交媒体舆情数据
  • 数据缓存和持久化
  • 批量数据分析和导出

📚 使用示例:
  python main.py fetch --exchange binance --market spot --symbols BTC/USDT,ETH/USDT
  python main.py analyze --exchange binance --market swap --symbols BTC/USDT
  python main.py info

💾 数据存储位置:
  ./data_manager_storage/  (数据文件)
  ./data/cache/            (缓存文件)
  ./logs/                  (日志文件)
    """)


# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='加密货币数据系统 - 一站式数据获取和分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 获取现货数据
  %(prog)s fetch --exchange binance --market spot --symbols BTC/USDT,ETH/USDT
  
  # 获取永续合约数据
  %(prog)s fetch --exchange binance --market swap --symbols BTC/USDT
  
  # 分析市场
  %(prog)s analyze --exchange binance --market spot --symbols BTC/USDT
  
  # 显示系统信息
  %(prog)s info
        """
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # fetch 命令
    fetch_parser = subparsers.add_parser('fetch', help='获取数据')
    fetch_parser.add_argument('--exchange', required=True, help='交易所名称')
    fetch_parser.add_argument('--market', required=True, choices=SUPPORTED_MARKETS, help='市场类型')
    fetch_parser.add_argument('--symbols', default='BTC/USDT', help='交易对列表（逗号分隔）')
    fetch_parser.add_argument('--timeframe', default='1h', help='K线时间间隔')
    fetch_parser.add_argument('--limit', type=int, default=100, help='数据条数')
    fetch_parser.add_argument('--output', help='输出文件路径')
    fetch_parser.set_defaults(func=cmd_fetch)
    
    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='分析数据')
    analyze_parser.add_argument('--exchange', default='binance', help='交易所名称')
    analyze_parser.add_argument('--market', required=True, choices=SUPPORTED_MARKETS, help='市场类型')
    analyze_parser.add_argument('--symbols', help='交易对列表（逗号分隔）')
    analyze_parser.add_argument('--output', help='输出文件路径')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # info 命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.set_defaults(func=cmd_info)
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 执行命令
    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"发生错误: {e}")
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
