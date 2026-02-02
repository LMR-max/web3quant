"""
加密货币数据系统 (Crypto Data System)

一个全面的加密货币数据获取、处理和分析系统，支持：
- 多交易所数据获取（现货、期货、合约、期权、杠杆等）
- 链上数据分析（以太坊、Polygon 等）
- 社交媒体舆情数据
- 数据持久化和缓存管理
- 时间序列分析和技术指标计算

Version: 1.0.0
"""

import sys
import os

# 项目基本信息
__version__ = '1.0.0'
__author__ = 'CryptoDataTeam'
__license__ = 'MIT'

# ==================== 工具模块导出 ====================

# 缓存和日志工具
try:
    from .utils.cache import CacheManager, CacheConfig, CacheStrategy, cached
    from .utils.logger import get_logger, LogManager, LogConfig
    from .utils.date_utils import DateRange, DateTimeUtils, split_date_range
    __all__ = [
        'CacheManager', 'CacheConfig', 'CacheStrategy', 'cached',
        'get_logger', 'LogManager', 'LogConfig',
        'DateRange', 'DateTimeUtils', 'split_date_range'
    ]
except ImportError as e:
    print(f"⚠️  工具模块导入失败: {e}")
    __all__ = []

# ==================== 数据模型导出 ====================

try:
    from .data_models import (
        BaseData, OHLCVData, OrderBookData, TradeData,
        VolatilitySurfaceData
    )
    __all__.extend([
        'BaseData', 'OHLCVData', 'OrderBookData', 'TradeData',
        'VolatilitySurfaceData'
    ])
except ImportError as e:
    print(f"⚠️  数据模型导入失败: {e}")

# ==================== 数据聚合器导出 ====================

try:
    from .data_aggregator import (
        MarketSnapshot, MarketDataAggregator, MultiSymbolAggregator
    )
    __all__.extend([
        'MarketSnapshot', 'MarketDataAggregator', 'MultiSymbolAggregator'
    ])
except ImportError as e:
    print(f"⚠️  数据聚合器导入失败: {e}")

# ==================== 存储模块导出 ====================

try:
    from .storage import BaseDataManager, FileDataManager
    __all__.extend(['BaseDataManager', 'FileDataManager'])
except ImportError as e:
    print(f"⚠️  存储模块导入失败: {e}")

# ==================== Fetchers 模块导出 ====================

try:
    from .fetchers import BaseFetcher
    __all__.append('BaseFetcher')
except ImportError as e:
    print(f"⚠️  Fetchers 基类导入失败: {e}")

# ==================== 现货交易所 Fetchers ====================

try:
    from .fetchers.spot_fetcher import CCXTSpotFetcher, SpotDataManager
    __all__.extend(['CCXTSpotFetcher', 'SpotDataManager'])
except ImportError as e:
    print(f"⚠️  SpotFetcher 导入失败: {e}")

# ==================== 永续合约 Fetchers ====================

try:
    from .fetchers.swap_fetcher import CCXTSwapFetcher, SwapDataManager
    __all__.extend(['CCXTSwapFetcher', 'SwapDataManager'])
except ImportError as e:
    print(f"⚠️  SwapFetcher 导入失败: {e}")

# ==================== 期货 Fetchers ====================

try:
    from .fetchers.future_fetcher import CCXTFutureFetcher, FutureDataManager
    __all__.extend(['CCXTFutureFetcher', 'FutureDataManager'])
except ImportError as e:
    print(f"⚠️  FutureFetcher 导入失败: {e}")

# ==================== 期权 Fetchers ====================

try:
    from .fetchers.option_fetcher import CCXTOptionFetcher, OptionDataManager
    __all__.extend(['CCXTOptionFetcher', 'OptionDataManager'])
except ImportError as e:
    print(f"⚠️  OptionFetcher 导入失败: {e}")

# ==================== 杠杆交易 Fetchers ====================

try:
    from .fetchers.margin_fetcher import CCXTMarginFetcher, MarginDataManager
    __all__.extend(['CCXTMarginFetcher', 'MarginDataManager'])
except ImportError as e:
    print(f"⚠️  MarginFetcher 导入失败: {e}")

# ==================== 链上数据 Fetchers ====================

try:
    from .fetchers.onchain_fetcher import OnChainFetcher, EthereumOnChainFetcher, OnChainDataManager
    __all__.extend(['OnChainFetcher', 'EthereumOnChainFetcher', 'OnChainDataManager'])
except ImportError as e:
    print(f"⚠️  OnChainFetcher 导入失败: {e}")

# ==================== 链上地址跟踪 ====================

try:
    from .onchain_tracking import AddressTracker
    __all__.extend(['AddressTracker'])
except ImportError as e:
    print(f"⚠️  AddressTracker 导入失败: {e}")

# ==================== 链上资金流 ====================

try:
    from .onchain_flow import ExchangeFlowAnalyzer
    __all__.extend(['ExchangeFlowAnalyzer'])
except ImportError as e:
    print(f"⚠️  ExchangeFlowAnalyzer 导入失败: {e}")

# ==================== 地址行为分析 ====================

try:
    from .onchain_behavior import AddressBehaviorAnalyzer
    __all__.extend(['AddressBehaviorAnalyzer'])
except ImportError as e:
    print(f"⚠️  AddressBehaviorAnalyzer 导入失败: {e}")

# ==================== 大额异动分析 ====================

try:
    from .onchain_large_moves import LargeMoveAnalyzer
    __all__.extend(['LargeMoveAnalyzer'])
except ImportError as e:
    print(f"⚠️  LargeMoveAnalyzer 导入失败: {e}")

# ==================== 交易结构 / MEV ====================

try:
    from .onchain_mev import MEVAnalyzer
    __all__.extend(['MEVAnalyzer'])
except ImportError as e:
    print(f"⚠️  MEVAnalyzer 导入失败: {e}")

# ==================== Gas 维度分析 ====================

try:
    from .onchain_gas import GasAnalyzer
    __all__.extend(['GasAnalyzer'])
except ImportError as e:
    print(f"⚠️  GasAnalyzer 导入失败: {e}")

# ==================== 协议层分析 ====================

try:
    from .onchain_protocol import ProtocolAnalyzer
    __all__.extend(['ProtocolAnalyzer'])
except ImportError as e:
    print(f"⚠️  ProtocolAnalyzer 导入失败: {e}")

# ==================== 资金循环分析 ====================

try:
    from .onchain_capital_cycle import CapitalCycleAnalyzer
    __all__.extend(['CapitalCycleAnalyzer'])
except ImportError as e:
    print(f"⚠️  CapitalCycleAnalyzer 导入失败: {e}")

# ==================== 代币持仓分布分析 ====================

try:
    from .onchain_token_distribution import TokenDistributionAnalyzer
    __all__.extend(['TokenDistributionAnalyzer'])
except ImportError as e:
    print(f"⚠️  TokenDistributionAnalyzer 导入失败: {e}")

# ==================== NFT 分析 ====================

try:
    from .onchain_nft import NFTAnalyzer
    __all__.extend(['NFTAnalyzer'])
except ImportError as e:
    print(f"⚠️  NFTAnalyzer 导入失败: {e}")

# ==================== 价格关联分析 ====================

try:
    from .onchain_price_relation import PriceRelationAnalyzer
    __all__.extend(['PriceRelationAnalyzer'])
except ImportError as e:
    print(f"⚠️  PriceRelationAnalyzer 导入失败: {e}")

# ==================== 社交媒体 Fetchers ====================

try:
    from .fetchers.social_fetcher import BaseSocialFetcher
    __all__.extend(['BaseSocialFetcher'])
except ImportError as e:
    print(f"⚠️  SocialFetcher 导入失败: {e}")

# ==================== 主数据管理器 ====================
#
# 说明：历史版本里曾尝试从一个误创建的目录（“新建文件夹”）导入 CryptoDataManager。
# 该路径在当前项目结构中不存在，会导致每次启动输出导入警告。
# 为避免噪音与潜在的循环导入风险，这里不再做该导入。

# ==================== 便捷函数 ====================

def create_fetcher(exchange: str, market_type: str, config=None, cache_manager=None):
    """
    工厂函数：创建指定的 Fetcher 实例
    
    参数:
        exchange: 交易所名称 (binance, okx, bybit 等)
        market_type: 市场类型 (spot, swap, future, option, margin, onchain, social)
        config: 配置字典
        cache_manager: 缓存管理器
        
    返回:
        相应的 Fetcher 实例
        
    示例:
        >>> spot_fetcher = create_fetcher('binance', 'spot')
        >>> swap_fetcher = create_fetcher('binance', 'swap')
        >>> onchain_fetcher = create_fetcher('ethereum', 'onchain')
    """
    try:
        if market_type == 'spot':
            from .fetchers.spot_fetcher import CCXTSpotFetcher
            return CCXTSpotFetcher(exchange=exchange, config=config, cache_manager=cache_manager)
        elif market_type == 'swap':
            from .fetchers.swap_fetcher import CCXTSwapFetcher
            return CCXTSwapFetcher(exchange=exchange, config=config, cache_manager=cache_manager)
        elif market_type == 'future':
            from .fetchers.future_fetcher import CCXTFutureFetcher
            return CCXTFutureFetcher(exchange=exchange, config=config, cache_manager=cache_manager)
        elif market_type == 'option':
            from .fetchers.option_fetcher import CCXTOptionFetcher
            return CCXTOptionFetcher(exchange=exchange, config=config, cache_manager=cache_manager)
        elif market_type == 'margin':
            from .fetchers.margin_fetcher import CCXTMarginFetcher
            return CCXTMarginFetcher(exchange=exchange, config=config, cache_manager=cache_manager)
        elif market_type == 'onchain':
            from .fetchers.onchain_fetcher import OnChainFetcher, EthereumOnChainFetcher
            network = (exchange or '').lower()
            if network in ('ethereum', 'eth'):
                return EthereumOnChainFetcher(config=config, cache_manager=cache_manager)
            return OnChainFetcher(network=exchange, config=config, cache_manager=cache_manager)
        elif market_type == 'social':
            from .fetchers.social_fetcher import BaseSocialFetcher
            # exchange 字段在 social 场景下表示平台 (twitter/reddit/telegram)
            return BaseSocialFetcher(platform=exchange, config=config, cache_manager=cache_manager)
        else:
            raise ValueError(f"不支持的市场类型: {market_type}")
    except ImportError as e:
        raise ImportError(f"创建 {market_type} fetcher 失败: {e}")


def create_data_manager(market_type: str, **kwargs):
    """
    工厂函数：创建指定的 DataManager 实例
    
    参数:
        market_type: 市场类型 (spot, swap, future, option, margin, onchain)
        **kwargs: 传递给 DataManager 的参数
        
    返回:
        相应的 DataManager 实例
        
    示例:
        >>> spot_mgr = create_data_manager('spot', exchange='binance')
        >>> swap_mgr = create_data_manager('swap', exchange='binance', contract_type='linear')
        >>> option_mgr = create_data_manager('option', exchange='binance', option_style='european')
    """
    try:
        if market_type == 'spot':
            from .fetchers.spot_fetcher import SpotDataManager
            return SpotDataManager(**kwargs)
        elif market_type == 'swap':
            from .fetchers.swap_fetcher import SwapDataManager
            return SwapDataManager(**kwargs)
        elif market_type == 'future':
            from .fetchers.future_fetcher import FutureDataManager
            return FutureDataManager(**kwargs)
        elif market_type == 'option':
            from .fetchers.option_fetcher import OptionDataManager
            return OptionDataManager(**kwargs)
        elif market_type == 'margin':
            from .fetchers.margin_fetcher import MarginDataManager
            return MarginDataManager(**kwargs)
        elif market_type == 'onchain':
            from .fetchers.onchain_fetcher import OnChainDataManager
            return OnChainDataManager(**kwargs)
        else:
            raise ValueError(f"不支持的市场类型: {market_type}")
    except ImportError as e:
        raise ImportError(f"创建 {market_type} data manager 失败: {e}")


__all__.extend(['create_fetcher', 'create_data_manager'])

# ==================== 版本检查 ====================

def check_version():
    """检查依赖版本"""
    import pandas as pd
    import numpy as np
    print(f"✓ pandas {pd.__version__}")
    print(f"✓ numpy {np.__version__}")
    try:
        import ccxt
        print(f"✓ ccxt {ccxt.__version__}")
    except ImportError:
        print("⚠️  ccxt 未安装")
    try:
        from web3 import Web3
        print(f"✓ web3.py 已安装")
    except ImportError:
        print("⚠️  web3.py 未安装")

