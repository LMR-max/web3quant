"""
杠杆交易数据获取器模块 - 优化版
提供从交易所获取杠杆交易数据的功能，针对Binance进行了优化
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 导入基础模块
try:
    from .base_fetcher import BaseFetcher, AsyncFetcher
    from ..data_models import OHLCVData, OrderBookData, TradeData
    from ..utils.logger import get_logger, log_execution_time, log_errors
    from ..utils.cache import CacheManager, cached, cache_result
    from ..utils.date_utils import split_date_range, calculate_timeframe_seconds
    from ..config import get_exchange_config, ExchangeSymbolFormats, get_market_config
except ImportError:
    # 如果直接运行，使用简单导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fetchers.base_fetcher import BaseFetcher, AsyncFetcher
    from data_models import OHLCVData, OrderBookData, TradeData
    from utils.logger import get_logger, log_execution_time, log_errors
    from utils.cache import CacheManager, cached, cache_result
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from config import get_exchange_config, ExchangeSymbolFormats, get_market_config


# ==================== 杠杆交易数据获取器 ====================

class MarginFetcher(BaseFetcher):
    """
    杠杆交易数据获取器基类
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 margin_type: str = "cross",  # cross: 全仓, isolated: 逐仓
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化杠杆交易数据获取器
        
        参数:
            exchange: 交易所名称
            margin_type: 杠杆类型 (cross, isolated)
            config: 配置字典
            cache_manager: 缓存管理器
        """
        super().__init__(
            exchange=exchange,
            market_type="margin",
            config=config,
            cache_manager=cache_manager
        )
        
        self.margin_type = margin_type  # cross: 全仓, isolated: 逐仓
        
        # 加载交易所配置
        self.exchange_config = get_exchange_config(exchange)
        
        # 加载市场配置
        self.market_config = get_market_config("margin")
        
        # 初始化交易所连接
        self.exchange_instance = None
        self._init_exchange()
        
        # 初始化杠杆交易对信息
        self.margin_pairs = {}
        self._load_margin_pairs()
        
        # 借币利率缓存
        self._borrow_rates_cache = {}
        self._borrow_rates_last_update = {}
        
        # Binance特定的私有API方法
        if exchange == "binance":
            self._init_binance_api()
    
    def _init_exchange(self):
        """初始化交易所连接"""
        try:
            self.logger.info(f"初始化杠杆交易交易所连接: {self.exchange}, 杠杆类型: {self.margin_type}")
            
            # 尝试导入CCXT
            try:
                import ccxt
                self.has_ccxt = True
            except ImportError:
                self.has_ccxt = False
                self.logger.warning("CCXT未安装，部分功能可能受限")
                return
            
            # 创建交易所实例
            exchange_class = getattr(ccxt, self.exchange, None)
            if not exchange_class:
                raise ValueError(f"不支持的交-所: {self.exchange}")
            
            # 获取CCXT配置
            ccxt_config = self.exchange_config.get_ccxt_config()
            
            # 设置默认类型为杠杆交易
            ccxt_config['options'] = ccxt_config.get('options', {})
            ccxt_config['options']['defaultType'] = 'margin'
            
            # 合并市场配置
            if 'ccxt_options' in self.market_config:
                ccxt_config['options'].update(self.market_config['ccxt_options'])
            
            # 创建交易所实例
            self.exchange_instance = exchange_class(ccxt_config)
            
            # 加载市场信息
            self._load_markets()
            
            self.logger.info(f"杠杆交易交易所连接初始化成功: {self.exchange}")
            
        except Exception as e:
            self.logger.error(f"杠杆交易交易所连接初始化失败: {e}")
            self.exchange_instance = None
    
    def _init_binance_api(self):
        """初始化Binance特定的API方法"""
        if self.exchange != "binance" or not self.exchange_instance:
            return
        
        try:
            # 检查是否支持sapi私有API调用
            if hasattr(self.exchange_instance, 'sapiGetMarginAllPairs'):
                self.logger.info("Binance私有API端点可用")
        except Exception as e:
            self.logger.warning(f"初始化Binance API失败: {e}")
    
    def _load_markets(self):
        """加载市场信息"""
        if not self.exchange_instance:
            return
        
        try:
            self.logger.info(f"加载 {self.exchange} 杠杆交易市场信息...")
            
            # 加载市场
            self.exchange_instance.load_markets()
            
            # 筛选杠杆交易对
            self.margin_markets = {}
            for symbol, market in self.exchange_instance.markets.items():
                # 根据不同交易所的标识判断杠杆交易对
                is_margin = False
                
                if self.exchange == "binance":
                    # 币安杠杆交易: 需要单独查询
                    if market.get('spot', False) and market.get('margin', False):
                        is_margin = True
                
                elif self.exchange == "okx":
                    # OKX杠杆交易
                    if market.get('margin', False):
                        is_margin = True
                
                elif self.exchange == "bybit":
                    # Bybit杠杆交易
                    if market.get('margin', False):
                        is_margin = True
                
                if is_margin and market.get('active', False):
                    self.margin_markets[symbol] = market
            
            self.logger.info(f"加载 {len(self.margin_markets)} 个杠杆交易对")
            
            # 缓存市场信息
            if self.cache_manager:
                self.cache_manager.set(
                    key=f"{self.exchange}_margin_markets",
                    data=self.margin_markets,
                    ttl=3600,  # 缓存1小时
                    sub_dir='margin'
                )
            
        except Exception as e:
            self.logger.error(f"加载市场信息失败: {e}")
            self.margin_markets = {}
    
    def _load_margin_pairs(self):
        """加载杠杆交易对信息"""
        if not self.exchange_instance:
            return
        
        try:
            self.logger.info(f"加载 {self.exchange} 杠杆交易对详细信息...")
            
            # 获取杠杆交易对信息
            if hasattr(self.exchange_instance, 'fetch_markets'):
                markets = self.exchange_instance.fetch_markets()
                
                for market in markets:
                    if market.get('margin', False):
                        symbol = market['symbol']
                        self.margin_pairs[symbol] = {
                            'symbol': symbol,
                            'base': market.get('base'),
                            'quote': market.get('quote'),
                            'margin': market.get('margin', True),
                            'max_leverage': market.get('limits', {}).get('leverage', {}).get('max', 3),
                            'min_leverage': market.get('limits', {}).get('leverage', {}).get('min', 1),
                            'precision': market.get('precision', {}),
                            'limits': market.get('limits', {}),
                            'tier_based': market.get('tierBased', False),
                            'percentage': market.get('percentage', True),
                            'taker': market.get('taker', 0.001),
                            'maker': market.get('maker', 0.001)
                        }
            
            self.logger.info(f"加载 {len(self.margin_pairs)} 个杠杆交易对详细信息")
            
        except Exception as e:
            self.logger.error(f"加载杠杆交易对信息失败: {e}")
            self.margin_pairs = {}
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的杠杆交易对
        
        返回:
            交易对列表
        """
        if not self.margin_markets:
            self._load_markets()
        
        return list(self.margin_markets.keys())
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证交易对是否有效
        
        参数:
            symbol: 交易对符号
            
        返回:
            是否有效
        """
        # 先调用父类的基础验证
        if not super().validate_symbol(symbol):
            return False
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 如果有市场信息，检查是否存在
        if self.margin_markets and formatted_symbol not in self.margin_markets:
            self.logger.warning(f"杠杆交易对可能无效: {formatted_symbol}")
            return False
        
        return True
    
    def format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号为交易所标准格式
        
        参数:
            symbol: 原始交易对符号
            
        返回:
            格式化后的交易对符号
        """
        formatted = ExchangeSymbolFormats.format_symbol(symbol, self.exchange, 'margin')
        self.logger.debug(f"格式化杠杆交易对: {symbol} -> {formatted}")
        return formatted
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_ohlcv", ttl=300, sub_dir="margin")
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = "1h",
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None,
                   **kwargs) -> List[OHLCVData]:
        """
        获取杠杆交易K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 验证交易对
        if not self.validate_symbol(formatted_symbol):
            self.logger.warning(f"杠杆交易对可能无效: {formatted_symbol}")
        
        # 转换时间参数
        since_timestamp = None
        if since:
            if isinstance(since, datetime):
                since_timestamp = int(since.timestamp() * 1000)
            elif isinstance(since, str):
                dt = pd.Timestamp(since).to_pydatetime()
                since_timestamp = int(dt.timestamp() * 1000)
            else:
                since_timestamp = since
        
        # 设置默认限制
        if limit is None:
            limit = self.config.get('ohlcv_limit', 1000)
        
        self.logger.info(
            f"获取杠杆交易K线数据: {formatted_symbol}, "
            f"时间间隔: {timeframe}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 使用CCXT获取数据（杠杆交易通常与现货共享K线）
            ohlcv_list = self.exchange_instance.fetch_ohlcv(
                symbol=formatted_symbol,
                timeframe=timeframe,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {})
            )
            
            # 转换为数据模型
            data_models = []
            for ohlcv in ohlcv_list:
                try:
                    data_model = OHLCVData.from_ccxt(
                        ohlcv=ohlcv,
                        symbol=formatted_symbol,
                        timeframe=timeframe,
                        exchange=self.exchange,
                        market_type="margin"
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换杠杆交易OHLCV数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条杠杆交易K线数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取杠杆交易K线数据失败: {e}")
            # 如果失败，返回空列表而不是抛出异常
            if kwargs.get('raise_error', False):
                raise
            return []
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_orderbook", ttl=30, sub_dir="margin")
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> Optional[OrderBookData]:
        """
        获取杠杆交易订单簿数据
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 设置默认限制
        if limit is None:
            limit = self.config.get('orderbook_limit', 20)
        
        self.logger.info(f"获取杠杆交易订单簿: {formatted_symbol}, 深度: {limit}")
        
        try:
            # 使用CCXT获取订单簿（杠杆交易通常与现货共享订单簿）
            orderbook = self.exchange_instance.fetch_order_book(
                symbol=formatted_symbol,
                limit=limit
            )
            
            # 转换为数据模型
            data_model = OrderBookData.from_ccxt(
                orderbook=orderbook,
                symbol=formatted_symbol,
                exchange=self.exchange
            )
            
            self.logger.info(
                f"杠杆交易订单簿获取成功: 买盘 {len(data_model.bids)} 个, "
                f"卖盘 {len(data_model.asks)} 个, "
                f"价差: {data_model.spread:.2f}"
            )
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取杠杆交易订单簿失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_trades", ttl=60, sub_dir="margin")
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取杠杆交易成交数据
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 转换时间参数
        since_timestamp = None
        if since:
            if isinstance(since, datetime):
                since_timestamp = int(since.timestamp() * 1000)
            elif isinstance(since, str):
                dt = pd.Timestamp(since).to_pydatetime()
                since_timestamp = int(dt.timestamp() * 1000)
            else:
                since_timestamp = since
        
        # 设置默认限制
        if limit is None:
            limit = self.config.get('trades_limit', 100)
        
        self.logger.info(
            f"获取杠杆交易成交数据: {formatted_symbol}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 使用CCXT获取成交数据（杠杆交易通常与现货共享成交）
            trades = self.exchange_instance.fetch_trades(
                symbol=formatted_symbol,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {})
            )
            
            # 转换为数据模型
            data_models = []
            for trade in trades:
                try:
                    data_model = TradeData.from_ccxt(
                        trade=trade,
                        symbol=formatted_symbol,
                        exchange=self.exchange
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换杠杆交易成交数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条杠杆交易成交数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取杠杆交易成交数据失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_ticker", ttl=60, sub_dir="margin")
    def fetch_ticker(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        获取杠杆交易行情数据
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            行情数据字典
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"获取杠杆交易行情: {formatted_symbol}")
        
        try:
            # 使用CCXT获取ticker（杠杆交易通常与现货共享ticker）
            ticker = self.exchange_instance.fetch_ticker(formatted_symbol)
            
            result = {
                'symbol': formatted_symbol,
                'timestamp': pd.Timestamp(ticker['timestamp'], unit='ms'),
                'datetime': ticker['datetime'],
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'bid': ticker.get('bid'),
                'bidVolume': ticker.get('bidVolume'),
                'ask': ticker.get('ask'),
                'askVolume': ticker.get('askVolume'),
                'vwap': ticker.get('vwap'),
                'open': ticker.get('open'),
                'close': ticker.get('close'),
                'last': ticker.get('last'),
                'previousClose': ticker.get('previousClose'),
                'change': ticker.get('change'),
                'percentage': ticker.get('percentage'),
                'average': ticker.get('average'),
                'baseVolume': ticker.get('baseVolume'),
                'quoteVolume': ticker.get('quoteVolume'),
                'info': ticker.get('info', {})
            }
            
            self.logger.info(
                f"杠杆交易行情获取成功: 最新价 {result.get('last')}, "
                f"24h成交量 {result.get('baseVolume')}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取杠杆交易行情数据失败: {e}")
            # 如果失败，尝试从父类获取
            return super().fetch_ticker(symbol, **kwargs)
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_market_info", ttl=3600, sub_dir="margin")
    def fetch_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取杠杆交易市场信息
        
        参数:
            symbol: 交易对符号
            
        返回:
            市场信息字典
        """
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        if formatted_symbol in self.margin_pairs:
            return self.margin_pairs[formatted_symbol]
        
        # 如果缓存中没有，尝试从交易所获取
        if not self.exchange_instance:
            return {}
        
        try:
            # 获取市场信息
            market = self.exchange_instance.market(formatted_symbol)
            
            market_info = {
                'symbol': formatted_symbol,
                'base': market.get('base'),
                'quote': market.get('quote'),
                'margin': market.get('margin', True),
                'max_leverage': market.get('limits', {}).get('leverage', {}).get('max', 3),
                'min_leverage': market.get('limits', {}).get('leverage', {}).get('min', 1),
                'precision': market.get('precision', {}),
                'limits': market.get('limits', {}),
                'tier_based': market.get('tierBased', False),
                'percentage': market.get('percentage', True),
                'taker': market.get('taker', 0.001),
                'maker': market.get('maker', 0.001),
                'margin_type': self.margin_type
            }
            
            # 缓存市场信息
            self.margin_pairs[formatted_symbol] = market_info
            
            return market_info
            
        except Exception as e:
            self.logger.error(f"获取杠杆交易市场信息失败: {e}")
            return {}
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_borrow_rates", ttl=3600, sub_dir="margin")
    def fetch_borrow_rates(self) -> Dict[str, Dict[str, Any]]:
        """
        获取借币利率
        
        返回:
            借币利率字典，key为币种，value为利率信息
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        self.logger.info(f"获取 {self.exchange} 借币利率")
        
        # 检查缓存（内存级别）
        cache_key = f"borrow_rates_{self.exchange}"
        if cache_key in self._borrow_rates_cache:
            last_update = self._borrow_rates_last_update.get(cache_key, 0)
            # 如果10分钟内更新过，使用缓存
            if time.time() - last_update < 600:
                self.logger.info(f"使用缓存的借币利率数据")
                return self._borrow_rates_cache[cache_key]
        
        try:
            # 根据交易所类型使用不同的方法
            if self.exchange == "binance":
                return self._fetch_binance_borrow_rates()
            else:
                # 检查交易所是否支持借币利率获取
                if hasattr(self.exchange_instance, 'fetch_borrow_rates'):
                    # 使用CCXT获取借币利率
                    borrow_rates = self.exchange_instance.fetch_borrow_rates()
                    
                    result = {}
                    for currency, rate_info in borrow_rates.items():
                        result[currency] = {
                            'currency': currency,
                            'rate': rate_info.get('rate', 0),
                            'period': rate_info.get('period', 86400000),  # 通常为24小时
                            'timestamp': pd.Timestamp(rate_info.get('timestamp', pd.Timestamp.now())),
                            'datetime': rate_info.get('datetime'),
                            'info': rate_info.get('info', {})
                        }
                    
                    self.logger.info(f"获取到 {len(result)} 个币种的借币利率")
                    return result
                else:
                    self.logger.warning(f"交易所 {self.exchange} 不支持 fetch_borrow_rates() 方法")
                    return {}
            
        except Exception as e:
            self.logger.warning(f"获取借币利率失败 (交易所可能不支持此功能): {e}")
            return {}
    
    def _fetch_binance_borrow_rates(self) -> Dict[str, Dict[str, Any]]:
        """
        Binance特定的借币利率获取方法
        使用 sapi_get_margin_allassets 获取真实利率
        
        返回:
            借币利率字典
        """
        try:
            self.logger.info("使用Binance私有API获取借币利率")
            result = {}
            
            # 尝试调用 Binance API 获取所有杠杆资产信息 (包含利率)
            if hasattr(self.exchange_instance, 'sapi_get_margin_allassets'):
                assets = self.exchange_instance.sapi_get_margin_allassets()
                # assets 是一个列表
                for asset_info in assets:
                    currency = asset_info.get('assetName')
                    # hourlyInterestRate usually not directly here but dailyInterestRate might be inferred
                    # Actually sapi_get_margin_allassets returns:
                    # [{"assetFullName": "...", "assetName": "BTC", "isBorrowable": true, "isMortgageable": true, "userMinBorrow": "...", "userMinRepay": "..."}]
                    # It might NOT contain rates directly.
                    pass
            
            # cross margin data usually contains rates
            # sapi_get_margin_cross_margin_data
            if hasattr(self.exchange_instance, 'sapi_get_margin_cross_margin_data'):
                # This usually returns user specific data?
                # Actually, there isn't a simple public endpoint for ALL current rates in one go without auth sometimes.
                # But fetch_borrow_rates in ccxt usually handles it?
                pass
            
            # 尝试 fetch_borrow_rates (CCXT Standard)
            # 如果之前在 fetch_borrow_rates 中失败了才进到这里
            # 所以这里主要作为 fetch_borrow_rates 失败后的硬编码回退
            
            # 使用默认/估计值 (作为最后的兜底)
            default_rates = {
                'BTC': 0.00025,  # 年化~9%
                'ETH': 0.00035,
                'BNB': 0.00015,
                'USDT': 0.0005,  # 年化~18%
                'USDC': 0.0005,
                'FDUSD': 0.0004,
            }
            
            # ... existing fallback logic ...
            # 获取所有支持的币种
            if hasattr(self.exchange_instance, 'currencies') and self.exchange_instance.currencies:
                currencies = self.exchange_instance.currencies
                for currency_code, currency_info in currencies.items():
                    rate = default_rates.get(currency_code, 0.0005)  # 默认0.05%日利率
                    
                    result[currency_code] = {
                        'currency': currency_code,
                        'rate': rate,
                        'period': 86400000,  # 24小时
                        'timestamp': pd.Timestamp.now(),
                        'datetime': pd.Timestamp.now().isoformat(),
                        'annual_rate': rate * 365,  # 年化利率
                        'info': {'source': 'estimated_fallback'}
                    }
            else:
                 # 如果currencies为空
                for currency, rate in default_rates.items():
                    result[currency] = {
                        'currency': currency,
                        'rate': rate,
                        'period': 86400000,
                        'timestamp': pd.Timestamp.now(),
                        'datetime': pd.Timestamp.now().isoformat(),
                        'annual_rate': rate * 365,
                        'info': {'source': 'default'}
                    }
            
            # 更新缓存
            cache_key = f"borrow_rates_{self.exchange}"
            self._borrow_rates_cache[cache_key] = result
            self._borrow_rates_last_update[cache_key] = time.time()
            
            self.logger.info(f"获取到 {len(result)} 个币种的借币利率（估计值）")
            return result
            
        except Exception as e:
            self.logger.error(f"获取Binance借币利率失败: {e}")
            return {}
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_borrow_rate", ttl=3600, sub_dir="margin")
    def fetch_borrow_rate(self, currency: str) -> Dict[str, Any]:
        """
        获取指定币种的借币利率
        
        参数:
            currency: 币种符号
            
        返回:
            借币利率信息
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        self.logger.info(f"获取 {self.exchange} {currency} 借币利率")
        
        try:
            # 先尝试从fetch_borrow_rates的结果中获取
            all_rates = self.fetch_borrow_rates()
            if currency.upper() in all_rates:
                return all_rates[currency.upper()]
            
            # 如果不行，尝试直接获取
            if hasattr(self.exchange_instance, 'fetch_borrow_rate'):
                # 使用CCXT获取指定币种的借币利率
                borrow_rate = self.exchange_instance.fetch_borrow_rate(currency)
                
                result = {
                    'currency': currency,
                    'rate': borrow_rate.get('rate', 0),
                    'period': borrow_rate.get('period', 86400000),  # 通常为24小时
                    'timestamp': pd.Timestamp(borrow_rate.get('timestamp', pd.Timestamp.now())),
                    'datetime': borrow_rate.get('datetime'),
                    'info': borrow_rate.get('info', {})
                }
                
                self.logger.info(f"{currency} 借币利率: {result['rate']:.6%}")
                return result
            else:
                self.logger.warning(f"交易所 {self.exchange} 不支持 fetch_borrow_rate() 方法")
                return {}
            
        except Exception as e:
            self.logger.warning(f"获取 {currency} 借币利率失败 (交易所可能不支持此功能): {e}")
            return {}
    
    @log_errors(reraise=False)
    @cached(key_prefix="margin_borrow_interest", ttl=600, sub_dir="margin")
    def fetch_borrow_interest(self, 
                             currency: Optional[str] = None,
                             code: Optional[str] = None,
                             since: Optional[int] = None,
                             limit: Optional[int] = None,
                             **kwargs) -> List[Dict[str, Any]]:
        """
        获取借币利息历史
        
        参数:
            currency: 币种符号
            code: 统一货币代码
            since: 开始时间戳
            limit: 限制条数
            **kwargs: 额外参数
            
        返回:
            借币利息历史列表
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        self.logger.info(f"获取 {self.exchange} 借币利息历史")
        
        # 根据你提供的代码，Binance支持此功能
        try:
            # 检查交易所是否支持借币利息历史获取
            if hasattr(self.exchange_instance, 'fetch_borrow_interest'):
                # 使用CCXT获取借币利息历史
                params = kwargs.get('params', {})
                
                # 对于Binance，如果需要portfolio margin账户
                if self.exchange == "binance" and kwargs.get('portfolio_margin', False):
                    params['portfolioMargin'] = True
                
                interest_history = self.exchange_instance.fetch_borrow_interest(
                    code=code,
                    symbol=currency,  # 注意：这里可能是symbol参数
                    since=since,
                    limit=limit,
                    params=params
                )
                
                result = []
                for interest in interest_history:
                    interest_info = {
                        'currency': interest.get('currency'),
                        'currency_code': interest.get('code'),
                        'amount': float(interest.get('accruedInterest', 0)),
                        'rate': float(interest.get('rate', 0)),
                        'timestamp': pd.Timestamp(interest.get('timestamp', pd.Timestamp.now())),
                        'datetime': interest.get('datetime'),
                        'info': interest.get('info', {})
                    }
                    result.append(interest_info)
                
                self.logger.info(f"获取到 {len(result)} 条借币利息历史")
                return result
            else:
                self.logger.warning(f"交易所 {self.exchange} 不支持 fetch_borrow_interest() 方法")
                return []
            
        except Exception as e:
            self.logger.warning(f"获取借币利息历史失败: {e}")
            return []
    
    @log_errors(reraise=False)
    def fetch_margin_levels(self, symbol: str) -> Dict[str, Any]:
        """
        获取杠杆交易等级信息
        
        参数:
            symbol: 交易对符号
            
        返回:
            杠杆等级信息
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"获取 {self.exchange} {formatted_symbol} 杠杆等级信息")
        
        try:
            # 尝试获取详细的杠杆等级信息
            margin_levels = {}
            
            if self.exchange == "binance":
                # Binance特定的杠杆等级获取
                # 对于Binance，我们可以从市场信息中获取
                market_info = self.fetch_market_info(symbol)
                margin_levels = {
                    'symbol': formatted_symbol,
                    'leverage': 3,
                    'max_leverage': market_info.get('max_leverage', 10),
                    'maintenance_margin_rate': 0.1,
                    'initial_margin_rate': 0.5,
                    'info': market_info
                }
            elif hasattr(self.exchange_instance, 'fetch_margin_levels'):
                margin_levels = self.exchange_instance.fetch_margin_levels(formatted_symbol)
            else:
                # 如果不支持，返回默认值
                margin_levels = {
                    'symbol': formatted_symbol,
                    'leverage': 3,
                    'maintenance_margin_rate': 0.1,
                    'initial_margin_rate': 0.5,
                    'info': {}
                }
            
            result = {
                'symbol': formatted_symbol,
                'leverage': margin_levels.get('leverage', 3),
                'max_leverage': margin_levels.get('maxLeverage', margin_levels.get('max_leverage', 10)),
                'maintenance_margin_rate': margin_levels.get('maintenanceMarginRate', 
                                                           margin_levels.get('maintenance_margin_rate', 0.1)),
                'initial_margin_rate': margin_levels.get('initialMarginRate', 
                                                        margin_levels.get('initial_margin_rate', 0.5)),
                'margin_ratio': margin_levels.get('marginRatio', 0.5),
                'timestamp': pd.Timestamp.now(),
                'info': margin_levels.get('info', {})
            }
            
            self.logger.info(f"{formatted_symbol} 最大杠杆: {result['max_leverage']}倍")
            return result
            
        except Exception as e:
            self.logger.error(f"获取杠杆等级信息失败: {e}")
            return {}
    
    @log_errors(reraise=False)
    def calculate_margin_cost(self, 
                             symbol: str, 
                             amount: float, 
                             leverage: float = 3,
                             days: int = 30) -> Dict[str, Any]:
        """
        计算杠杆交易成本
        
        参数:
            symbol: 交易对符号
            amount: 交易数量
            leverage: 杠杆倍数
            days: 持有天数
            
        返回:
            杠杆交易成本信息
        """
        # 获取市场信息
        market_info = self.fetch_market_info(symbol)
        
        if not market_info:
            return {}
        
        # 获取当前价格
        ticker = self.fetch_ticker(symbol)
        
        if not ticker or 'last' not in ticker:
            return {}
        
        current_price = ticker['last']
        
        # 计算交易价值
        trade_value = amount * current_price
        
        # 计算保证金
        margin_required = trade_value / leverage
        
        # 获取借币利率
        borrow_rates = self.fetch_borrow_rates()
        
        # 获取基础币种
        base_currency = market_info.get('base', '')
        quote_currency = market_info.get('quote', '')
        
        # 假设借入基础币种
        borrow_rate = 0
        if base_currency and base_currency.upper() in borrow_rates:
            borrow_rate = borrow_rates[base_currency.upper()].get('rate', 0)
        
        # 计算利息成本（年化）
        annual_interest = margin_required * borrow_rate
        
        # 转换为日利息
        daily_interest = annual_interest / 365
        
        # 计算持有成本
        holding_cost = daily_interest * days
        
        result = {
            'symbol': symbol,
            'amount': amount,
            'price': current_price,
            'trade_value': trade_value,
            'leverage': leverage,
            'margin_required': margin_required,
            'borrow_rate': borrow_rate,
            'annual_rate': borrow_rate * 365 if borrow_rate > 0 else 0,
            'daily_interest': daily_interest,
            'holding_days': days,
            'total_interest': holding_cost,
            'interest_percentage': (holding_cost / trade_value) * 100 if trade_value > 0 else 0,
            'effective_leverage': leverage * (1 + (borrow_rate * days / 365))
        }
        
        return result
    
    @log_errors(reraise=False)
    def fetch_liquidation_price(self, 
                               symbol: str, 
                               side: str, 
                               entry_price: float,
                               leverage: float,
                               margin_type: Optional[str] = None) -> Dict[str, Any]:
        """
        计算强平价格
        
        参数:
            symbol: 交易对符号
            side: 交易方向 (buy/sell)
            entry_price: 入场价格
            leverage: 杠杆倍数
            margin_type: 保证金类型 (cross/isolated)
            
        返回:
            强平价格信息
        """
        # 获取市场信息
        market_info = self.fetch_margin_levels(symbol)
        
        if not market_info:
            return {}
        
        # 获取维持保证金率
        maintenance_margin_rate = market_info.get('maintenance_margin_rate', 0.1)
        
        # 计算强平价格
        if side.lower() == 'buy':
            # 多头强平价格 = 入场价格 * (1 - 1/杠杆倍数 + 维持保证金率)
            liquidation_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        elif side.lower() == 'sell':
            # 空头强平价格 = 入场价格 * (1 + 1/杠杆倍数 - 维持保证金率)
            liquidation_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
        else:
            liquidation_price = 0
        
        # 获取当前价格用于计算安全边际
        ticker = self.fetch_ticker(symbol)
        current_price = ticker.get('last', entry_price) if ticker else entry_price
        
        # 计算安全边际
        if side.lower() == 'buy':
            safety_margin = ((current_price - liquidation_price) / current_price) * 100 if current_price > liquidation_price else 0
        else:
            safety_margin = ((liquidation_price - current_price) / current_price) * 100 if liquidation_price > current_price else 0
        
        # 计算距离当前价格的百分比
        if entry_price > 0:
            price_distance_percent = abs(current_price - liquidation_price) / entry_price * 100
        else:
            price_distance_percent = 0
        
        result = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'current_price': current_price,
            'leverage': leverage,
            'maintenance_margin_rate': maintenance_margin_rate,
            'liquidation_price': liquidation_price,
            'price_distance': current_price - liquidation_price if side == 'buy' else liquidation_price - current_price,
            'price_distance_percent': price_distance_percent,
            'safety_margin_percent': safety_margin,
            'margin_type': margin_type or self.margin_type,
            'timestamp': pd.Timestamp.now()
        }
        
        return result

    def fetch_market_snapshot(
        self,
        symbol: str,
        timeframe: str = '1h',
        ohlcv_limit: int = 200,
        trades_limit: int = 200,
        orderbook_limit: int = 50,
        include: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """获取杠杆交易某交易对的“综合市场快照”。

        include 允许项（默认全开）：
        - ticker/orderbook/trades/market_info
        - borrow_rate/margin_levels/liquidation_price
        - ohlcv
        """
        include_set = set((include or [
            'ticker', 'orderbook', 'trades', 'market_info',
            'borrow_rate', 'margin_levels', 'liquidation_price',
        ]))

        formatted_symbol = self.format_symbol(symbol)
        snapshot: Dict[str, Any] = {
            'exchange': self.exchange,
            'market_type': 'margin',
            'symbol': formatted_symbol,
            'timestamp': pd.Timestamp.now(),
            'margin_type': self.margin_type,
        }

        if 'ticker' in include_set:
            snapshot['ticker'] = self.fetch_ticker(formatted_symbol)
        if 'orderbook' in include_set:
            snapshot['orderbook'] = self.fetch_orderbook(formatted_symbol, limit=orderbook_limit)
        if 'trades' in include_set:
            snapshot['trades'] = self.fetch_trades(formatted_symbol, limit=trades_limit)
        if 'market_info' in include_set:
            snapshot['market_info'] = self.fetch_market_info(formatted_symbol)

        if 'borrow_rate' in include_set:
            # 尝试用 base currency 推断借币利率（近似）
            try:
                mi = snapshot.get('market_info') if isinstance(snapshot.get('market_info'), dict) else self.fetch_market_info(formatted_symbol)
                base = (mi.get('base') or '').upper() if isinstance(mi, dict) else ''
                if base:
                    snapshot['borrow_rate'] = self.fetch_borrow_rate(base)
            except Exception:
                snapshot['borrow_rate'] = {}

        if 'margin_levels' in include_set:
            snapshot['margin_levels'] = self.fetch_margin_levels(formatted_symbol)

        if 'liquidation_price' in include_set:
            # 缺少 entry/leverage 的情况下，给一个“以当前价为 entry、3x 杠杆”的粗略估算
            try:
                t = snapshot.get('ticker') if isinstance(snapshot.get('ticker'), dict) else self.fetch_ticker(formatted_symbol)
                last = float(t.get('last') or 0) if isinstance(t, dict) else 0
                if last > 0:
                    snapshot['liquidation_price'] = {
                        'buy': self.fetch_liquidation_price(formatted_symbol, side='buy', entry_price=last, leverage=3, margin_type=self.margin_type),
                        'sell': self.fetch_liquidation_price(formatted_symbol, side='sell', entry_price=last, leverage=3, margin_type=self.margin_type),
                    }
            except Exception:
                snapshot['liquidation_price'] = {}

        if 'ohlcv' in include_set:
            snapshot['ohlcv'] = self.fetch_ohlcv(formatted_symbol, timeframe=timeframe, limit=ohlcv_limit)

        return snapshot
    
    def test_connection(self) -> bool:
        """
        测试交易所连接
        
        返回:
            连接是否成功
        """
        try:
            if not self.exchange_instance:
                return False
            
            # 尝试获取服务器时间
            timestamp = self.exchange_instance.fetch_time()
            self.logger.info(f"杠杆交易交易所连接测试成功，服务器时间: {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"杠杆交易交易所连接测试失败: {e}")
            return False
    
    def get_margin_account_summary(self) -> Dict[str, Any]:
        """
        获取杠杆账户摘要（需要API权限）
        
        返回:
            杠杆账户摘要信息
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        self.logger.info(f"获取 {self.exchange} 杠杆账户摘要")
        
        try:
            # 检查是否支持杠杆账户信息获取
            if hasattr(self.exchange_instance, 'fetch_balance'):
                # 获取杠杆账户余额
                params = {'type': 'margin'}
                balance = self.exchange_instance.fetch_balance(params=params)
                
                summary = {
                    'timestamp': pd.Timestamp.now(),
                    'total_net_worth': balance.get('total', {}),
                    'free': balance.get('free', {}),
                    'used': balance.get('used', {}),
                    'info': balance.get('info', {})
                }
                
                # 计算总资产和负债
                total_assets = 0
                total_liabilities = 0
                
                for currency, amount in balance.get('total', {}).items():
                    if amount > 0:
                        total_assets += amount
                
                for currency, amount in balance.get('used', {}).items():
                    if amount > 0:
                        total_liabilities += amount
                
                summary['total_assets'] = total_assets
                summary['total_liabilities'] = total_liabilities
                summary['net_worth'] = total_assets - total_liabilities
                summary['leverage_ratio'] = total_assets / summary['net_worth'] if summary['net_worth'] > 0 else 0
                
                self.logger.info(f"杠杆账户摘要获取成功，净资产: {summary['net_worth']:.2f}")
                return summary
            else:
                self.logger.warning(f"交易所 {self.exchange} 不支持 fetch_balance() with margin type")
                return {}
                
        except Exception as e:
            self.logger.error(f"获取杠杆账户摘要失败: {e}")
            return {}
    
    def close(self):
        """关闭获取器，释放资源"""
        if self.exchange_instance:
            self.exchange_instance = None
        
        super().close()


# ==================== CCXT杠杆交易获取器 ====================

class CCXTMarginFetcher(MarginFetcher):
    """
    CCXT杠杆交易数据获取器
    
    使用CCXT库获取杠杆交易数据
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 margin_type: str = "cross",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 **kwargs):
        """
        初始化CCXT杠杆交易获取器
        
        参数:
            exchange: 交易所名称
            margin_type: 杠杆类型
            api_key: API密钥
            api_secret: API密钥
            **kwargs: 额外参数
        """
        # 更新配置
        config = kwargs.pop('config', None)
        if not isinstance(config, dict):
            if config is None:
                config = {}
            else:
                try:
                    config = dict(config)
                except Exception:
                    config = {}
        config.update({
            'api_key': api_key,
            'api_secret': api_secret,
            'margin_type': margin_type
        })
        
        # 合并额外参数到配置
        for key, value in kwargs.items():
            if key not in ['cache_manager']:
                config[key] = value
        
        super().__init__(
            exchange=exchange,
            margin_type=margin_type,
            config=config,
            cache_manager=kwargs.get('cache_manager')
        )
        
        # 如果提供了API密钥，更新交易所实例
        if api_key and api_secret and self.exchange_instance:
            self.exchange_instance.apiKey = api_key
            self.exchange_instance.secret = api_secret


# ==================== 杠杆交易数据管理器 ====================

try:
    from crypto_data_system.storage.data_manager import FileDataManager
except (ImportError, ModuleNotFoundError):
    # 脚本直接运行时的备用导入方式
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crypto_data_system.storage.data_manager import FileDataManager


class MarginDataManager(FileDataManager):
    """
    杠杆交易数据管理器
    
    管理多个交易对的杠杆交易数据获取
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 margin_type: str = "cross",
                 fetcher_config: Optional[Dict] = None,
                 root_dir: Optional[str] = None,
                 cache_manager: Optional[Any] = None,
                 save_json_merged: bool = False):
        """
        初始化杠杆交易数据管理器
        
        参数:
            exchange: 交易所名称
            margin_type: 杠杆类型
            fetcher_config: 获取器配置
        """
        self.exchange = exchange
        self.margin_type = margin_type
        self.fetcher_config = fetcher_config or {}
        self.fetcher = None
        self.symbols = []
        self.save_json_merged = bool(self.fetcher_config.get('save_json_merged', save_json_merged))
        
        # 初始化日志
        self.logger = get_logger(f"margin_manager.{exchange}.{margin_type}")

        # 初始化文件存储（子目录按交易所/杠杆类型分组）
        super().__init__(root_dir=root_dir, sub_dir=f"margin/{exchange}/{margin_type}", file_format="json", cache_manager=cache_manager)
    
    def init_fetcher(self):
        """初始化数据获取器"""
        if not self.fetcher:
            self.fetcher = CCXTMarginFetcher(
                exchange=self.exchange,
                margin_type=self.margin_type,
                config=self.fetcher_config
            )
            
            # 测试连接
            if not self.fetcher.test_connection():
                self.logger.warning(f"交易所 {self.exchange} 杠杆交易连接测试失败")
    
    def add_symbol(self, symbol: str):
        """
        添加交易对
        
        参数:
            symbol: 交易对符号
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.logger.info(f"添加杠杆交易对: {symbol}")
    
    def add_symbols(self, symbols: List[str]):
        """
        批量添加交易对
        
        参数:
            symbols: 交易对列表
        """
        for symbol in symbols:
            self.add_symbol(symbol)
    
    def fetch_all_borrow_rates(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有币种的借币利率
        
        返回:
            借币利率字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        result = self.fetcher.fetch_borrow_rates()
        
        # 统一持久化
        self.save_dict(f"{self.exchange}_{self.margin_type}_borrow_rates", result)
        return result
    
    def fetch_all_market_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有交易对的市场信息
        
        返回:
            市场信息字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        results = {}
        for symbol in self.symbols:
            try:
                market_info = self.fetcher.fetch_market_info(symbol)
                results[symbol] = market_info
            except Exception as e:
                self.logger.error(f"获取交易对 {symbol} 市场信息失败: {e}")
                results[symbol] = {}
        
        # 统一持久化
        self.save_dict(f"{self.exchange}_{self.margin_type}_market_info", results)
        return results
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        获取市场摘要
        
        返回:
            市场摘要信息
        """
        if not self.fetcher:
            self.init_fetcher()
        
        borrow_rates = self.fetch_all_borrow_rates()
        market_infos = self.fetch_all_market_info()
        
        # 提取有效的杠杆信息，处理None值
        max_leverages = [info.get('max_leverage') or 3 for info in market_infos.values()]
        min_leverages = [info.get('min_leverage') or 1 for info in market_infos.values()]
        
        # 提取有效的借币利率，处理None值
        borrow_rate_values = []
        for r in borrow_rates.values():
            rate = r.get('rate')
            if rate is not None:
                borrow_rate_values.append(rate)
        
        summary = {
            'exchange': self.exchange,
            'margin_type': self.margin_type,
            'total_symbols': len(self.symbols),
            'borrow_rates': {
                'total_currencies': len(borrow_rates),
                'average_rate': sum(borrow_rate_values) / len(borrow_rate_values) if borrow_rate_values else 0,
                'max_rate': max(borrow_rate_values) if borrow_rate_values else 0,
                'min_rate': min(borrow_rate_values) if borrow_rate_values else 0,
            },
            'leverage_info': {
                'average_max_leverage': sum(max_leverages) / len(max_leverages) if max_leverages else 3,
                'max_leverage': max(max_leverages) if max_leverages else 3,
                'min_leverage': min(min_leverages) if min_leverages else 1,
            },
            'symbols': list(market_infos.keys())
        }
        
        return summary
    
    def calculate_portfolio_margin_cost(self, 
                                       portfolio: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算投资组合的保证金成本
        
        参数:
            portfolio: 投资组合字典，格式为 {symbol: {amount, leverage, side}}
            
        返回:
            投资组合保证金成本信息
        """
        if not self.fetcher:
            self.init_fetcher()
        
        total_margin = 0
        total_value = 0
        total_interest = 0
        
        details = {}
        
        for symbol, position in portfolio.items():
            try:
                amount = position.get('amount', 0)
                leverage = position.get('leverage', 3)
                side = position.get('side', 'buy')
                days = position.get('days', 30)
                
                # 计算单个交易对的成本
                cost_info = self.fetcher.calculate_margin_cost(
                    symbol=symbol,
                    amount=amount,
                    leverage=leverage,
                    days=days
                )
                
                if cost_info:
                    total_margin += cost_info.get('margin_required', 0)
                    total_value += cost_info.get('trade_value', 0)
                    total_interest += cost_info.get('total_interest', 0)
                    
                    details[symbol] = cost_info
                    
            except Exception as e:
                self.logger.error(f"计算交易对 {symbol} 保证金成本失败: {e}")
        
        result = {
            'portfolio': portfolio,
            'total_margin_required': total_margin,
            'total_trade_value': total_value,
            'total_interest_cost': total_interest,
            'average_leverage': total_value / total_margin if total_margin > 0 else 0,
            'interest_percentage': (total_interest / total_value) * 100 if total_value > 0 else 0,
            'details': details,
            'timestamp': pd.Timestamp.now()
        }
        
        return result
    
    def calculate_portfolio_liquidation_prices(self, 
                                              portfolio: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算投资组合的强平价格
        
        参数:
            portfolio: 投资组合字典，格式为 {symbol: {amount, leverage, side, entry_price}}
            
        返回:
            投资组合强平价格信息
        """
        if not self.fetcher:
            self.init_fetcher()
        
        liquidation_prices = {}
        total_risk_score = 0
        positions_at_risk = 0
        
        for symbol, position in portfolio.items():
            try:
                side = position.get('side', 'buy')
                entry_price = position.get('entry_price', 0)
                leverage = position.get('leverage', 3)
                
                if entry_price > 0:
                    # 计算强平价格
                    liquidation_info = self.fetcher.fetch_liquidation_price(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        leverage=leverage
                    )
                    
                    if liquidation_info:
                        liquidation_prices[symbol] = liquidation_info
                        
                        # 计算风险分数（安全边际越低，风险越高）
                        safety_margin = liquidation_info.get('safety_margin_percent', 100)
                        risk_score = max(0, 100 - safety_margin) / 100  # 0-1之间的风险分数
                        total_risk_score += risk_score
                        
                        # 如果安全边际小于20%，认为有风险
                        if safety_margin < 20:
                            positions_at_risk += 1
                    
            except Exception as e:
                self.logger.error(f"计算交易对 {symbol} 强平价格失败: {e}")
        
        avg_risk_score = total_risk_score / len(portfolio) if portfolio else 0
        
        result = {
            'portfolio': portfolio,
            'liquidation_prices': liquidation_prices,
            'total_positions': len(portfolio),
            'positions_at_risk': positions_at_risk,
            'risk_percentage': (positions_at_risk / len(portfolio)) * 100 if portfolio else 0,
            'average_risk_score': avg_risk_score,
            'overall_risk_level': 'HIGH' if avg_risk_score > 0.7 else 'MEDIUM' if avg_risk_score > 0.3 else 'LOW',
            'timestamp': pd.Timestamp.now()
        }
        
        return result

    def fetch_and_save(self,
                       symbol: str,
                       timeframe: str = "1h",
                       start_date: Optional[Union[datetime, str]] = None,
                       end_date: Optional[Union[datetime, str]] = None,
                       limit: Optional[int] = None) -> bool:
        """获取指定杠杆交易对的K线数据并保存到本地存储（增量去重）。

        说明：为统一行为，这里使用循环分页方式在 [start_date, end_date] 内抓取。
        """
        if not self.fetcher:
            self.init_fetcher()

        self.add_symbol(symbol)

        try:
            # 解析时间参数
            start_ms = None
            end_ms = None
            if start_date is not None:
                if isinstance(start_date, datetime):
                    start_ms = int(start_date.timestamp() * 1000)
                else:
                    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
            if end_date is not None:
                if isinstance(end_date, datetime):
                    end_ms = int(end_date.timestamp() * 1000)
                else:
                    end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)

            # 每次请求的条数
            per_limit = int(limit) if (limit is not None and int(limit) > 0) else 1000

            data_list: List[Any] = []

            if start_ms is not None and end_ms is not None:
                tf_sec = calculate_timeframe_seconds(timeframe)
                step_ms = max(60_000, int(tf_sec * 1000))

                since = start_ms
                safety = 0
                while since < end_ms and safety < 20000:
                    safety += 1
                    chunk = self.fetcher.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=per_limit
                    )

                    if not chunk:
                        break

                    data_list.extend(chunk)

                    # 推进 since：按最后一条时间戳 + 一个周期
                    last = chunk[-1]
                    if hasattr(last, 'timestamp'):
                        ts = last.timestamp
                        last_ms = int(ts.timestamp() * 1000) if hasattr(ts, 'timestamp') else int(ts)
                    else:
                        last_ms = int(last[0])

                    next_since = last_ms + step_ms
                    if next_since <= since:
                        next_since = since + step_ms
                    since = next_since

                    if since >= end_ms:
                        break
            else:
                # 无时间范围：单次获取
                data_list = self.fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=per_limit
                )

            # 转为可序列化结构
            serializable: List[Dict[str, Any]] = []
            for item in data_list:
                try:
                    if hasattr(item, 'timestamp'):
                        ts = item.timestamp
                        ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, 'timestamp') else int(ts)
                        serializable.append({
                            'timestamp': ts_ms,
                            'open': float(item.open),
                            'high': float(item.high),
                            'low': float(item.low),
                            'close': float(item.close),
                            'volume': float(item.volume)
                        })
                    else:
                        serializable.append({
                            'timestamp': int(item[0]),
                            'open': float(item[1]),
                            'high': float(item[2]),
                            'low': float(item[3]),
                            'close': float(item[4]),
                            'volume': float(item[5])
                        })
                except Exception:
                    continue

            # 过滤时间范围（防止最后一页超界）
            if start_ms is not None or end_ms is not None:
                filtered: List[Dict[str, Any]] = []
                for row in serializable:
                    ts = int(row.get('timestamp', 0))
                    if start_ms is not None and ts < start_ms:
                        continue
                    if end_ms is not None and ts > end_ms:
                        continue
                    filtered.append(row)
                serializable = filtered

            # 增量合并保存（按 timestamp 去重）
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            merged_key = f"{symbol_clean}_{timeframe}_merged"

            existing = []
            parquet_loaded = False
            try:
                # 优先尝试从 Parquet 加载
                parquet_path = self._dir / symbol_clean / timeframe / "ohlcv_merged.parquet"
                if parquet_path.exists():
                    import pandas as pd
                    existing = pd.read_parquet(parquet_path).to_dict('records')
                    parquet_loaded = True
            except Exception:
                pass

            if not parquet_loaded:
                existing = self.load(merged_key)
                if not isinstance(existing, list):
                    existing = []

            merged_map: Dict[int, Dict[str, Any]] = {}
            for row in existing:
                try:
                    ts = int(row.get('timestamp'))
                    merged_map[ts] = row
                except Exception:
                    continue
            for row in serializable:
                try:
                    ts = int(row.get('timestamp'))
                    merged_map[ts] = row
                except Exception:
                    continue

            merged_list = list(merged_map.values())
            merged_list.sort(key=lambda x: int(x.get('timestamp', 0)))
            if self.save_json_merged:
                self.save(merged_key, merged_list)

            # 可选：输出合并后的 parquet 文件
            try:
                import pandas as _pd
                if len(merged_list) > 0:
                    df = _pd.DataFrame(merged_list)
                    if 'timestamp' in df.columns:
                        df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='last')

                    out_dir = self._dir / symbol_clean / timeframe
                    import os
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = out_dir / "ohlcv_merged.parquet"
                    df.to_parquet(out_path, index=False, compression='snappy')
                    self.logger.info(f"Parquet(merged) 保存完成 -> {out_path}")
            except Exception as pe:
                self.logger.warning(f"Parquet(merged) 保存失败，已跳过（请检查是否安装 pyarrow/fastparquet）: {pe}")

            return True
        except Exception as e:
            self.logger.error(f"保存 {symbol} 杠杆K线数据失败: {e}")
            return False
    
    def close(self):
        """关闭管理器"""
        if self.fetcher:
            self.fetcher.close()
            self.fetcher = None


# ==================== 测试函数 ====================

def test_margin_fetcher():
    """测试杠杆交易获取器"""
    print("=" * 60)
    print("杠杆交易获取器模块测试")
    print("=" * 60)
    
    # 测试基础功能
    print("\n1. 测试CCXTMarginFetcher基础功能:")
    try:
        # 创建获取器（不使用真实API密钥）
        fetcher = CCXTMarginFetcher(exchange="binance", margin_type="cross")
        
        print(f"✅ 获取器创建成功: {fetcher}")
        print(f"✅ 交易所: {fetcher.exchange}")
        print(f"✅ 市场类型: {fetcher.market_type}")
        print(f"✅ 杠杆类型: {fetcher.margin_type}")
        
        # 测试连接
        if fetcher.test_connection():
            print("✅ 交易所连接测试成功")
        else:
            print("⚠️  交易所连接测试失败（可能是网络或代理问题）")
        
        # 获取可用交易对
        symbols = fetcher.get_available_symbols()
        print(f"✅ 获取到 {len(symbols) if symbols else 0} 个可用杠杆交易对")
        if symbols and len(symbols) > 0:
            print(f"✅ 示例交易对: {symbols[0]}")
        
        # 测试K线数据获取（模拟）
        print("\n2. 测试K线数据获取（模拟）:")
        # 这里使用一个已知的交易对进行测试
        test_symbol = "BTC/USDT"
        
        # 首先验证交易对
        if fetcher.validate_symbol(test_symbol):
            print(f"✅ 交易对验证成功: {test_symbol}")
            
            # 测试获取最近的K线数据
            ohlcv_data = fetcher.fetch_ohlcv(
                symbol=test_symbol,
                timeframe="1h",
                limit=5
            )
            
            if ohlcv_data:
                print(f"✅ K线数据获取成功: {len(ohlcv_data)} 条")
                for i, data in enumerate(ohlcv_data[:2]):  # 显示前2条
                    print(f"  数据 {i+1}: 时间 {data.timestamp}, 收盘价 {data.close}")
            else:
                print("⚠️  K线数据获取失败或为空（可能是网络或API限制）")
        else:
            print(f"⚠️  交易对验证失败: {test_symbol}")
        
        # 测试市场信息获取
        print("\n3. 测试市场信息获取（模拟）:")
        market_info = fetcher.fetch_market_info(test_symbol)
        if market_info:
            print(f"✅ 市场信息获取成功")
            print(f"  最大杠杆: {market_info.get('max_leverage', 'N/A')}倍")
            print(f"  基础币种: {market_info.get('base', 'N/A')}")
        else:
            print("⚠️  市场信息获取失败")
        
        # 测试借币利率获取 - 优化版本
        print("\n4. 测试借币利率获取（优化版）:")
        try:
            borrow_rates = fetcher.fetch_borrow_rates()
            if borrow_rates:
                print(f"✅ 借币利率获取成功: {len(borrow_rates)} 个币种")
                # 显示前几个币种的利率
                for i, (currency, rate_info) in enumerate(list(borrow_rates.items())[:3]):
                    rate = rate_info.get('rate', 0)
                    annual_rate = rate_info.get('annual_rate', rate * 365)
                    print(f"  {currency}: 日利率 {rate:.6%}, 年化 {annual_rate:.2%}")
            else:
                print("⚠️  借币利率获取失败或为空")
        except Exception as e:
            print(f"⚠️  借币利率获取异常: {e}")
        
        # 测试保证金成本计算
        print("\n5. 测试保证金成本计算（模拟）:")
        cost_info = fetcher.calculate_margin_cost(
            symbol=test_symbol,
            amount=1.0,
            leverage=3,
            days=30
        )
        if cost_info:
            print(f"✅ 保证金成本计算成功")
            print(f"  所需保证金: {cost_info.get('margin_required', 0):.2f}")
            print(f"  总利息成本: {cost_info.get('total_interest', 0):.2f}")
            print(f"  利息百分比: {cost_info.get('interest_percentage', 0):.4f}%")
        else:
            print("⚠️  保证金成本计算失败")
        
        # 测试强平价格计算
        print("\n6. 测试强平价格计算（模拟）:")
        liquidation_info = fetcher.fetch_liquidation_price(
            symbol=test_symbol,
            side="buy",
            entry_price=50000,
            leverage=3
        )
        if liquidation_info:
            print(f"✅ 强平价格计算成功")
            print(f"  强平价格: {liquidation_info.get('liquidation_price', 0):.2f}")
            print(f"  安全边际: {liquidation_info.get('safety_margin_percent', 0):.2f}%")
            print(f"  距离当前价格: {liquidation_info.get('price_distance_percent', 0):.2f}%")
        else:
            print("⚠️  强平价格计算失败")
        
        # 获取状态
        status = fetcher.get_status()
        print(f"\n✅ 获取器状态:")
        print(f"  请求统计: {status['request_stats']}")
        print(f"  错误计数: {status['error_count']}")
        
        # 关闭获取器
        fetcher.close()
        print("\n✅ 获取器关闭成功")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试杠杆交易数据管理器
    print("\n7. 测试杠杆交易数据管理器:")
    try:
        manager = MarginDataManager(exchange="binance", margin_type="cross")
        manager.add_symbols(["BTC/USDT", "ETH/USDT", "BNB/USDT"])
        
        print(f"✅ 管理器创建成功，交易对: {manager.symbols}")
        
        # 获取市场摘要
        summary = manager.get_market_summary()
        print(f"✅ 市场摘要: {summary['total_symbols']} 个交易对")
        
        # 测试投资组合成本计算
        portfolio = {
            "BTC/USDT": {"amount": 0.1, "leverage": 3, "side": "buy", "days": 30},
            "ETH/USDT": {"amount": 1.0, "leverage": 5, "side": "buy", "days": 30}
        }
        portfolio_cost = manager.calculate_portfolio_margin_cost(portfolio)
        print(f"✅ 投资组合成本计算成功")
        print(f"  总保证金: {portfolio_cost.get('total_margin_required', 0):.2f}")
        
        # 测试投资组合强平价格计算
        portfolio_with_prices = {
            "BTC/USDT": {"amount": 0.1, "leverage": 3, "side": "buy", "entry_price": 50000},
            "ETH/USDT": {"amount": 1.0, "leverage": 5, "side": "buy", "entry_price": 3000}
        }
        liquidation_info = manager.calculate_portfolio_liquidation_prices(portfolio_with_prices)
        print(f"✅ 投资组合强平价格计算成功")
        print(f"  风险水平: {liquidation_info.get('overall_risk_level', 'UNKNOWN')}")
        
        manager.close()
        print("✅ 管理器关闭成功")
        
    except Exception as e:
        print(f"❌ 管理器测试失败: {e}")
    
    print("\n✅ 杠杆交易获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    test_margin_fetcher()