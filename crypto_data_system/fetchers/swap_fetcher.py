"""
永续合约（Swap）数据获取器模块
提供从交易所获取永续合约数据的功能
"""

import time
import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 导入基础模块
try:
    from .base_fetcher import BaseFetcher, AsyncFetcher
    from ..data_models import SwapOHLCVData, OrderBookData, TradeData, FundingRateData, OpenInterestData
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
    from data_models import SwapOHLCVData, OrderBookData, TradeData, FundingRateData, OpenInterestData
    from utils.logger import get_logger, log_execution_time, log_errors
    from utils.cache import CacheManager, cached, cache_result
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from config import get_exchange_config, ExchangeSymbolFormats, get_market_config


# ==================== 永续合约数据获取器 ====================

class SwapFetcher(BaseFetcher):
    """
    永续合约数据获取器基类
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 contract_type: str = "linear",  # linear: USDT合约, inverse: 币本位合约
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化永续合约数据获取器
        
        参数:
            exchange: 交易所名称
            contract_type: 合约类型 (linear, inverse)
            config: 配置字典
            cache_manager: 缓存管理器
        """
        super().__init__(
            exchange=exchange,
            market_type="swap",
            config=config,
            cache_manager=cache_manager
        )
        
        self.contract_type = contract_type  # linear: USDT合约, inverse: 币本位合约
        
        # 加载交易所配置
        self.exchange_config = get_exchange_config(exchange)
        
        # 加载市场配置
        self.market_config = get_market_config("swap")
        
        # 初始化交易所连接
        self.exchange_instance = None
        self._init_exchange()
        
        # 初始化合约信息
        self.contracts_info = {}
        self._load_contracts_info()
    
    def _init_exchange(self):
        """初始化交易所连接"""
        try:
            self.logger.info(f"初始化永续合约交易所连接: {self.exchange}, 合约类型: {self.contract_type}")
            
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
            
            # 设置默认类型为永续合约
            ccxt_config['options'] = ccxt_config.get('options', {})
            
            if self.exchange == "binance":
                # 币安永续合约使用future类型，但通过defaultSettle指定USDT
                ccxt_config['options']['defaultType'] = 'future'
                if self.contract_type == "linear":
                    ccxt_config['options']['defaultSettle'] = 'USDT'
                elif self.contract_type == "inverse":
                    ccxt_config['options']['defaultSettle'] = 'BTC'
            else:
                # 其他交易所使用swap类型
                ccxt_config['options']['defaultType'] = 'swap'
                if self.contract_type == "linear":
                    ccxt_config['options']['defaultSettle'] = 'USDT'
                elif self.contract_type == "inverse":
                    ccxt_config['options']['defaultSettle'] = 'BTC'
            
            # 合并市场配置
            if 'ccxt_options' in self.market_config:
                if 'options' not in ccxt_config:
                    ccxt_config['options'] = {}
                ccxt_config['options'].update(self.market_config['ccxt_options'])

            # 如果有代理配置，添加
            if self.config.proxy_url:
                if self.config.verbose:
                    self.logger.info(f"使用代理: {self.config.proxy_url}")
                ccxt_config['proxies'] = {
                    'http': self.config.proxy_url,
                    'https': self.config.proxy_url,
                }
            
            # 创建交易所实例
            self.exchange_instance = exchange_class(ccxt_config)
            
            # 加载市场信息
            self._load_markets()
            
            self.logger.info(f"永续合约交易所连接初始化成功: {self.exchange}")
            
        except Exception as e:
            self.logger.error(f"永续合约交易所连接初始化失败: {e}")
            self.exchange_instance = None
    
    def _load_markets(self):
        """加载市场信息"""
        if not self.exchange_instance:
            return
        
        try:
            self.logger.info(f"加载 {self.exchange} 永续合约市场信息...")
            
            # 加载市场
            self.exchange_instance.load_markets()
            
            # 筛选永续合约交易对
            self.swap_markets = {}
            for symbol, market in self.exchange_instance.markets.items():
                # 根据不同交易所的标识判断永续合约
                is_swap = False
                
                if self.exchange == "binance":
                    # 币安永续合约: symbol包含 ':USDT' 或 ':BTC'，或者包含 'PERP'
                    if ':' in symbol:
                        if self.contract_type == "linear" and (':USDT' in symbol or ':BUSD' in symbol):
                            # 检查是否是永续合约（没有到期日）
                            expiry = market.get('expiry', None)
                            if expiry is None:
                                is_swap = True
                        elif self.contract_type == "inverse" and ':BTC' in symbol:
                            expiry = market.get('expiry', None)
                            if expiry is None:
                                is_swap = True
                    elif 'PERP' in symbol or 'SWAP' in symbol:
                        is_swap = True
                
                elif self.exchange == "okx":
                    # OKX永续合约: symbol包含 '-SWAP'
                    if '-SWAP' in symbol:
                        is_swap = True
                
                elif self.exchange == "bybit":
                    # Bybit永续合约
                    if market.get('linear', False) or market.get('inverse', False):
                        # 检查是否是永续合约
                        if not re.search(r'\d{6,8}$', symbol):  # 没有到期日数字
                            is_swap = True
                
                if is_swap and market.get('active', False):
                    self.swap_markets[symbol] = market
            
            self.logger.info(f"加载 {len(self.swap_markets)} 个永续合约交易对")
            
            # 缓存市场信息
            if self.cache_manager:
                self.cache_manager.set(
                    key=f"{self.exchange}_swap_{self.contract_type}_markets",
                    data=self.swap_markets,
                    ttl=3600,  # 缓存1小时
                    sub_dir='swap'
                )
            
        except Exception as e:
            self.logger.error(f"加载市场信息失败: {e}")
            self.swap_markets = {}
    
    def _load_contracts_info(self):
        """加载合约信息"""
        if not self.swap_markets:
            return
        
        for symbol, market in self.swap_markets.items():
            self.contracts_info[symbol] = {
                'symbol': symbol,
                'contract_size': market.get('contractSize', 1.0),
                'settle_currency': market.get('settle', 'USDT'),
                'margin_currency': market.get('margin', 'USDT'),
                'tick_size': market.get('precision', {}).get('price', 0.01),
                'lot_size': market.get('precision', {}).get('amount', 1.0),
                'min_notional': market.get('limits', {}).get('cost', {}).get('min', 0),
                'max_leverage': market.get('limits', {}).get('leverage', {}).get('max', 125),
                'maintenance_margin_rate': market.get('info', {}).get('maintMarginPercent', 0.005),
            }
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的永续合约交易对
        
        返回:
            交易对列表
        """
        if not self.swap_markets:
            self._load_markets()
        
        return list(self.swap_markets.keys())
    
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
        
        # 尝试格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 如果有市场信息，检查是否存在
        if self.swap_markets and formatted_symbol not in self.swap_markets:
            # 尝试查找相似的符号
            similar_symbols = self.find_similar_symbols(symbol)
            if similar_symbols:
                self.logger.warning(f"交易对 {formatted_symbol} 未找到，相似交易对: {similar_symbols[:3]}")
            else:
                self.logger.warning(f"交易对可能无效: {formatted_symbol}")
            return False
        
        return True
    
    def find_similar_symbols(self, symbol: str) -> List[str]:
        """
        查找相似的交易对
        
        参数:
            symbol: 原始交易对
            
        返回:
            相似交易对列表
        """
        if not self.swap_markets:
            return []
        
        # 提取基础交易对
        base_symbol = self.extract_base_symbol(symbol)
        if not base_symbol:
            return []
        
        # 查找包含基础交易对的符号
        similar = []
        for market_symbol in self.swap_markets.keys():
            if base_symbol in market_symbol:
                similar.append(market_symbol)
        
        return similar
    
    def extract_base_symbol(self, symbol: str) -> str:
        """
        提取基础交易对
        
        参数:
            symbol: 原始交易对
            
        返回:
            基础交易对
        """
        # 移除永续合约标识
        symbol = symbol.replace(':USDT', '').replace(':BTC', '').replace(':BUSD', '')
        symbol = symbol.replace('-SWAP', '').replace('-PERP', '')
        
        # 移除日期后缀（如果有）
        if re.search(r'-\d{6,8}$', symbol):
            symbol = re.sub(r'-\d{6,8}$', '', symbol)
        
        return symbol
    
    def format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号为交易所标准格式
        
        参数:
            symbol: 原始交易对符号
            
        返回:
            格式化后的交易对符号
        """
        # 如果已经包含永续合约标识，直接返回
        if (':USDT' in symbol or ':BTC' in symbol or ':BUSD' in symbol or 
            '-SWAP' in symbol or '-PERP' in symbol):
            return symbol
        
        # 移除重复的格式化（安全检查）
        if symbol.count(':') > 1:
            parts = symbol.split(':')
            symbol = ':'.join(parts[:2])  # 只保留前两部分
        
        # 根据交易所格式化
        if self.exchange == "binance":
            # 币安永续合约格式: BTC/USDT:USDT
            if '/' in symbol:
                base, quote = symbol.split('/')
                if self.contract_type == "linear":
                    formatted = f"{base}/{quote}:USDT"
                else:
                    formatted = f"{base}/{quote}:{base}"
            else:
                # 尝试解析 BTCUSDT 格式
                if symbol.endswith('USDT'):
                    base = symbol.replace('USDT', '')
                    formatted = f"{base}/USDT:USDT"
                elif symbol.endswith('BTC'):
                    base = symbol.replace('BTC', '')
                    formatted = f"{base}/BTC:{base}"
                else:
                    formatted = symbol
        
        elif self.exchange == "okx":
            # OKX永续合约格式: BTC-USDT-SWAP
            if '/' in symbol:
                base, quote = symbol.split('/')
                formatted = f"{base}-{quote}-SWAP"
            elif '-' in symbol and not symbol.endswith('-SWAP'):
                formatted = f"{symbol}-SWAP"
            else:
                formatted = symbol
        
        else:
            formatted = symbol
        
        self.logger.debug(f"格式化永续合约交易对: {symbol} -> {formatted}")
        return formatted
    
    @log_errors(reraise=False)
    @cached(key_prefix="swap_ohlcv", ttl=300, sub_dir="swap")
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = "1h",
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None,
                   **kwargs) -> List[SwapOHLCVData]:
        """
        获取永续合约K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            SwapOHLCV数据列表
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 查找最匹配的符号
        if formatted_symbol not in self.swap_markets:
            similar_symbols = self.find_similar_symbols(symbol)
            if similar_symbols:
                formatted_symbol = similar_symbols[0]
                self.logger.info(f"使用相似交易对: {symbol} -> {formatted_symbol}")
        
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
            f"获取永续合约K线数据: {formatted_symbol}, "
            f"时间间隔: {timeframe}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 使用CCXT获取数据
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
                    data_model = SwapOHLCVData.from_ccxt(
                        ohlcv=ohlcv,
                        symbol=formatted_symbol,
                        timeframe=timeframe,
                        exchange=self.exchange,
                        market_type="swap"
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换永续合约OHLCV数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条永续合约K线数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取永续合约K线数据失败: {e}")
            # 如果失败，返回空列表而不是抛出异常
            if kwargs.get('raise_error', False):
                raise
            return []
    
    @log_errors(reraise=False)
    @cached(key_prefix="swap_orderbook", ttl=30, sub_dir="swap")
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> Optional[OrderBookData]:
        """
        获取永续合约订单簿数据
        
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
        
        self.logger.info(f"获取永续合约订单簿: {formatted_symbol}, 深度: {limit}")
        
        try:
            # 使用CCXT获取订单簿
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
                f"永续合约订单簿获取成功: 买盘 {len(data_model.bids)} 个, "
                f"卖盘 {len(data_model.asks)} 个, "
                f"价差: {data_model.spread:.2f}"
            )
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取永续合约订单簿失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="swap_trades", ttl=60, sub_dir="swap")
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取永续合约成交数据
        
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
            f"获取永续合约成交数据: {formatted_symbol}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 使用CCXT获取成交数据
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
                    self.logger.warning(f"转换永续合约成交数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条永续合约成交数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取永续合约成交数据失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []

    @log_errors(reraise=False)
    @cached(key_prefix="swap_ticker", ttl=30, sub_dir="swap")
    def fetch_ticker(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """获取永续合约行情（Ticker）。"""
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        formatted_symbol = self.format_symbol(symbol)
        self.logger.info(f"获取永续合约行情: {formatted_symbol}")

        try:
            ticker = self.exchange_instance.fetch_ticker(symbol=formatted_symbol, params=kwargs.get('params', {}))
            return ticker or {}
        except Exception as e:
            self.logger.error(f"获取永续合约行情失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return {}

    @log_errors(reraise=False)
    @cached(key_prefix="swap_market_info", ttl=3600, sub_dir="swap")
    def fetch_market_info(self, symbol: str = None) -> Dict[str, Any]:
        """获取永续合约市场信息（Market Info）。"""
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        if not symbol:
            return {}

        formatted_symbol = self.format_symbol(symbol)
        try:
            market = self.exchange_instance.market(formatted_symbol)
            return market or {}
        except Exception as e:
            self.logger.error(f"获取永续合约市场信息失败: {e}")
            return {}

    def fetch_market_snapshot(
        self,
        symbol: str,
        timeframe: str = '1h',
        ohlcv_limit: int = 200,
        trades_limit: int = 200,
        orderbook_limit: int = 50,
        liquidation_limit: int = 50,
        include: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """获取永续合约某交易对的“综合市场快照”。

        include 允许项（默认全开）：
        - ticker/orderbook/trades/market_info
        - funding_rate/open_interest/mark_price/liquidations/contract_info
        - ohlcv
        """
        include_set = set((include or [
            'ticker', 'orderbook', 'trades', 'market_info',
            'funding_rate', 'open_interest', 'mark_price',
            'liquidations', 'contract_info',
        ]))

        formatted_symbol = self.format_symbol(symbol)
        snapshot: Dict[str, Any] = {
            'exchange': self.exchange,
            'market_type': 'swap',
            'symbol': formatted_symbol,
            'timestamp': pd.Timestamp.now(),
        }

        if 'ticker' in include_set:
            snapshot['ticker'] = self.fetch_ticker(formatted_symbol)
        if 'orderbook' in include_set:
            snapshot['orderbook'] = self.fetch_orderbook(formatted_symbol, limit=orderbook_limit)
        if 'trades' in include_set:
            snapshot['trades'] = self.fetch_trades(formatted_symbol, limit=trades_limit)
        if 'market_info' in include_set:
            snapshot['market_info'] = self.fetch_market_info(formatted_symbol)

        if 'funding_rate' in include_set:
            snapshot['funding_rate'] = self.fetch_funding_rate(formatted_symbol)
        if 'open_interest' in include_set:
            snapshot['open_interest'] = self.fetch_open_interest(formatted_symbol)
        if 'mark_price' in include_set:
            snapshot['mark_price'] = self.fetch_mark_price(formatted_symbol)
        if 'liquidations' in include_set:
            snapshot['liquidations'] = self.fetch_liquidation_info(formatted_symbol, limit=liquidation_limit)
        if 'contract_info' in include_set:
            snapshot['contract_info'] = self.fetch_contract_info(formatted_symbol)

        if 'ohlcv' in include_set:
            snapshot['ohlcv'] = self.fetch_ohlcv(formatted_symbol, timeframe=timeframe, limit=ohlcv_limit)

        return snapshot
    
    @log_errors(reraise=False)
    @cached(key_prefix="swap_funding_rate", ttl=300, sub_dir="swap")
    def fetch_funding_rate(self, 
                          symbol: str,
                          **kwargs) -> Optional[FundingRateData]:
        """
        获取资金费率
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            资金费率数据
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"获取永续合约资金费率: {formatted_symbol}")
        
        try:
            # 使用CCXT获取资金费率
            funding_rate_info = self.exchange_instance.fetch_funding_rate(
                symbol=formatted_symbol
            )
            
            # 安全地处理None值
            funding_rate_value = funding_rate_info.get('fundingRate')
            next_funding_rate_value = funding_rate_info.get('nextFundingRate')
            timestamp_value = funding_rate_info.get('timestamp')
            
            if funding_rate_value is None:
                self.logger.warning(f"资金费率数据为None，使用默认值0")
                funding_rate_value = 0
            
            if next_funding_rate_value is None:
                next_funding_rate_value = funding_rate_value  # 使用当前费率作为预测费率
            
            # 转换为数据模型
            data_model = FundingRateData(
                timestamp=pd.Timestamp.now(),
                funding_time=pd.Timestamp(timestamp_value if timestamp_value else pd.Timestamp.now()),
                symbol=formatted_symbol,
                exchange=self.exchange,
                market_type="swap",
                funding_rate=float(funding_rate_value),
                predicted_rate=float(next_funding_rate_value),
                interval_hours=funding_rate_info.get('fundingInterval', 8)
            )
            
            self.logger.info(
                f"资金费率获取成功: {data_model.funding_rate:.6%}, "
                f"预测费率: {data_model.predicted_rate:.6%}"
            )
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取资金费率失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="swap_funding_rate_history", ttl=600, sub_dir="swap")
    def fetch_funding_rate_history(self, 
                                  symbol: str,
                                  since: Optional[Union[int, datetime, str]] = None,
                                  limit: Optional[int] = None,
                                  **kwargs) -> List[FundingRateData]:
        """
        获取资金费率历史
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            资金费率历史数据列表
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
            limit = self.config.get('funding_rate_limit', 100)
        
        self.logger.info(
            f"获取永续合约资金费率历史: {formatted_symbol}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 使用CCXT获取资金费率历史
            funding_rates = self.exchange_instance.fetch_funding_rate_history(
                symbol=formatted_symbol,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {})
            )
            
            # 转换为数据模型
            data_models = []
            for fr in funding_rates:
                try:
                    # 安全处理None值
                    funding_rate_value = fr.get('fundingRate', 0)
                    if funding_rate_value is None:
                        funding_rate_value = 0
                    
                    next_funding_rate_value = fr.get('nextFundingRate', funding_rate_value)
                    if next_funding_rate_value is None:
                        next_funding_rate_value = funding_rate_value
                    
                    data_model = FundingRateData(
                        timestamp=pd.Timestamp.now(),
                        funding_time=pd.Timestamp(fr.get('timestamp', pd.Timestamp.now())),
                        symbol=formatted_symbol,
                        exchange=self.exchange,
                        market_type="swap",
                        funding_rate=float(funding_rate_value),
                        predicted_rate=float(next_funding_rate_value),
                        interval_hours=fr.get('fundingInterval', 8)
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换资金费率历史数据失败: {e}")
                    continue
            
            # 计算累计资金费率
            if data_models:
                cumulative = 0
                for data_model in data_models:
                    cumulative += data_model.funding_rate
                    data_model.cumulative_rate = cumulative
            
            self.logger.info(f"获取到 {len(data_models)} 条资金费率历史数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取资金费率历史失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []
    
    @log_errors(reraise=False)
    @cached(key_prefix="swap_open_interest", ttl=300, sub_dir="swap")
    def fetch_open_interest(self, 
                           symbol: str,
                           **kwargs) -> Optional[OpenInterestData]:
        """
        获取未平仓合约数据
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            未平仓合约数据
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"获取永续合约未平仓合约: {formatted_symbol}")
        
        try:
            # 使用CCXT获取未平仓合约数据
            oi_info = self.exchange_instance.fetch_open_interest(
                symbol=formatted_symbol
            )
            
            # 安全处理None值
            open_interest_value = oi_info.get('openInterestAmount')
            open_interest_value_amount = oi_info.get('openInterestValue')
            
            if open_interest_value is None:
                open_interest_value = 0
                self.logger.warning(f"未平仓合约为None，使用默认值0")
            
            if open_interest_value_amount is None:
                open_interest_value_amount = 0
            
            # 转换为数据模型
            data_model = OpenInterestData(
                timestamp=pd.Timestamp.now(),
                symbol=formatted_symbol,
                exchange=self.exchange,
                market_type="swap",
                open_interest=float(open_interest_value),
                open_interest_value=float(open_interest_value_amount),
                volume_24h=float(oi_info.get('baseVolume', 0) or 0),
                turnover_24h=float(oi_info.get('quoteVolume', 0) or 0)
            )
            
            self.logger.info(
                f"未平仓合约获取成功: {data_model.open_interest:.2f} "
                f"(价值: {data_model.open_interest_value:.2f} {self.contracts_info.get(formatted_symbol, {}).get('settle_currency', 'USDT')})"
            )
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取未平仓合约失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="swap_open_interest_history", ttl=600, sub_dir="swap")
    def fetch_open_interest_history(self, 
                                   symbol: str,
                                   timeframe: str = "1h",
                                   since: Optional[Union[int, datetime, str]] = None,
                                   limit: Optional[int] = None,
                                   **kwargs) -> List[OpenInterestData]:
        """
        获取未平仓合约历史数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            未平仓合约历史数据列表
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
            limit = self.config.get('open_interest_limit', 100)
        
        self.logger.info(
            f"获取永续合约未平仓合约历史: {formatted_symbol}, "
            f"时间间隔: {timeframe}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 使用CCXT获取未平仓合约历史
            oi_history = self.exchange_instance.fetch_open_interest_history(
                symbol=formatted_symbol,
                timeframe=timeframe,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {})
            )
            
            # 转换为数据模型
            data_models = []
            for oi in oi_history:
                try:
                    # 安全处理None值
                    open_interest_value = oi.get('openInterestAmount', 0)
                    if open_interest_value is None:
                        open_interest_value = 0
                    
                    open_interest_value_amount = oi.get('openInterestValue', 0)
                    if open_interest_value_amount is None:
                        open_interest_value_amount = 0
                    
                    data_model = OpenInterestData(
                        timestamp=pd.Timestamp(oi.get('timestamp', pd.Timestamp.now())),
                        symbol=formatted_symbol,
                        exchange=self.exchange,
                        market_type="swap",
                        open_interest=float(open_interest_value),
                        open_interest_value=float(open_interest_value_amount),
                        volume_24h=float(oi.get('baseVolume', 0) or 0),
                        turnover_24h=float(oi.get('quoteVolume', 0) or 0)
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换未平仓合约历史数据失败: {e}")
                    continue
            
            # 计算变化率
            if len(data_models) > 1:
                for i in range(1, len(data_models)):
                    data_models[i].calculate_changes(data_models[i-1])
            
            self.logger.info(f"获取到 {len(data_models)} 条未平仓合约历史数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取未平仓合约历史失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []
    
    def fetch_mark_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取标记价格
        
        参数:
            symbol: 交易对符号
            
        返回:
            标记价格信息
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"获取永续合约标记价格: {formatted_symbol}")
        
        try:
            # 使用CCXT获取标记价格
            mark_price_info = self.exchange_instance.fetch_mark_price(
                symbol=formatted_symbol
            )
            
            # 安全处理None值
            mark_price_value = mark_price_info.get('markPrice', 0)
            if mark_price_value is None:
                mark_price_value = 0
            
            index_price_value = mark_price_info.get('indexPrice', 0)
            if index_price_value is None:
                index_price_value = 0
            
            funding_rate_value = mark_price_info.get('fundingRate', 0)
            if funding_rate_value is None:
                funding_rate_value = 0
            
            # 兼容时间戳处理
            ts = mark_price_info.get('timestamp')
            if ts:
                ts = pd.Timestamp(ts, unit='ms' if isinstance(ts, (int, float)) and ts > 2e10 else None)
            else:
                ts = pd.Timestamp.now()

            result = {
                'symbol': symbol,  # 使用原始symbol
                'timestamp': ts,
                'mark_price': float(mark_price_value),
                'index_price': float(index_price_value),
                'settlement_price': float(mark_price_info.get('settlementPrice', 0) or 0),
                'funding_rate': float(funding_rate_value),
                'next_funding_time': pd.Timestamp(mark_price_info.get('nextFundingTime', 0), unit='ms') if mark_price_info.get('nextFundingTime') else None,
                'interest_rate': float(mark_price_info.get('interestRate', 0) or 0)
            }
            
            self.logger.info(
                f"标记价格获取成功: {result['mark_price']}, "
                f"指数价格: {result['index_price']}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取标记价格失败: {e}")
            return None
    
    def fetch_basis(self, symbol: str, spot_price: Optional[float] = None) -> Dict[str, Any]:
        """
        计算基差（永续合约价格与现货价格的差异）
        
        参数:
            symbol: 交易对符号
            spot_price: 现货价格（如果不提供，会尝试获取）
            
        返回:
            基差信息
        """
        # 获取永续合约最新价格
        swap_ohlcv = self.fetch_ohlcv(symbol, timeframe="1m", limit=1)
        
        if not swap_ohlcv:
            return {}
        
        swap_price = swap_ohlcv[0].close
        
        # 如果未提供现货价格，尝试从现货市场获取
        if spot_price is None:
            try:
                # 创建现货获取器
                from .spot_fetcher import SpotFetcher
                spot_fetcher = SpotFetcher(exchange=self.exchange)
                # 提取基础交易对
                base_symbol = self.extract_base_symbol(symbol)
                spot_ohlcv = spot_fetcher.fetch_ohlcv(base_symbol, timeframe="1m", limit=1)
                
                if spot_ohlcv:
                    spot_price = spot_ohlcv[0].close
                else:
                    spot_price = 0
            except Exception as e:
                self.logger.warning(f"获取现货价格失败: {e}")
                spot_price = 0
        
        # 计算基差
        if spot_price > 0:
            basis = swap_price - spot_price
            basis_percent = (basis / spot_price * 100)
        else:
            basis = 0
            basis_percent = 0
        
        # 获取资金费率
        funding_rate_info = self.fetch_funding_rate(symbol)
        funding_rate = funding_rate_info.funding_rate if funding_rate_info else 0
        
        # 年化资金费率
        # 资金费率通常是每8小时收取一次
        daily_funding_rate = funding_rate * 3  # 每天3次
        annualized_funding = daily_funding_rate * 365 * 100  # 年化百分比
        
        result = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now(),
            'swap_price': swap_price,
            'spot_price': spot_price,
            'basis': basis,
            'basis_percent': basis_percent,
            'funding_rate': funding_rate,
            'annualized_funding': annualized_funding,
            'premium': basis_percent,  # 溢价率
            'is_premium': basis > 0,  # 是否溢价
            'is_discount': basis < 0,  # 是否折价
        }
        
        return result
    
    def fetch_liquidation_info(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取强平信息
        
        参数:
            symbol: 交易对符号
            limit: 限制条数
            
        返回:
            强平信息列表
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"获取永续合约强平信息: {formatted_symbol}")
        
        try:
            # 使用CCXT获取强平信息
            liquidations = self.exchange_instance.fetch_liquidations(
                symbol=formatted_symbol,
                limit=limit
            )
            
            result = []
            for liq in liquidations:
                # 安全处理None值
                quantity_value = liq.get('quantity', 0)
                if quantity_value is None:
                    quantity_value = 0
                
                price_value = liq.get('price', 0)
                if price_value is None:
                    price_value = 0
                
                liquidation_info = {
                    'symbol': formatted_symbol,
                    'timestamp': pd.Timestamp(liq.get('timestamp', pd.Timestamp.now())),
                    'side': liq.get('side', 'unknown'),
                    'quantity': float(quantity_value),
                    'price': float(price_value),
                    'value': float(quantity_value) * float(price_value),
                    'type': liq.get('type', 'unknown'),
                    'leverage': float(liq.get('leverage', 0) or 0)
                }
                result.append(liquidation_info)
            
            self.logger.info(f"获取到 {len(result)} 条强平信息")
            return result
            
        except Exception as e:
            self.logger.error(f"获取强平信息失败: {e}")
            return []
    
    def fetch_contract_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取合约信息
        
        参数:
            symbol: 交易对符号
            
        返回:
            合约信息
        """
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        if formatted_symbol in self.contracts_info:
            return self.contracts_info[formatted_symbol]
        
        # 如果缓存中没有，尝试从交易所获取
        if not self.exchange_instance:
            return {}
        
        try:
            market = self.exchange_instance.market(formatted_symbol)
            
            contract_info = {
                'symbol': formatted_symbol,
                'contract_size': market.get('contractSize', 1.0),
                'settle_currency': market.get('settle', 'USDT'),
                'margin_currency': market.get('margin', 'USDT'),
                'tick_size': market.get('precision', {}).get('price', 0.01),
                'lot_size': market.get('precision', {}).get('amount', 1.0),
                'min_notional': market.get('limits', {}).get('cost', {}).get('min', 0),
                'max_leverage': market.get('limits', {}).get('leverage', {}).get('max', 125),
                'maintenance_margin_rate': market.get('info', {}).get('maintMarginPercent', 0.005),
                'taker_fee': market.get('taker', 0.0004),
                'maker_fee': market.get('maker', 0.0002),
                'settlement_time': market.get('settlementTime', None),
                'delivery_time': market.get('deliveryTime', None)
            }
            
            # 缓存合约信息
            self.contracts_info[formatted_symbol] = contract_info
            
            return contract_info
            
        except Exception as e:
            self.logger.error(f"获取合约信息失败: {e}")
            return {}
    
    def calculate_funding_cost(self, symbol: str, position_size: float, days: int = 30) -> Dict[str, float]:
        """
        计算资金成本
        
        参数:
            symbol: 交易对符号
            position_size: 持仓数量
            days: 天数
            
        返回:
            资金成本信息
        """
        # 获取资金费率历史
        funding_history = self.fetch_funding_rate_history(symbol, limit=100)
        
        if not funding_history:
            # 获取当前资金费率作为替代
            current_funding = self.fetch_funding_rate(symbol)
            if current_funding:
                funding_history = [current_funding]
            else:
                return {}
        
        # 计算平均资金费率
        rates = [fr.funding_rate for fr in funding_history]
        avg_rate = sum(rates) / len(rates) if rates else 0
        
        # 计算预计成本
        contract_info = self.fetch_contract_info(symbol)
        contract_size = contract_info.get('contract_size', 1.0)
        
        # 每8小时收取一次资金费率
        funding_times_per_day = 24 / 8  # 3次
        
        # 每日资金成本 = 持仓价值 * 平均资金费率 * 每日收取次数
        position_value = position_size * contract_size
        daily_cost = position_value * avg_rate * funding_times_per_day
        
        result = {
            'symbol': symbol,
            'position_size': position_size,
            'contract_size': contract_size,
            'position_value': position_value,
            'avg_funding_rate': avg_rate,
            'daily_funding_times': funding_times_per_day,
            'daily_cost': daily_cost,
            'monthly_cost': daily_cost * days,
            'annual_cost': daily_cost * 365,
            'funding_history_count': len(funding_history)
        }
        
        return result
    
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
            self.logger.info(f"永续合约交易所连接测试成功，服务器时间: {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"永续合约交易所连接测试失败: {e}")
            return False
    
    def close(self):
        """关闭获取器，释放资源"""
        if self.exchange_instance:
            self.exchange_instance = None
        
        super().close()


# ==================== CCXT永续合约获取器 ====================

class CCXTSwapFetcher(SwapFetcher):
    """
    CCXT永续合约数据获取器
    
    使用CCXT库获取永续合约数据
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 contract_type: str = "linear",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 **kwargs):
        """
        初始化CCXT永续合约获取器
        
        参数:
            exchange: 交易所名称
            contract_type: 合约类型
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
            'contract_type': contract_type
        })
        
        # 合并额外参数到配置
        for key, value in kwargs.items():
            if key not in ['cache_manager']:
                config[key] = value
        
        super().__init__(
            exchange=exchange,
            contract_type=contract_type,
            config=config,
            cache_manager=kwargs.get('cache_manager')
        )
        
        # 如果提供了API密钥，更新交易所实例
        if api_key and api_secret and self.exchange_instance:
            self.exchange_instance.apiKey = api_key
            self.exchange_instance.secret = api_secret


# ==================== 永续合约数据管理器 ====================

try:
    from crypto_data_system.storage.data_manager import FileDataManager
except (ImportError, ModuleNotFoundError):
    # 脚本直接运行时的备用导入方式
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crypto_data_system.storage.data_manager import FileDataManager


class SwapDataManager(FileDataManager):
    """
    永续合约数据管理器
    
    管理多个交易对的永续合约数据获取
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 contract_type: str = "linear",
                 fetcher_config: Optional[Dict] = None,
                 root_dir: Optional[str] = None,
                 cache_manager: Optional[Any] = None,
                 save_json_merged: bool = False):
        """
        初始化永续合约数据管理器
        
        参数:
            exchange: 交易所名称
            contract_type: 合约类型
            fetcher_config: 获取器配置
        """
        self.exchange = exchange
        self.contract_type = contract_type
        self.fetcher_config = fetcher_config or {}
        self.fetcher = None
        self.symbols = []
        self.save_json_merged = bool(self.fetcher_config.get('save_json_merged', save_json_merged))
        
        # 初始化日志
        self.logger = get_logger(f"swap_manager.{exchange}.{contract_type}")

        # 初始化文件存储（子目录按交易所/合约类型分组）
        super().__init__(root_dir=root_dir, sub_dir=f"swap/{exchange}/{contract_type}", file_format="json", cache_manager=cache_manager)
    
    def init_fetcher(self):
        """初始化数据获取器"""
        if not self.fetcher:
            self.fetcher = CCXTSwapFetcher(
                exchange=self.exchange,
                contract_type=self.contract_type,
                config=self.fetcher_config
            )
            
            # 测试连接
            if not self.fetcher.test_connection():
                self.logger.warning(f"交易所 {self.exchange} 永续合约连接测试失败")
    
    def add_symbol(self, symbol: str):
        """
        添加交易对
        
        参数:
            symbol: 交易对符号
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.logger.info(f"添加永续合约交易对: {symbol}")
    
    def add_symbols(self, symbols: List[str]):
        """
        批量添加交易对
        
        参数:
            symbols: 交易对列表
        """
        for symbol in symbols:
            self.add_symbol(symbol)
    
    def fetch_all_funding_rates(self) -> Dict[str, FundingRateData]:
        """
        获取所有交易对的资金费率
        
        返回:
            交易对资金费率字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        results = {}
        for symbol in self.symbols:
            try:
                funding_rate = self.fetcher.fetch_funding_rate(symbol)
                results[symbol] = funding_rate
            except Exception as e:
                self.logger.error(f"获取交易对 {symbol} 资金费率失败: {e}")
                results[symbol] = None
        
        # 统一持久化
        self.save_dict(f"{self.exchange}_{self.contract_type}_funding_rates", results)
        return results
    
    def fetch_all_open_interest(self) -> Dict[str, OpenInterestData]:
        """
        获取所有交易对的未平仓合约
        
        返回:
            交易对未平仓合约字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        results = {}
        for symbol in self.symbols:
            try:
                oi = self.fetcher.fetch_open_interest(symbol)
                results[symbol] = oi
            except Exception as e:
                self.logger.error(f"获取交易对 {symbol} 未平仓合约失败: {e}")
                results[symbol] = None
        
        # 统一持久化
        self.save_dict(f"{self.exchange}_{self.contract_type}_open_interest", results)
        return results
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        获取市场摘要
        
        返回:
            市场摘要信息
        """
        if not self.fetcher:
            self.init_fetcher()
        
        funding_rates = self.fetch_all_funding_rates()
        open_interests = self.fetch_all_open_interest()
        
        # 统计正负资金费率
        funding_rates_values = [v for v in funding_rates.values() if v]
        positive_funding = [v for v in funding_rates_values if v.funding_rate > 0]
        negative_funding = [v for v in funding_rates_values if v.funding_rate < 0]
        
        # 计算平均资金费率
        avg_funding_rate = 0
        if funding_rates_values:
            avg_funding_rate = sum(v.funding_rate for v in funding_rates_values) / len(funding_rates_values)
        
        # 统计未平仓合约
        open_interests_values = [v for v in open_interests.values() if v]
        total_open_interest_value = sum(v.open_interest_value for v in open_interests_values)
        avg_open_interest = 0
        if open_interests_values:
            avg_open_interest = sum(v.open_interest for v in open_interests_values) / len(open_interests_values)
        
        summary = {
            'exchange': self.exchange,
            'contract_type': self.contract_type,
            'total_symbols': len(self.symbols),
            'funding_rates': {
                'positive_count': len(positive_funding),
                'negative_count': len(negative_funding),
                'zero_count': len(funding_rates_values) - len(positive_funding) - len(negative_funding),
                'average_rate': avg_funding_rate,
                'max_positive_rate': max((v.funding_rate for v in positive_funding), default=0),
                'max_negative_rate': min((v.funding_rate for v in negative_funding), default=0),
            },
            'open_interest': {
                'total_value': total_open_interest_value,
                'average_oi': avg_open_interest,
                'max_oi': max((v.open_interest for v in open_interests_values), default=0),
            },
            'symbols': list(funding_rates.keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return summary

    def fetch_and_save(self,
                       symbol: str,
                       timeframe: str = "1h",
                       start_date: Optional[Union[datetime, str]] = None,
                       end_date: Optional[Union[datetime, str]] = None,
                       limit: Optional[int] = None) -> bool:
        """获取指定永续合约的K线数据并保存到本地存储（增量去重）。

        说明：SwapFetcher 当前未内置 bulk 接口，这里用循环分页方式在 [start_date, end_date] 内抓取。
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

                    # 若已经超出 end_ms，停止
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
            self.logger.error(f"保存 {symbol} 永续合约K线数据失败: {e}")
            return False
    
    def close(self):
        """关闭管理器"""
        if self.fetcher:
            self.fetcher.close()
            self.fetcher = None


# ==================== 测试函数 ====================

def test_swap_fetcher():
    """测试永续合约获取器"""
    print("=" * 60)
    print("永续合约获取器模块测试")
    print("=" * 60)
    
    # 测试基础功能
    print("\n1. 测试CCXTSwapFetcher基础功能:")
    try:
        # 创建获取器
        fetcher = CCXTSwapFetcher(exchange="binance", contract_type="linear")
        
        print(f"✅ 获取器创建成功: {fetcher}")
        print(f"✅ 交易所: {fetcher.exchange}")
        print(f"✅ 市场类型: {fetcher.market_type}")
        print(f"✅ 合约类型: {fetcher.contract_type}")
        
        # 测试连接
        if fetcher.test_connection():
            print("✅ 交易所连接测试成功")
        else:
            print("⚠️  交易所连接测试失败（可能是网络或代理问题）")
        
        # 获取可用交易对
        symbols = fetcher.get_available_symbols()
        print(f"✅ 获取到 {len(symbols)} 个可用永续合约交易对")
        if symbols:
            print(f"✅ 示例交易对: {symbols[0]}")
        
        # 测试K线数据获取
        print("\n2. 测试K线数据获取:")
        # 选择一个交易对进行测试
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
            # 尝试使用可用交易对中的第一个
            if symbols:
                test_symbol = symbols[0]
                print(f"⚠️  尝试使用可用交易对: {test_symbol}")
        
        # 测试资金费率获取
        print("\n3. 测试资金费率获取:")
        funding_rate = fetcher.fetch_funding_rate(test_symbol)
        if funding_rate:
            print(f"✅ 资金费率获取成功")
            print(f"  当前费率: {funding_rate.funding_rate:.6%}")
            print(f"  预测费率: {funding_rate.predicted_rate:.6%}")
        else:
            print("⚠️  资金费率获取失败，可能是API限制或数据不可用")
        
        # 测试未平仓合约获取
        print("\n4. 测试未平仓合约获取:")
        open_interest = fetcher.fetch_open_interest(test_symbol)
        if open_interest:
            print(f"✅ 未平仓合约获取成功")
            print(f"  未平仓量: {open_interest.open_interest}")
            print(f"  未平仓价值: {open_interest.open_interest_value}")
        else:
            print("⚠️  未平仓合约获取失败，可能是API限制或数据不可用")
        
        # 测试标记价格获取
        print("\n5. 测试标记价格获取:")
        mark_price = fetcher.fetch_mark_price(test_symbol)
        if mark_price:
            print(f"✅ 标记价格获取成功")
            print(f"  标记价格: {mark_price['mark_price']}")
            print(f"  指数价格: {mark_price['index_price']}")
            if mark_price['funding_rate'] is not None:
                print(f"  资金费率: {mark_price['funding_rate']:.6%}")
        else:
            print("⚠️  标记价格获取失败")
        
        # 测试基差计算
        print("\n6. 测试基差计算:")
        basis_info = fetcher.fetch_basis(test_symbol, spot_price=50000)
        if basis_info:
            print(f"✅ 基差计算成功")
            print(f"  永续价格: {basis_info['swap_price']}")
            print(f"  现货价格: {basis_info['spot_price']}")
            print(f"  基差: {basis_info['basis']}")
            print(f"  基差百分比: {basis_info['basis_percent']:.4f}%")
            print(f"  资金费率: {basis_info['funding_rate']:.6%}")
        else:
            print("⚠️  基差计算失败")
        
        # 测试合约信息获取
        print("\n7. 测试合约信息获取:")
        contract_info = fetcher.fetch_contract_info(test_symbol)
        if contract_info:
            print(f"✅ 合约信息获取成功")
            print(f"  合约大小: {contract_info.get('contract_size')}")
            print(f"  结算货币: {contract_info.get('settle_currency')}")
            print(f"  最大杠杆: {contract_info.get('max_leverage')}")
        else:
            print("⚠️  合约信息获取失败")
        
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
    
    # 测试永续合约数据管理器
    print("\n8. 测试永续合约数据管理器:")
    try:
        manager = SwapDataManager(exchange="binance", contract_type="linear")
        manager.add_symbols(["BTC/USDT", "ETH/USDT", "BNB/USDT"])
        
        print(f"✅ 管理器创建成功，交易对: {manager.symbols}")
        
        # 初始化获取器
        manager.init_fetcher()
        
        # 获取市场摘要
        summary = manager.get_market_summary()
        print(f"✅ 市场摘要:")
        print(f"  交易对数量: {summary['total_symbols']}")
        print(f"  平均资金费率: {summary['funding_rates']['average_rate']:.6%}")
        print(f"  正费率数量: {summary['funding_rates']['positive_count']}")
        print(f"  负费率数量: {summary['funding_rates']['negative_count']}")
        print(f"  总未平仓价值: {summary['open_interest']['total_value']:.2f}")
        
        manager.close()
        print("✅ 管理器关闭成功")
        
    except Exception as e:
        print(f"❌ 管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 永续合约获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    test_swap_fetcher()