"""
期货数据获取器模块
提供从交易所获取期货合约数据的功能
期货合约有固定到期日，与永续合约不同
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
    from ..data_models import OHLCVData, OrderBookData, TradeData, FutureContractData, TermStructureData, OpenInterestData
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
    from data_models import OHLCVData, OrderBookData, TradeData, FutureContractData, TermStructureData, OpenInterestData
    from utils.logger import get_logger, log_execution_time, log_errors
    from utils.cache import CacheManager, cached, cache_result
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from config import get_exchange_config, ExchangeSymbolFormats, get_market_config


# ==================== 期货数据获取器 ====================

class FutureFetcher(BaseFetcher):
    """
    期货数据获取器基类
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 contract_type: str = "inverse",  # inverse: 币本位合约, linear: USDT合约
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化期货数据获取器
        
        参数:
            exchange: 交易所名称
            contract_type: 合约类型 (inverse, linear)
            config: 配置字典
            cache_manager: 缓存管理器
        """
        super().__init__(
            exchange=exchange,
            market_type="future",
            config=config,
            cache_manager=cache_manager
        )
        
        self.contract_type = contract_type  # inverse: 币本位合约, linear: USDT合约
        
        # 加载交易所配置
        self.exchange_config = get_exchange_config(exchange)
        
        # 加载市场配置
        self.market_config = get_market_config("future")
        
        # 初始化交易所连接
        self.exchange_instance = None
        self._init_exchange()
        
        # 初始化合约信息
        self.future_contracts = {}  # 按到期日分类的合约
        self.active_contracts = []  # 活跃合约列表
        self._load_contracts_info()
    
    def _init_exchange(self):
        """初始化交易所连接"""
        try:
            self.logger.info(f"初始化期货交易所连接: {self.exchange}, 合约类型: {self.contract_type}")
            
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
                raise ValueError(f"不支持的交易所: {self.exchange}")
            
            # 获取CCXT配置
            ccxt_config = self.exchange_config.get_ccxt_config()
            
            # 设置默认类型为期货
            ccxt_config['options'] = ccxt_config.get('options', {})
            ccxt_config['options']['defaultType'] = 'future'
            
            # 根据合约类型设置
            if self.contract_type == "linear":
                ccxt_config['options']['defaultSettle'] = 'USDT'
            elif self.contract_type == "inverse":
                ccxt_config['options']['defaultSettle'] = 'BTC'  # 或对应的基础货币
            
            # 合并市场配置
            if 'ccxt_options' in self.market_config:
                ccxt_config['options'].update(self.market_config['ccxt_options'])
            
            # 创建交易所实例
            self.exchange_instance = exchange_class(ccxt_config)
            
            # 加载市场信息
            self._load_markets()
            
            self.logger.info(f"期货交易所连接初始化成功: {self.exchange}")
            
        except Exception as e:
            self.logger.error(f"期货交易所连接初始化失败: {e}")
            self.exchange_instance = None
    
    def _load_markets(self):
        """加载市场信息"""
        if not self.exchange_instance:
            return
        
        try:
            self.logger.info(f"加载 {self.exchange} 期货市场信息...")
            
            # 加载市场
            self.exchange_instance.load_markets()
            
            # 筛选期货交易对
            self.future_markets = {}
            for symbol, market in self.exchange_instance.markets.items():
                # 根据不同交易所的标识判断期货合约
                is_future = False
                
                if self.exchange == "binance":
                    # 币安期货: 检查是否是期货合约
                    if market.get('future', False) or '-PERP' in symbol:
                        is_future = True
                
                elif self.exchange == "okx":
                    # OKX期货: symbol包含特定到期日格式
                    if '-FUTURE' in symbol or re.match(r'.*-\d{6}$', symbol):
                        is_future = True
                
                elif self.exchange == "bybit":
                    # Bybit期货
                    if market.get('future', False):
                        is_future = True
                
                elif self.exchange == "huobi":
                    # 火币期货
                    if '-CW' in symbol or '-NW' in symbol or '-CQ' in symbol:
                        is_future = True
                
                if is_future and market.get('active', False):
                    self.future_markets[symbol] = market
            
            self.logger.info(f"加载 {len(self.future_markets)} 个期货交易对")
            
            # 缓存市场信息
            if self.cache_manager:
                self.cache_manager.set(
                    key=f"{self.exchange}_future_{self.contract_type}_markets",
                    data=self.future_markets,
                    ttl=3600,  # 缓存1小时
                    sub_dir='future'
                )
            
        except Exception as e:
            self.logger.error(f"加载市场信息失败: {e}")
            self.future_markets = {}
    
    def _load_contracts_info(self):
        """加载合约信息"""
        if not self.future_markets:
            return
        
        self.future_contracts.clear()
        self.active_contracts.clear()
        
        for symbol, market in self.future_markets.items():
            try:
                # 解析合约信息
                contract_info = self._parse_contract_info(symbol, market)
                
                if contract_info:
                    # 按基础交易对分组
                    base_symbol = contract_info['base_symbol']
                    if base_symbol not in self.future_contracts:
                        self.future_contracts[base_symbol] = []
                    
                    self.future_contracts[base_symbol].append(contract_info)
                    
                    # 如果是活跃合约
                    if contract_info.get('is_active', True):
                        self.active_contracts.append(symbol)
            except Exception as e:
                self.logger.warning(f"解析合约 {symbol} 信息失败: {e}")
        
        self.logger.info(f"加载 {len(self.active_contracts)} 个活跃期货合约")
    
    def _parse_contract_info(self, symbol: str, market: Dict) -> Optional[Dict]:
        """解析合约信息"""
        try:
            # 提取到期日等信息
            expiry_date = None
            expiry_timestamp = market.get('expiry', None)
            
            if expiry_timestamp:
                expiry_date = pd.Timestamp(expiry_timestamp, unit='ms')
            
            # 解析基础交易对
            if self.exchange == "binance":
                # 币安期货格式: BTC/USDT:USDT 或 BTC/USDT:USDT-20241231
                if ':' in symbol:
                    symbol_part, rest = symbol.split(':')
                    base, quote = symbol_part.split('/')
                    
                    # 检查是否是永续合约
                    if rest == 'USDT' or rest == 'BUSD' or rest == base:
                        # 永续合约
                        expiry_date = None
                    elif '-' in rest:
                        # 有到期日的期货合约
                        settle, expiry_str = rest.split('-')
                        # 解析到期日: YYYYMMDD
                        if len(expiry_str) == 8:
                            expiry_date = pd.Timestamp(f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:]}")
                    
                    base_symbol = f"{base}/{quote}"
                else:
                    # 可能是旧格式或非标准格式
                    base_symbol = symbol
                    
            elif self.exchange == "okx":
                # OKX期货格式: BTC-USDT-20241231 或 BTC-USDT-SWAP
                parts = symbol.split('-')
                if len(parts) >= 3:
                    base = parts[0]
                    quote = parts[1]
                    expiry_str = parts[2]
                    
                    base_symbol = f"{base}/{quote}"
                    
                    # 解析到期日: YYYYMMDD 或 SWAP（永续）
                    if expiry_str == 'SWAP':
                        expiry_date = None  # 永续合约
                    elif len(expiry_str) == 6 or len(expiry_str) == 8:
                        # 格式: YYMMDD 或 YYYYMMDD
                        if len(expiry_str) == 6:
                            expiry_str = '20' + expiry_str
                        expiry_date = pd.Timestamp(f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:]}")
                else:
                    base_symbol = symbol
            
            elif self.exchange == "bybit":
                # Bybit期货格式: BTCUSDT 或 BTCUSDT20241231
                if re.match(r'^[A-Z]+USDT$', symbol):
                    # 永续合约
                    base = symbol.replace('USDT', '')
                    quote = 'USDT'
                    base_symbol = f"{base}/{quote}"
                    expiry_date = None
                elif re.match(r'^[A-Z]+USDT\d{6,8}$', symbol):
                    # 期货合约
                    match = re.match(r'^([A-Z]+)USDT(\d{6,8})$', symbol)
                    if match:
                        base = match.group(1)
                        quote = 'USDT'
                        expiry_str = match.group(2)
                        base_symbol = f"{base}/{quote}"
                        
                        if len(expiry_str) == 6:
                            expiry_str = '20' + expiry_str
                        expiry_date = pd.Timestamp(f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:]}")
                else:
                    base_symbol = symbol
            
            else:
                base_symbol = symbol
            
            contract_info = {
                'contract_symbol': symbol,
                'base_symbol': base_symbol,
                'expiry_date': expiry_date,
                'contract_size': market.get('contractSize', 1.0),
                'settlement_asset': market.get('settle', 'USDT'),
                'contract_type': self.contract_type,
                'tick_size': market.get('precision', {}).get('price', 0.01),
                'lot_size': market.get('precision', {}).get('amount', 1.0),
                'is_active': market.get('active', True),
                'market_info': market,
                'is_perpetual': expiry_date is None  # 是否为永续合约
            }
            
            # 计算到期天数（如果不是永续合约）
            if expiry_date:
                days_to_expiry = (expiry_date - pd.Timestamp.now()).days
                contract_info['days_to_expiry'] = days_to_expiry
                contract_info['expiry_percent'] = max(0, min(100, (1 - days_to_expiry / 365) * 100))
            else:
                contract_info['days_to_expiry'] = None
                contract_info['expiry_percent'] = None
            
            return contract_info
            
        except Exception as e:
            self.logger.error(f"解析合约信息失败 {symbol}: {e}")
            return None

    @log_errors(reraise=False)
    @cached(key_prefix="future_open_interest_history", ttl=600, sub_dir="future")
    def fetch_open_interest_history(
                                   self,
                                   symbol: str,
                                   timeframe: str = "1h",
                                   since: Optional[Union[int, datetime, str]] = None,
                                   limit: Optional[int] = None,
                                   **kwargs) -> List[OpenInterestData]:
        """获取期货未平仓合约历史数据（若交易所支持 CCXT fetch_open_interest_history）。"""
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        formatted_symbol = self.format_symbol(symbol)

        since_timestamp = None
        if since:
            if isinstance(since, datetime):
                since_timestamp = int(since.timestamp() * 1000)
            elif isinstance(since, str):
                dt = pd.Timestamp(since).to_pydatetime()
                since_timestamp = int(dt.timestamp() * 1000)
            else:
                since_timestamp = since

        if limit is None:
            limit = self.config.get('open_interest_limit', 100)

        self.logger.info(
            f"获取期货未平仓合约历史: {formatted_symbol}, 时间间隔: {timeframe}, 开始时间: {since_timestamp}, 限制: {limit}"
        )

        try:
            if not hasattr(self.exchange_instance, 'fetch_open_interest_history'):
                raise AttributeError('fetch_open_interest_history not supported by exchange')

            oi_history = self.exchange_instance.fetch_open_interest_history(
                symbol=formatted_symbol,
                timeframe=timeframe,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {})
            )

            data_models: List[OpenInterestData] = []
            for oi in oi_history or []:
                try:
                    open_interest_value = oi.get('openInterestAmount', 0)
                    if open_interest_value is None:
                        open_interest_value = 0

                    open_interest_value_amount = oi.get('openInterestValue', 0)
                    if open_interest_value_amount is None:
                        open_interest_value_amount = 0

                    ts = oi.get('timestamp', pd.Timestamp.now())
                    data_model = OpenInterestData(
                        timestamp=pd.Timestamp(ts),
                        symbol=formatted_symbol,
                        exchange=self.exchange,
                        market_type="future",
                        open_interest=float(open_interest_value),
                        open_interest_value=float(open_interest_value_amount),
                        volume_24h=float(oi.get('baseVolume', 0) or 0),
                        turnover_24h=float(oi.get('quoteVolume', 0) or 0)
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换期货未平仓合约历史数据失败: {e}")
                    continue

            if len(data_models) > 1:
                for i in range(1, len(data_models)):
                    data_models[i].calculate_changes(data_models[i-1])

            self.logger.info(f"获取到 {len(data_models)} 条期货未平仓合约历史数据")
            return data_models

        except Exception as e:
            self.logger.error(f"获取期货未平仓合约历史失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的期货交易对
        
        返回:
            交易对列表
        """
        if not self.future_markets:
            self._load_markets()
        
        return list(self.future_markets.keys())
    
    def get_contracts_by_base(self, base_symbol: str) -> List[Dict]:
        """
        根据基础交易对获取合约列表
        
        参数:
            base_symbol: 基础交易对，如 BTC/USDT
            
        返回:
            合约信息列表
        """
        # 尝试格式化符号
        formatted_symbol = self.format_base_symbol(base_symbol)
        return self.future_contracts.get(formatted_symbol, [])
    
    def get_active_contracts(self) -> List[str]:
        """
        获取活跃合约列表
        
        返回:
            活跃合约列表
        """
        return self.active_contracts
    
    def get_nearest_contract(self, base_symbol: str, exclude_perpetual: bool = True) -> Optional[str]:
        """
        获取最近到期的合约
        
        参数:
            base_symbol: 基础交易对
            exclude_perpetual: 是否排除永续合约
            
        返回:
            最近到期合约符号
        """
        contracts = self.get_contracts_by_base(base_symbol)
        
        if not contracts:
            return None
        
        # 如果需要排除永续合约，过滤掉永续合约
        if exclude_perpetual:
            contracts = [c for c in contracts if not c.get('is_perpetual', True)]
        
        # 按到期日排序
        sorted_contracts = sorted(
            [c for c in contracts if c.get('expiry_date')], 
            key=lambda x: x['expiry_date']
        )
        
        if sorted_contracts:
            return sorted_contracts[0]['contract_symbol']
        
        return None
    
    def format_base_symbol(self, symbol: str) -> str:
        """
        格式化基础交易对符号
        
        参数:
            symbol: 原始交易对符号
            
        返回:
            标准化基础交易对符号
        """
        # 标准化基础符号格式为 "BTC/USDT"
        if '/' in symbol:
            return symbol
        else:
            # 尝试解析符号
            if re.match(r'^[A-Z]+USDT$', symbol):
                # BTCUSDT -> BTC/USDT
                base = symbol.replace('USDT', '')
                return f"{base}/USDT"
            elif re.match(r'^[A-Z]+BTC$', symbol):
                # ETHBTC -> ETH/BTC
                base = symbol.replace('BTC', '')
                return f"{base}/BTC"
            elif re.match(r'^[A-Z]+USD$', symbol):
                # BTCUSD -> BTC/USD
                base = symbol.replace('USD', '')
                return f"{base}/USD"
            else:
                # 无法解析，返回原符号
                return symbol
    
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
        
        # 如果有市场信息，检查是否存在
        if self.future_markets and symbol not in self.future_markets:
            # 尝试寻找类似的符号
            similar_symbols = [s for s in self.future_markets.keys() if symbol in s]
            if similar_symbols:
                self.logger.warning(f"期货合约 {symbol} 未找到，类似合约: {similar_symbols[:3]}")
            else:
                self.logger.warning(f"期货合约可能无效: {symbol}")
            return False
        
        return True
    
    def find_similar_symbol(self, symbol: str) -> Optional[str]:
        """
        查找类似的期货合约符号
        
        参数:
            symbol: 目标符号
            
        返回:
            最匹配的符号
        """
        if not self.future_markets:
            return None
        
        # 如果符号直接存在
        if symbol in self.future_markets:
            return symbol
        
        # 尝试不同的格式
        formats_to_try = []
        
        if self.exchange == "binance":
            # 币安格式
            if '/' in symbol:
                base, quote = symbol.split('/')
                formats_to_try.append(f"{base}{quote}")  # BTCUSDT
                formats_to_try.append(f"{base}/{quote}:{quote}")  # BTC/USDT:USDT
                if quote == 'USDT':
                    formats_to_try.append(f"{base}{quote}-PERP")  # BTCUSDT-PERP
            else:
                # 尝试添加分隔符
                if re.match(r'^[A-Z]+USDT$', symbol):
                    base = symbol.replace('USDT', '')
                    formats_to_try.append(f"{base}/USDT")
                    formats_to_try.append(f"{base}/USDT:USDT")
                    formats_to_try.append(f"{symbol}-PERP")
        
        # 尝试所有格式
        for fmt in formats_to_try:
            if fmt in self.future_markets:
                self.logger.info(f"找到类似合约: {symbol} -> {fmt}")
                return fmt
        
        return None
    
    def format_symbol(self, symbol: str, expiry_date: Optional[str] = None, is_perpetual: bool = False) -> str:
        """
        格式化交易对符号为交易所标准格式
        
        参数:
            symbol: 原始交易对符号
            expiry_date: 到期日期，格式: YYYYMMDD
            is_perpetual: 是否为永续合约
            
        返回:
            格式化后的交易对符号
        """
        # 如果已经包含到期信息，直接返回
        if ('-' in symbol and (symbol.endswith('PERP') or 
                             symbol.endswith('SWAP') or
                             re.match(r'.*-\d{6,8}$', symbol) or 
                             re.match(r'.*:\w+-\d{6,8}$', symbol))):
            return symbol
        
        # 标准化基础符号
        if '/' in symbol:
            base, quote = symbol.split('/')
        else:
            # 尝试解析
            if re.match(r'^[A-Z]+USDT$', symbol):
                base = symbol.replace('USDT', '')
                quote = 'USDT'
            elif re.match(r'^[A-Z]+BTC$', symbol):
                base = symbol.replace('BTC', '')
                quote = 'BTC'
            elif re.match(r'^[A-Z]+USD$', symbol):
                base = symbol.replace('USD', '')
                quote = 'USD'
            else:
                base, quote = symbol, 'USDT'  # 默认
        
        formatted = ""
        
        if self.exchange == "binance":
            if is_perpetual:
                # 永续合约: BTCUSDT 或 BTC/USDT:USDT
                formatted = f"{base}{quote}"
                # 检查是否存在PERP后缀版本
                if f"{formatted}-PERP" in self.future_markets:
                    formatted = f"{formatted}-PERP"
                elif f"{base}/{quote}:{quote}" in self.future_markets:
                    formatted = f"{base}/{quote}:{quote}"
            else:
                if expiry_date:
                    # 期货合约: BTC/USDT:USDT-20241231
                    formatted = f"{base}/{quote}:{quote}-{expiry_date}"
                else:
                    # 默认返回永续合约格式
                    formatted = f"{base}{quote}"
                    if f"{formatted}-PERP" in self.future_markets:
                        formatted = f"{formatted}-PERP"
        
        elif self.exchange == "okx":
            if is_perpetual:
                # 永续合约: BTC-USDT-SWAP
                formatted = f"{base}-{quote}-SWAP"
            else:
                # 期货合约: BTC-USDT-20241231
                if expiry_date:
                    # 确保日期格式为YYMMDD
                    if len(expiry_date) == 8:
                        expiry_date = expiry_date[2:]  # YYMMDD
                    formatted = f"{base}-{quote}-{expiry_date}"
                else:
                    formatted = f"{base}-{quote}-FUTURE"
        
        elif self.exchange == "bybit":
            if is_perpetual:
                # 永续合约: BTCUSDT
                formatted = f"{base}{quote}"
            else:
                # 期货合约: BTCUSDT20241231
                if expiry_date:
                    formatted = f"{base}{quote}{expiry_date}"
                else:
                    formatted = f"{base}{quote}"
        
        else:
            formatted = symbol
        
        self.logger.debug(f"格式化期货合约: {symbol} -> {formatted}")
        return formatted
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_ohlcv", ttl=300, sub_dir="future")
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = "1h",
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None,
                   **kwargs) -> List[OHLCVData]:
        """
        获取期货K线数据
        
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
        
        # 查找最匹配的符号
        formatted_symbol = self.find_similar_symbol(symbol)
        if not formatted_symbol:
            formatted_symbol = symbol
        
        # 验证交易对
        if not self.validate_symbol(formatted_symbol):
            self.logger.warning(f"期货合约可能无效: {formatted_symbol}")
            # 尝试使用永续合约格式
            if not formatted_symbol.endswith('-PERP') and not formatted_symbol.endswith('-SWAP'):
                if self.exchange == "binance":
                    alt_symbol = f"{formatted_symbol}-PERP"
                    if alt_symbol in self.future_markets:
                        self.logger.info(f"尝试使用永续合约格式: {alt_symbol}")
                        formatted_symbol = alt_symbol
        
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
            f"获取期货K线数据: {formatted_symbol}, "
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
                    data_model = OHLCVData.from_ccxt(
                        ohlcv=ohlcv,
                        symbol=formatted_symbol,
                        timeframe=timeframe,
                        exchange=self.exchange,
                        market_type="future"
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换期货OHLCV数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条期货K线数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取期货K线数据失败: {e}")
            # 如果失败，返回空列表而不是抛出异常
            if kwargs.get('raise_error', False):
                raise
            return []
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_orderbook", ttl=30, sub_dir="future")
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> Optional[OrderBookData]:
        """
        获取期货订单簿数据
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 查找最匹配的符号
        formatted_symbol = self.find_similar_symbol(symbol)
        if not formatted_symbol:
            formatted_symbol = symbol
        
        # 设置默认限制
        if limit is None:
            limit = self.config.get('orderbook_limit', 20)
        
        self.logger.info(f"获取期货订单簿: {formatted_symbol}, 深度: {limit}")
        
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
                f"期货订单簿获取成功: 买盘 {len(data_model.bids)} 个, "
                f"卖盘 {len(data_model.asks)} 个, "
                f"价差: {data_model.spread:.2f}"
            )
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取期货订单簿失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_trades", ttl=60, sub_dir="future")
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取期货成交数据
        
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
        
        # 查找最匹配的符号
        formatted_symbol = self.find_similar_symbol(symbol)
        if not formatted_symbol:
            formatted_symbol = symbol
        
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
            f"获取期货成交数据: {formatted_symbol}, "
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
                    self.logger.warning(f"转换期货成交数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条期货成交数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取期货成交数据失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []

    @log_errors(reraise=False)
    @cached(key_prefix="future_ticker", ttl=30, sub_dir="future")
    def fetch_ticker(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """获取期货行情（Ticker）。"""
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        formatted_symbol = self.format_symbol(symbol)
        self.logger.info(f"获取期货行情: {formatted_symbol}")

        try:
            ticker = self.exchange_instance.fetch_ticker(symbol=formatted_symbol, params=kwargs.get('params', {}))
            return ticker or {}
        except Exception as e:
            self.logger.error(f"获取期货行情失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return {}

    @log_errors(reraise=False)
    @cached(key_prefix="future_market_info", ttl=3600, sub_dir="future")
    def fetch_market_info(self, symbol: str = None) -> Dict[str, Any]:
        """获取期货市场信息（Market Info）。"""
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        if not symbol:
            return {}

        formatted_symbol = self.format_symbol(symbol)
        try:
            market = self.exchange_instance.market(formatted_symbol)
            return market or {}
        except Exception as e:
            self.logger.error(f"获取期货市场信息失败: {e}")
            return {}

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
        """获取期货某合约的“综合市场快照”。

        include 允许项（默认全开）：
        - ticker/orderbook/trades/market_info
        - contract_info/open_interest/basis/term_structure/settlement_price
        - ohlcv
        """
        include_set = set((include or [
            'ticker', 'orderbook', 'trades', 'market_info',
            'contract_info', 'open_interest', 'basis', 'term_structure', 'settlement_price',
        ]))

        formatted_symbol = self.format_symbol(symbol)
        snapshot: Dict[str, Any] = {
            'exchange': self.exchange,
            'market_type': 'future',
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

        if 'contract_info' in include_set and hasattr(self, 'fetch_contract_info'):
            snapshot['contract_info'] = self.fetch_contract_info(formatted_symbol)
        if 'open_interest' in include_set and hasattr(self, 'fetch_open_interest'):
            snapshot['open_interest'] = self.fetch_open_interest(formatted_symbol)
        if 'basis' in include_set and hasattr(self, 'fetch_basis'):
            snapshot['basis'] = self.fetch_basis(formatted_symbol)
        if 'term_structure' in include_set and hasattr(self, 'fetch_term_structure'):
            snapshot['term_structure'] = self.fetch_term_structure(formatted_symbol)
        if 'settlement_price' in include_set and hasattr(self, 'fetch_settlement_price'):
            snapshot['settlement_price'] = self.fetch_settlement_price(formatted_symbol)

        if 'ohlcv' in include_set:
            snapshot['ohlcv'] = self.fetch_ohlcv(formatted_symbol, timeframe=timeframe, limit=ohlcv_limit)

        return snapshot
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_contract_info", ttl=3600, sub_dir="future")
    def fetch_contract_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取期货合约信息
        
        参数:
            symbol: 交易对符号
            
        返回:
            合约信息
        """
        # 查找最匹配的符号
        formatted_symbol = self.find_similar_symbol(symbol)
        if not formatted_symbol:
            formatted_symbol = symbol
        
        # 从缓存中查找
        if formatted_symbol in self.future_markets:
            market = self.future_markets[formatted_symbol]
            
            # 解析合约信息
            contract_info = self._parse_contract_info(formatted_symbol, market)
            
            if contract_info:
                # 获取当前价格计算基差
                ohlcv = self.fetch_ohlcv(formatted_symbol, timeframe="1m", limit=1)
                if ohlcv:
                    contract_info['current_price'] = ohlcv[0].close
                
                return contract_info
        
        return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_basis", ttl=300, sub_dir="future")
    def fetch_basis(self, 
                   symbol: str, 
                   spot_price: Optional[float] = None,
                   annualize: bool = True) -> Dict[str, Any]:
        """
        计算基差（期货价格与现货价格的差异）
        
        参数:
            symbol: 交易对符号
            spot_price: 现货价格（如果不提供，会尝试获取）
            annualize: 是否年化计算
            
        返回:
            基差信息
        """
        # 查找最匹配的符号
        formatted_symbol = self.find_similar_symbol(symbol)
        if not formatted_symbol:
            formatted_symbol = symbol
        
        # 获取期货最新价格
        future_ohlcv = self.fetch_ohlcv(formatted_symbol, timeframe="1m", limit=1)
        
        if not future_ohlcv:
            return {}
        
        future_price = future_ohlcv[0].close
        
        # 获取合约信息
        contract_info = self.fetch_contract_info(formatted_symbol)
        
        if not contract_info:
            return {}
        
        # 如果是永续合约，跳过基差计算
        if contract_info.get('is_perpetual', True):
            self.logger.info(f"合约 {formatted_symbol} 是永续合约，跳过基差计算")
            return {
                'symbol': formatted_symbol,
                'base_symbol': contract_info['base_symbol'],
                'timestamp': pd.Timestamp.now(),
                'future_price': future_price,
                'spot_price': spot_price,
                'is_perpetual': True,
                'message': '永续合约无基差计算'
            }
        
        # 如果未提供现货价格，尝试从现货市场获取
        if spot_price is None:
            try:
                # 创建现货获取器，先尝试相对导入，再回退到绝对导入
                try:
                    from .spot_fetcher import CCXTSpotFetcher as SpotFetcher
                except Exception:
                    from crypto_data_system.fetchers.spot_fetcher import CCXTSpotFetcher as SpotFetcher

                spot_fetcher = SpotFetcher(exchange=self.exchange)
                spot_symbol = contract_info['base_symbol']
                spot_ohlcv = spot_fetcher.fetch_ohlcv(spot_symbol, timeframe="1m", limit=1)

                if spot_ohlcv:
                    spot_price = spot_ohlcv[0].close
                else:
                    spot_price = 0
            except Exception as e:
                self.logger.warning(f"获取现货价格失败: {e}")
                spot_price = 0
        
        # 计算基差
        if future_price > 0 and spot_price > 0:
            basis = future_price - spot_price
            basis_percent = (basis / spot_price) * 100
            
            # 计算年化基差
            annualized_basis = None
            if annualize and contract_info.get('days_to_expiry'):
                days_to_expiry = contract_info['days_to_expiry']
                if days_to_expiry > 0:
                    annualized_basis = (basis_percent * 365) / days_to_expiry
        else:
            basis = 0
            basis_percent = 0
            annualized_basis = 0
        
        # 计算滚动收益
        roll_yield = None
        if annualized_basis is not None:
            roll_yield = -annualized_basis  # 负基差为正的展期收益
        
        result = {
            'symbol': formatted_symbol,
            'base_symbol': contract_info['base_symbol'],
            'timestamp': pd.Timestamp.now(),
            'future_price': future_price,
            'spot_price': spot_price,
            'basis': basis,
            'basis_percent': basis_percent,
            'annualized_basis': annualized_basis,
            'roll_yield': roll_yield,
            'expiry_date': contract_info.get('expiry_date'),
            'days_to_expiry': contract_info.get('days_to_expiry'),
            'contract_type': contract_info.get('contract_type'),
            'is_perpetual': contract_info.get('is_perpetual', False),
            'is_contango': basis > 0,  # 升水
            'is_backwardation': basis < 0,  # 贴水
        }
        
        return result
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_term_structure", ttl=600, sub_dir="future")
    def fetch_term_structure(self, 
                            base_symbol: str,
                            include_all: bool = False) -> Optional[TermStructureData]:
        """
        获取期限结构数据
        
        参数:
            base_symbol: 基础交易对，如 BTC/USDT
            include_all: 是否包含非活跃合约
            
        返回:
            期限结构数据
        """
        # 获取该基础交易对的所有合约
        contracts = self.get_contracts_by_base(base_symbol)
        
        if not contracts:
            # 尝试使用标准化基础符号
            formatted_base = self.format_base_symbol(base_symbol)
            if formatted_base != base_symbol:
                contracts = self.get_contracts_by_base(formatted_base)
        
        if not contracts:
            self.logger.warning(f"未找到基础交易对 {base_symbol} 的期货合约")
            return None
        
        # 如果不包含所有合约，过滤非活跃合约
        if not include_all:
            contracts = [c for c in contracts if c.get('is_active', True)]
        
        # 获取现货价格
        spot_price = 0
        try:
            # 尝试导入SpotFetcher，处理可能得循环引用
            try:
                from .spot_fetcher import CCXTSpotFetcher as SpotFetcher
            except Exception:
                from crypto_data_system.fetchers.spot_fetcher import CCXTSpotFetcher as SpotFetcher

            # 创建临时现货 fetcher，注意避免频繁创建开销
            spot_fetcher = SpotFetcher(exchange=self.exchange)
            spot_ohlcv = spot_fetcher.fetch_ohlcv(base_symbol, timeframe="1m", limit=1)

            if spot_ohlcv:
                spot_price = spot_ohlcv[0].close
        except Exception as e:
            self.logger.warning(f"获取现货价格失败: {e}")
        
        # 为每个合约获取当前价格和计算基差
        contract_models = []
        for contract in contracts:
            contract_symbol = contract['contract_symbol']
            
            # 跳过永续合约（永续合约不属于期限结构）
            if contract.get('is_perpetual', True):
                continue
            
            # 获取合约当前价格
            future_ohlcv = self.fetch_ohlcv(contract_symbol, timeframe="1m", limit=1)
            
            # 初始化
            basis = None
            basis_percent = None
            annualized_basis = None
             
            if future_ohlcv and spot_price > 0:
                future_price = future_ohlcv[0].close
                basis = future_price - spot_price
                basis_percent = (basis / spot_price) * 100
                
                # 计算年化基差
                if contract.get('days_to_expiry') and contract['days_to_expiry'] > 0:
                   annualized_basis = (basis_percent * 365) / contract['days_to_expiry']
            
            # 创建FutureContractData对象
            contract_model = FutureContractData(
                timestamp=pd.Timestamp.now(),
                symbol=base_symbol,
                exchange=self.exchange,
                market_type="future",
                contract_symbol=contract_symbol,
                expiry_date=pd.Timestamp(contract.get('expiry_date')) if contract.get('expiry_date') else None,
                contract_size=float(contract.get('contract_size', 1.0) or 1.0),
                settlement_asset=contract.get('settlement_asset', 'USDT') or 'USDT',
                contract_type=contract.get('contract_type', 'inverse'),
                tick_size=float(contract.get('tick_size', 0.01) or 0.01),
                lot_size=float(contract.get('lot_size', 1.0) or 1.0),
                is_active=contract.get('is_active', True),
                basis=basis,
                annualized_basis=annualized_basis
            )
            # attach is_perpetual attribute for compatibility with callers
            try:
                contract_model.is_perpetual = contract.get('is_perpetual', False)
            except Exception:
                pass
            
            contract_models.append(contract_model)
        
        # 如果没有找到非永续合约，返回空
        if not contract_models:
            self.logger.info(f"基础交易对 {base_symbol} 没有找到非永续期货合约")
            return None
        
        # 创建TermStructureData对象
        term_structure = TermStructureData(
            timestamp=pd.Timestamp.now(),
            symbol=base_symbol,
            contracts=contract_models,
            spot_price=spot_price
        )
        
        return term_structure
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_open_interest", ttl=300, sub_dir="future")
    def fetch_open_interest(self, 
                           symbol: str,
                           **kwargs) -> Optional[OpenInterestData]:
        """
        获取期货未平仓合约数据
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            未平仓合约数据
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 查找最匹配的符号
        formatted_symbol = self.find_similar_symbol(symbol)
        if not formatted_symbol:
            formatted_symbol = symbol
        
        self.logger.info(f"获取期货未平仓合约: {formatted_symbol}")
        
        try:
            # 使用CCXT获取未平仓合约数据
            oi_info = self.exchange_instance.fetch_open_interest(
                symbol=formatted_symbol
            )
            
            # 使用Data Class标准化返回
            ts_val = oi_info.get('timestamp')
            if isinstance(ts_val, (int, float)):
                timestamp = pd.Timestamp(ts_val, unit='ms')
            else:
                timestamp = pd.Timestamp(ts_val if ts_val else pd.Timestamp.now())
            
            # Extract basic data
            open_interest = float(oi_info.get('openInterestAmount', 0.0) or 0.0)
            open_interest_value = float(oi_info.get('openInterestValue', 0.0) or 0.0)
            
            # Additional metrics if available
            volume_24h = float(oi_info.get('baseVolume', 0.0) or 0.0)
            turnover_24h = float(oi_info.get('quoteVolume', 0.0) or 0.0)
            
            result = OpenInterestData(
                timestamp=timestamp,
                symbol=formatted_symbol,
                exchange=self.exchange,
                market_type="future",
                open_interest=open_interest,
                open_interest_value=open_interest_value,
                volume_24h=volume_24h,
                turnover_24h=turnover_24h,
                open_interest_change=float(oi_info.get('openInterestChange', 0.0) or 0.0),
                open_interest_change_percent=float(oi_info.get('openInterestChangePercent', 0.0) or 0.0)
            )
            
            self.logger.info(
                f"期货未平仓合约获取成功: {result.open_interest:.2f} "
                f"(价值: {result.open_interest_value:.2f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取期货未平仓合约失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
            self.logger.error(f"获取期货未平仓合约失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="future_settlement_price", ttl=300, sub_dir="future")
    def fetch_settlement_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取期货结算价
        
        参数:
            symbol: 交易对符号
            
        返回:
            结算价格信息
        """
        if not self.exchange_instance:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 查找最匹配的符号
        formatted_symbol = self.find_similar_symbol(symbol)
        if not formatted_symbol:
            formatted_symbol = symbol
        
        self.logger.info(f"获取期货结算价格: {formatted_symbol}")
        
        try:
            # 获取合约信息
            contract_info = self.fetch_contract_info(formatted_symbol)
            
            if not contract_info:
                return None
            
            # 期货结算通常使用最后交易日的平均价或特定时间的价格
            # 这里我们返回当前价格作为近似
            ohlcv = self.fetch_ohlcv(formatted_symbol, timeframe="1m", limit=1)
            
            if not ohlcv:
                return None
            
            result = {
                'symbol': formatted_symbol,
                'timestamp': pd.Timestamp.now(),
                'settlement_price': ohlcv[0].close,
                'expiry_date': contract_info.get('expiry_date'),
                'days_to_expiry': contract_info.get('days_to_expiry'),
                'settlement_time': None,  # 实际结算时间
                'delivery_time': None,    # 交割时间
                'is_settled': False,      # 是否已结算
                'is_perpetual': contract_info.get('is_perpetual', False)
            }
            
            # 如果合约已到期，标记为已结算
            if contract_info.get('days_to_expiry') and contract_info['days_to_expiry'] <= 0:
                result['is_settled'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取期货结算价格失败: {e}")
            return None
    
    @log_errors(reraise=False)
    def fetch_roll_analysis(self, 
                           base_symbol: str,
                           front_month_symbol: Optional[str] = None,
                           back_month_symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        分析展期成本
        
        参数:
            base_symbol: 基础交易对
            front_month_symbol: 近月合约符号（可选）
            back_month_symbol: 远月合约符号（可选）
            
        返回:
            展期分析信息
        """
        # 获取期限结构
        term_structure = self.fetch_term_structure(base_symbol)
        
        if not term_structure or len(term_structure.contracts) < 2:
            return {}
        
        # 获取近月和远月合约
        contracts = sorted(term_structure.contracts, key=lambda x: x.expiry_date)
        
        front_contract = contracts[0]
        back_contract = contracts[1] if len(contracts) > 1 else None
        
        # 如果指定了合约，使用指定的合约
        if front_month_symbol:
            for contract in contracts:
                if contract.contract_symbol == front_month_symbol:
                    front_contract = contract
                    break
        
        if back_month_symbol and back_month_symbol:
            for contract in contracts:
                if contract.contract_symbol == back_month_symbol:
                    back_contract = contract
                    break
        
        if not front_contract or not back_contract:
            return {}
        
        # 获取合约价格
        front_price = None
        back_price = None
        
        front_ohlcv = self.fetch_ohlcv(front_contract.contract_symbol, timeframe="1m", limit=1)
        if front_ohlcv:
            front_price = front_ohlcv[0].close
        
        back_ohlcv = self.fetch_ohlcv(back_contract.contract_symbol, timeframe="1m", limit=1)
        if back_ohlcv:
            back_price = back_ohlcv[0].close
        
        if not front_price or not back_price:
            return {}
        
        # 计算展期成本
        days_between = (back_contract.expiry_date - front_contract.expiry_date).days
        price_difference = back_price - front_price
        
        # 计算年化展期成本
        if front_price > 0 and days_between > 0:
            roll_cost_percent = (price_difference / front_price) * 100
            annualized_roll_cost = (roll_cost_percent * 365) / days_between
        else:
            roll_cost_percent = 0
            annualized_roll_cost = 0
        
        # 展期收益（负的成本）
        roll_yield = -annualized_roll_cost
        
        result = {
            'base_symbol': base_symbol,
            'timestamp': pd.Timestamp.now(),
            'front_contract': front_contract.contract_symbol,
            'back_contract': back_contract.contract_symbol,
            'front_expiry': front_contract.expiry_date,
            'back_expiry': back_contract.expiry_date,
            'front_price': front_price,
            'back_price': back_price,
            'price_difference': price_difference,
            'days_between': days_between,
            'roll_cost_percent': roll_cost_percent,
            'annualized_roll_cost': annualized_roll_cost,
            'roll_yield': roll_yield,
            'spot_price': term_structure.spot_price,
            'is_contango': price_difference > 0,  # 升水市场
            'is_backwardation': price_difference < 0,  # 贴水市场
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
            self.logger.info(f"期货交易所连接测试成功，服务器时间: {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"期货交易所连接测试失败: {e}")
            return False
    
    def close(self):
        """关闭获取器，释放资源"""
        if self.exchange_instance:
            self.exchange_instance = None
        
        super().close()


# ==================== CCXT期货获取器 ====================

class CCXTFutureFetcher(FutureFetcher):
    """
    CCXT期货数据获取器
    
    使用CCXT库获取期货数据
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 contract_type: str = "linear",  # 默认使用USDT合约
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 **kwargs):
        """
        初始化CCXT期货获取器
        
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


# ==================== 期货数据管理器 ====================

try:
    from crypto_data_system.storage.data_manager import FileDataManager
except (ImportError, ModuleNotFoundError):
    # 脚本直接运行时的备用导入方式
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crypto_data_system.storage.data_manager import FileDataManager


class FutureDataManager(FileDataManager):
    """
    期货数据管理器
    
    管理多个交易对的期货数据获取
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 contract_type: str = "linear",  # 默认使用USDT合约
                 fetcher_config: Optional[Dict] = None,
                 root_dir: Optional[str] = None,
                 cache_manager: Optional[Any] = None,
                 save_json_merged: bool = False):
        """
        初始化期货数据管理器
        
        参数:
            exchange: 交易所名称
            contract_type: 合约类型
            fetcher_config: 获取器配置
        """
        self.exchange = exchange
        self.contract_type = contract_type
        self.fetcher_config = fetcher_config or {}
        self.fetcher = None
        self.base_symbols = []  # 基础交易对列表
        self.save_json_merged = bool(self.fetcher_config.get('save_json_merged', save_json_merged))
        
        # 初始化日志
        self.logger = get_logger(f"future_manager.{exchange}.{contract_type}")

        # 初始化文件存储（子目录按交易所/合约类型分组）
        super().__init__(root_dir=root_dir, sub_dir=f"future/{exchange}/{contract_type}", file_format="json", cache_manager=cache_manager)
    
    def init_fetcher(self):
        """初始化数据获取器"""
        if not self.fetcher:
            self.fetcher = CCXTFutureFetcher(
                exchange=self.exchange,
                contract_type=self.contract_type,
                config=self.fetcher_config
            )
            
            # 测试连接
            if not self.fetcher.test_connection():
                self.logger.warning(f"交易所 {self.exchange} 期货连接测试失败")
    
    def add_base_symbol(self, base_symbol: str):
        """
        添加基础交易对
        
        参数:
            base_symbol: 基础交易对符号
        """
        if base_symbol not in self.base_symbols:
            self.base_symbols.append(base_symbol)
            self.logger.info(f"添加基础交易对: {base_symbol}")
    
    def add_base_symbols(self, base_symbols: List[str]):
        """
        批量添加基础交易对
        
        参数:
            base_symbols: 基础交易对列表
        """
        for symbol in base_symbols:
            self.add_base_symbol(symbol)
    
    def fetch_all_term_structures(self) -> Dict[str, TermStructureData]:
        """
        获取所有基础交易对的期限结构
        
        返回:
            期限结构数据字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        results = {}
        for base_symbol in self.base_symbols:
            try:
                term_structure = self.fetcher.fetch_term_structure(base_symbol)
                results[base_symbol] = term_structure
            except Exception as e:
                self.logger.error(f"获取基础交易对 {base_symbol} 期限结构失败: {e}")
                results[base_symbol] = None
        
        return results
    
    def fetch_all_basis(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取所有合约的基差
        
        返回:
            基差数据字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        results = {}
        for base_symbol in self.base_symbols:
            try:
                # 获取该基础交易对的所有合约
                contracts = self.fetcher.get_contracts_by_base(base_symbol)
                
                basis_list = []
                for contract in contracts:
                    contract_symbol = contract['contract_symbol']
                    basis_info = self.fetcher.fetch_basis(contract_symbol)
                    
                    if basis_info:
                        basis_list.append(basis_info)
                
                results[base_symbol] = basis_list
                
            except Exception as e:
                self.logger.error(f"获取基础交易对 {base_symbol} 基差失败: {e}")
                results[base_symbol] = []
        
        return results
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        获取市场摘要
        
        返回:
            市场摘要信息
        """
        if not self.fetcher:
            self.init_fetcher()
        
        term_structures = self.fetch_all_term_structures()
        
        summary = {
            'exchange': self.exchange,
            'contract_type': self.contract_type,
            'total_base_symbols': len(self.base_symbols),
            'total_contracts': sum(len(ts.contracts) for ts in term_structures.values() if ts),
            'basis_analysis': {
                'contango_count': 0,
                'backwardation_count': 0,
                'average_annualized_basis': 0,
            },
            'term_structures': {}
        }
        
        # 分析基差情况
        total_basis = 0
        basis_count = 0
        
        for base_symbol, term_structure in term_structures.items():
            if term_structure:
                contracts = term_structure.contracts
                
                # 统计升水/贴水合约
                for contract in contracts:
                    if contract.basis is not None:
                        if contract.basis > 0:
                            summary['basis_analysis']['contango_count'] += 1
                        else:
                            summary['basis_analysis']['backwardation_count'] += 1
                        
                        if contract.annualized_basis is not None:
                            total_basis += abs(contract.annualized_basis)
                            basis_count += 1
                
                summary['term_structures'][base_symbol] = {
                    'contract_count': len(contracts),
                    'nearest_expiry': min((c.expiry_date for c in contracts if c.expiry_date), default=None),
                    'farthest_expiry': max((c.expiry_date for c in contracts if c.expiry_date), default=None),
                }
        
        # 计算平均年化基差
        if basis_count > 0:
            summary['basis_analysis']['average_annualized_basis'] = total_basis / basis_count
        
        return summary
    
    def calculate_optimal_roll(self, base_symbol: str, position_size: float) -> Dict[str, Any]:
        """
        计算最优展期策略
        
        参数:
            base_symbol: 基础交易对
            position_size: 持仓数量
            
        返回:
            展期策略分析
        """
        if not self.fetcher:
            self.init_fetcher()
        
        # 获取期限结构
        term_structure = self.fetcher.fetch_term_structure(base_symbol)
        
        if not term_structure or len(term_structure.contracts) < 2:
            return {}
        
        # 按到期日排序
        contracts = sorted(term_structure.contracts, key=lambda x: x.expiry_date)
        
        # 获取各合约价格
        contract_prices = {}
        for contract in contracts[:5]:  # 只分析前5个合约
            ohlcv = self.fetcher.fetch_ohlcv(contract.contract_symbol, timeframe="1m", limit=1)
            if ohlcv:
                contract_prices[contract.contract_symbol] = {
                    'price': ohlcv[0].close,
                    'expiry': contract.expiry_date,
                    'days_to_expiry': (contract.expiry_date - pd.Timestamp.now()).days
                }
        
        if len(contract_prices) < 2:
            return {}
        
        # 计算展期成本
        roll_strategies = []
        contract_symbols = list(contract_prices.keys())
        
        for i in range(len(contract_symbols) - 1):
            front_contract = contract_symbols[i]
            back_contract = contract_symbols[i + 1]
            
            front_data = contract_prices[front_contract]
            back_data = contract_prices[back_contract]
            
            days_between = back_data['days_to_expiry'] - front_data['days_to_expiry']
            price_difference = back_data['price'] - front_data['price']
            
            if front_data['price'] > 0 and days_between > 0:
                roll_cost_percent = (price_difference / front_data['price']) * 100
                annualized_roll_cost = (roll_cost_percent * 365) / days_between
                roll_yield = -annualized_roll_cost
                
                # 计算展期成本金额
                roll_cost_amount = position_size * front_data['price'] * (roll_cost_percent / 100)
                annualized_roll_cost_amount = position_size * front_data['price'] * (annualized_roll_cost / 100)
                
                strategy = {
                    'front_contract': front_contract,
                    'back_contract': back_contract,
                    'front_expiry': front_data['expiry'],
                    'back_expiry': back_data['expiry'],
                    'days_between': days_between,
                    'roll_cost_percent': roll_cost_percent,
                    'annualized_roll_cost': annualized_roll_cost,
                    'roll_yield': roll_yield,
                    'roll_cost_amount': roll_cost_amount,
                    'annualized_roll_cost_amount': annualized_roll_cost_amount,
                    'recommendation': 'hold' if roll_yield > 0.5 else 'roll' if roll_yield < -0.5 else 'neutral'
                }
                
                roll_strategies.append(strategy)
        
        # 找出最优策略
        optimal_strategy = None
        if roll_strategies:
            # 寻找展期收益最高的策略
            optimal_strategy = max(roll_strategies, key=lambda x: x['roll_yield'])
        
        result = {
            'base_symbol': base_symbol,
            'position_size': position_size,
            'spot_price': term_structure.spot_price,
            'contract_count': len(contracts),
            'roll_strategies': roll_strategies,
            'optimal_strategy': optimal_strategy,
            'timestamp': pd.Timestamp.now()
        }
        
        return result

    def fetch_and_save(self,
                       symbol: str,
                       timeframe: str = "1h",
                       start_date: Optional[Union[datetime, str]] = None,
                       end_date: Optional[Union[datetime, str]] = None,
                       limit: Optional[int] = None) -> bool:
        """获取指定期货合约的K线数据并保存到本地存储（增量去重）。

        说明：为统一行为，这里使用循环分页方式在 [start_date, end_date] 内抓取。
        """
        if not self.fetcher:
            self.init_fetcher()

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

            # 针对 binance：用户可能传入非期货市场符号，这里用 ccxt 直接试探哪个符号能被接受
            resolved_symbol = symbol
            try:
                candidates: List[str] = [symbol]

                # 如果 markets 已加载，先尝试从 markets 里找一个“最像”的符号
                ex = getattr(self.fetcher, 'exchange_instance', None)
                markets = getattr(ex, 'markets', None)
                if isinstance(markets, dict) and markets:
                    if '/' in symbol and ':' not in symbol:
                        base, quote = symbol.split('/')
                        prefix = f"{base}/{quote}"
                        guessed = [s for s in markets.keys() if str(s).startswith(prefix)]
                        candidates = guessed + candidates

                if self.exchange == 'binance':
                    if '/' in symbol and ':' not in symbol:
                        base, quote = symbol.split('/')
                        candidates.extend([
                            f"{base}{quote}",
                            f"{base}/{quote}:{quote}",
                            f"{base}{quote}-PERP",
                        ])
                    else:
                        if re.match(r'^[A-Z]+USDT$', symbol):
                            base = symbol.replace('USDT', '')
                            candidates.extend([
                                f"{base}/USDT",
                                f"{base}/USDT:USDT",
                                f"{symbol}-PERP",
                            ])
                        if re.match(r'^[A-Z]+USD$', symbol):
                            base = symbol.replace('USD', '')
                            candidates.extend([
                                f"{base}/USD",
                            ])

                # 去重并用 ccxt 直接探测（绕开装饰器的吞异常行为）
                uniq: List[str] = []
                seen = set()
                for c in candidates:
                    if not c:
                        continue
                    if c in seen:
                        continue
                    seen.add(c)
                    uniq.append(c)

                if ex is not None and hasattr(ex, 'fetch_ohlcv'):
                    for cand in uniq:
                        try:
                            ex.fetch_ohlcv(cand, timeframe=timeframe, since=start_ms, limit=1)
                            resolved_symbol = cand
                            break
                        except Exception:
                            continue
            except Exception:
                resolved_symbol = symbol

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
                        symbol=resolved_symbol,
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
                    symbol=resolved_symbol,
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
            symbol_clean = resolved_symbol.replace('/', '_').replace(':', '_')
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
            self.logger.error(f"保存 {symbol} 期货K线数据失败: {e}")
            return False
    
    def close(self):
        """关闭管理器"""
        if self.fetcher:
            self.fetcher.close()
            self.fetcher = None


# ==================== 测试函数 ====================

def test_future_fetcher():
    """测试期货获取器"""
    print("=" * 60)
    print("期货获取器模块测试")
    print("=" * 60)
    
    # 测试基础功能
    print("\n1. 测试CCXTFutureFetcher基础功能:")
    try:
        # 创建获取器（不使用真实API密钥）
        fetcher = CCXTFutureFetcher(exchange="binance", contract_type="linear")
        
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
        print(f"✅ 获取到 {len(symbols)} 个可用期货交易对")
        if symbols:
            print(f"✅ 示例交易对: {symbols[:3]}")
        
        # 测试K线数据获取
        print("\n2. 测试K线数据获取:")
        # 首先获取可用的交易对
        if symbols:
            # 选择一个交易对进行测试
            test_symbol = None
            # 优先选择BTC相关合约
            for sym in symbols:
                if 'BTC' in sym and ('PERP' in sym or 'SWAP' in sym or '-' in sym):
                    test_symbol = sym
                    break
            if not test_symbol and symbols:
                test_symbol = symbols[0]
            
            if test_symbol:
                print(f"✅ 使用交易对进行测试: {test_symbol}")
                
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
                
                # 测试合约信息获取
                print("\n3. 测试合约信息获取:")
                contract_info = fetcher.fetch_contract_info(test_symbol)
                if contract_info:
                    print(f"✅ 合约信息获取成功")
                    print(f"  合约符号: {contract_info.get('contract_symbol')}")
                    print(f"  到期日期: {contract_info.get('expiry_date')}")
                    print(f"  合约类型: {contract_info.get('contract_type')}")
                    print(f"  是否永续: {contract_info.get('is_perpetual', '未知')}")
                else:
                    print("⚠️  合约信息获取失败")
                
                # 测试基差计算（仅非永续合约）
                print("\n4. 测试基差计算:")
                basis_info = fetcher.fetch_basis(test_symbol, spot_price=50000)
                if basis_info:
                    print(f"✅ 基差计算成功")
                    if basis_info.get('is_perpetual'):
                        print(f"  合约类型: 永续合约（无基差计算）")
                    else:
                        print(f"  基差: {basis_info.get('basis', 0)}")
                        print(f"  基差百分比: {basis_info.get('basis_percent', 0):.4f}%")
                        if basis_info.get('annualized_basis'):
                            print(f"  年化基差: {basis_info.get('annualized_basis', 0):.4f}%")
                else:
                    print("⚠️  基差计算失败")
                
                # 测试期限结构获取
                print("\n5. 测试期限结构获取:")
                # 获取基础交易对
                if contract_info:
                    base_symbol = contract_info.get('base_symbol', 'BTC/USDT')
                    term_structure = fetcher.fetch_term_structure(base_symbol)
                    if term_structure:
                        print(f"✅ 期限结构获取成功")
                        print(f"  合约数量: {len(term_structure.contracts)}")
                        print(f"  现货价格: {term_structure.spot_price}")
                        
                        if term_structure.contracts:
                            non_perpetual_contracts = [c for c in term_structure.contracts if not c.is_perpetual]
                            if non_perpetual_contracts:
                                nearest = min(non_perpetual_contracts, key=lambda x: x.expiry_date)
                                farthest = max(non_perpetual_contracts, key=lambda x: x.expiry_date)
                                print(f"  最近到期: {nearest.expiry_date}")
                                print(f"  最远到期: {farthest.expiry_date}")
                            else:
                                print(f"  所有合约都是永续合约")
                    else:
                        print(f"⚠️  期限结构获取失败或该交易对没有期货合约")
        
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
    
    # 测试期货数据管理器
    print("\n6. 测试期货数据管理器:")
    try:
        manager = FutureDataManager(exchange="binance", contract_type="linear")
        manager.add_base_symbols(["BTC/USDT", "ETH/USDT"])
        
        print(f"✅ 管理器创建成功，基础交易对: {manager.base_symbols}")
        
        # 初始化获取器
        manager.init_fetcher()
        
        # 获取市场摘要
        summary = manager.get_market_summary()
        print(f"✅ 市场摘要: {summary['total_base_symbols']} 个基础交易对")
        
        manager.close()
        print("✅ 管理器关闭成功")
        
    except Exception as e:
        print(f"❌ 管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 期货获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    test_future_fetcher()