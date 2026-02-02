"""
数据获取器基础模块
定义所有数据获取器的基类和公共功能
包括同步和异步数据获取器
"""

import time
import asyncio
import threading
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# 导入项目内部模块
# 直接导入或创建最小化版本
try:
    # 尝试导入配置
    from config import get_exchange_config, get_data_fetch_config
    from utils.logger import get_logger
    from utils.cache import CacheManager
    from utils.date_utils import DateTimeUtils, split_date_range, calculate_timeframe_seconds
    from data_models import (
        OHLCVData, OrderBookData, TradeData, FundingRateData,
        OpenInterestData, LiquidationData, GreeksData
    )
except ImportError:
    # 如果导入失败，创建最小化版本
    print("部分模块导入失败，使用最小化版本...")
    
    # 模拟配置函数
    def get_exchange_config(exchange):
        return None
    
    def get_data_fetch_config():
        return {}
    
    # 模拟日志器
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # 模拟缓存管理器
    class CacheManager:
        def __init__(self):
            self.cache = {}
        
        def get(self, key):
            return self.cache.get(key)
        
        def set(self, key, value, ttl=None):
            self.cache[key] = value
    
    # 模拟日期工具
    class DateTimeUtils:
        @staticmethod
        def parse_date(date_str):
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

    def calculate_timeframe_seconds(timeframe: str) -> int:
        """最小化版本：将 timeframe（如 1m/5m/1h/1d/1w）转换为秒数。"""
        tf = str(timeframe or '').strip().lower()
        m = None
        try:
            import re
            m = re.fullmatch(r"(\d+)([mhdw])", tf)
        except Exception:
            m = None
        if not m:
            return 60
        n = int(m.group(1))
        unit = m.group(2)
        if unit == 'm':
            return n * 60
        if unit == 'h':
            return n * 3600
        if unit == 'd':
            return n * 86400
        if unit == 'w':
            return n * 7 * 86400
        return 60
    
    def split_date_range(start_date, end_date, timeframe, max_bars):
        """分割日期范围"""
        # 简化的实现
        return [(start_date, end_date)]
    
    # 数据模型类 - 修正字段顺序问题
    @dataclass
    class OHLCVData:
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float
        symbol: str = ""
        exchange: str = ""
        market_type: str = ""
        timeframe: str = ""
    
    @dataclass
    class OrderBookData:
        timestamp: datetime
        symbol: str
        exchange: str
        market_type: str
        bids: List[List[float]]
        asks: List[List[float]]
    
    @dataclass
    class TradeData:
        timestamp: datetime
        symbol: str
        exchange: str
        market_type: str
        side: str
        price: float
        amount: float
        cost: float
    
    @dataclass
    class FundingRateData:
        timestamp: datetime
        symbol: str
        exchange: str
        market_type: str
        funding_rate: float
        funding_time: datetime
    
    @dataclass
    class OpenInterestData:
        timestamp: datetime
        symbol: str
        exchange: str
        market_type: str
        open_interest: float
        open_interest_value: float = 0.0
        volume_24h: float = 0.0
        turnover_24h: float = 0.0
    
    @dataclass
    class LiquidationData:
        timestamp: datetime
        symbol: str
        exchange: str
        market_type: str
        side: str
        price: float
        amount: float
    
    @dataclass
    class GreeksData:
        timestamp: datetime
        symbol: str
        exchange: str
        market_type: str
        delta: float
        gamma: float
        vega: float
        theta: float
        rho: float

# ==================== 基础配置和工具 ====================

@dataclass
class FetcherConfig:
    """获取器配置"""
    name: str = "base_fetcher"
    exchange: str = "binance"
    market_type: str = "spot"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit: int = 1000  # 请求频率限制 (ms)
    timeout: int = 30000  # 超时时间 (ms)
    retry_count: int = 3  # 重试次数
    max_retry_delay: int = 30  # 最大重试延迟 (秒)
    proxy_url: Optional[str] = None  # 代理地址
    enable_cache: bool = True  # 是否启用缓存
    # 额外配置（保留未被显式建模的配置项，例如 ohlcv_limit / orderbook_limit 等）
    extras: Dict[str, Any] = field(default_factory=dict)
    cache_ttl: Dict[str, int] = field(default_factory=lambda: {
        'tick': 60,      # 1分钟
        'minute': 300,   # 5分钟
        'hour': 3600,    # 1小时
        'day': 86400     # 1天
    })
    verbose: bool = False  # 详细日志

    def get(self, key: str, default: Any = None) -> Any:
        """兼容 dict.get 用法：优先读显式字段，其次读 extras。"""
        if hasattr(self, key):
            value = getattr(self, key)
            return default if value is None else value
        return self.extras.get(key, default)
    
    def __post_init__(self):
        """后初始化处理"""
        # 从全局配置加载
        try:
            global_exchange_config = get_exchange_config(self.exchange)
            if global_exchange_config:
                self.proxy_url = self.proxy_url or global_exchange_config.proxy_url
                self.rate_limit = global_exchange_config.rate_limit
        except:
            pass


@dataclass
class RequestStats:
    """请求统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    requests_by_type: Dict[str, int] = field(default_factory=lambda: {
        'ohlcv': 0,
        'orderbook': 0,
        'trades': 0,
        'ticker': 0,
        'funding_rate': 0,
        'open_interest': 0
    })
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """平均响应时间"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests
    
    def add_request(self, request_type: str, success: bool, response_time: float):
        """添加请求记录"""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.total_response_time += response_time
        else:
            self.failed_requests += 1
            
        self.last_request_time = datetime.now()
        
        # 统计请求类型
        if request_type in self.requests_by_type:
            self.requests_by_type[request_type] += 1
    
    def reset(self):
        """重置统计"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.last_request_time = None
        for key in self.requests_by_type:
            self.requests_by_type[key] = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'average_response_time': self.average_response_time,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'requests_by_type': self.requests_by_type.copy()
        }


# ==================== 基础异常类 ====================

class DataFetcherError(Exception):
    """数据获取器基础异常"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class ConnectionError(DataFetcherError):
    """连接异常"""
    pass


class RateLimitError(DataFetcherError):
    """频率限制异常"""
    pass


class AuthenticationError(DataFetcherError):
    """认证异常"""
    pass


class DataFormatError(DataFetcherError):
    """数据格式异常"""
    pass


class ExchangeError(DataFetcherError):
    """交易所错误"""
    pass


# ==================== 基础数据获取器 ====================

class BaseFetcher(ABC):
    """
    数据获取器基类
    定义所有数据获取器的公共接口和功能
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 market_type: str = "spot",
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化数据获取器
        
        参数:
            exchange: 交易所名称
            market_type: 市场类型
            config: 配置字典
            cache_manager: 缓存管理器
        """
        self.exchange = exchange.lower()
        self.market_type = market_type.lower()
        self.config = self._load_config(config)
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 缓存管理器
        self.cache_manager = cache_manager or CacheManager()
        
        # 请求统计
        self.request_stats = RequestStats()
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        
        # 请求锁和频率限制
        self.request_lock = threading.RLock()
        self.last_request_time = 0
        self.min_request_interval = self.config.rate_limit / 1000  # 转换为秒
        
        # 状态标记
        self.is_initialized = False
        self.is_running = False
        
        # CCXT实例
        self.ccxt_exchange = None
        
        self.logger.info(f"初始化数据获取器: 交易所={self.exchange}, 市场类型={self.market_type}")
    
    def _load_config(self, config: Optional[Dict]) -> FetcherConfig:
        """加载配置"""
        if config is None:
            config = {}

        # 保留未被显式建模的额外配置项
        known_keys = {
            'name', 'exchange', 'market_type',
            'api_key', 'api_secret',
            'rate_limit', 'timeout',
            'retry_count', 'max_retry_delay',
            'proxy_url', 'enable_cache',
            'cache_ttl', 'verbose'
        }
        extras = {k: v for k, v in config.items() if k not in known_keys}
        
        # 获取默认的 cache_ttl 配置
        default_cache_ttl = {
            'tick': 60,      # 1分钟
            'minute': 300,   # 5分钟
            'hour': 3600,    # 1小时
            'day': 86400     # 1天
        }
        
        return FetcherConfig(
            name=config.get('name', f"{self.exchange}_{self.market_type}_fetcher"),
            exchange=config.get('exchange', self.exchange),
            market_type=config.get('market_type', self.market_type),
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret'),
            rate_limit=config.get('rate_limit', 1000),
            timeout=config.get('timeout', 30000),
            retry_count=config.get('retry_count', 3),
            max_retry_delay=config.get('max_retry_delay', 30),
            proxy_url=config.get('proxy_url'),
            enable_cache=config.get('enable_cache', True),
            extras=extras,
            cache_ttl=config.get('cache_ttl', default_cache_ttl),
            verbose=config.get('verbose', False)
        )
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志器"""
        logger_name = f"fetcher.{self.exchange}.{self.market_type}"
        return get_logger(logger_name)
    
    def _rate_limit(self):
        """频率限制控制"""
        with self.request_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                if self.config.verbose:
                    self.logger.debug(f"频率限制: 等待 {sleep_time:.3f} 秒")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _record_request(self, request_type: str, success: bool, start_time: float):
        """记录请求统计"""
        response_time = time.time() - start_time
        self.request_stats.add_request(request_type, success, response_time)
        
        if success:
            self.consecutive_errors = 0
        else:
            self.error_count += 1
            self.last_error_time = datetime.now()
            self.consecutive_errors += 1
    
    def _retry_request(self, func: Callable, *args, request_type: str = "unknown", **kwargs):
        """
        带重试的请求执行
        
        参数:
            func: 要执行的函数
            *args: 函数参数
            request_type: 请求类型
            **kwargs: 函数关键字参数
            
        返回:
            函数执行结果
        """
        max_retries = self.config.retry_count
        max_retry_delay = self.config.max_retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()
                start_time = time.time()
                
                result = func(*args, **kwargs)
                
                self._record_request(request_type, True, start_time)
                return result
                
            except RateLimitError as e:
                # 频率限制错误，增加等待时间
                self.logger.warning(f"频率限制错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, max_retry_delay)  # 指数退避，有上限
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self._record_request(request_type, False, start_time)
                    raise
                    
            except (ConnectionError, TimeoutError) as e:
                # 连接或超时错误，重试
                self.logger.warning(f"连接错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait_time = min(1 + attempt, max_retry_delay)
                    time.sleep(wait_time)
                else:
                    self._record_request(request_type, False, start_time)
                    raise
                    
            except AuthenticationError as e:
                # 认证错误，不重试
                self.logger.error(f"认证错误: {e}")
                self._record_request(request_type, False, start_time)
                raise
                
            except Exception as e:
                # 其他错误
                self.logger.error(f"请求错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait_time = min(1 + attempt, max_retry_delay)
                    time.sleep(wait_time)
                else:
                    self._record_request(request_type, False, start_time)
                    raise
        
        # 所有重试都失败
        raise DataFetcherError(f"请求失败，已重试 {max_retries} 次")
    
    def _get_cache_key(self, 
                      method: str, 
                      symbol: str = None, 
                      timeframe: str = None,
                      **params) -> str:
        """生成缓存键"""
        key_parts = [self.exchange, self.market_type, method]
        
        if symbol:
            key_parts.append(symbol.replace('/', '_'))
        
        if timeframe:
            key_parts.append(timeframe)
        
        # 添加参数哈希
        if params:
            import hashlib
            param_str = str(sorted(params.items()))
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_parts.append(param_hash)
        
        return "_".join(key_parts)
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.config.enable_cache:
            return None
        
        return self.cache_manager.get(key, sub_dir=self.market_type)
    
    def _cache_set(self, key: str, data: Any, ttl: int = None):
        """设置缓存数据"""
        if not self.config.enable_cache or data is None:
            return
        
        # Avoid caching empty results so future runs can retry fetching fresh data
        if isinstance(data, pd.DataFrame) and data.empty:
            return

        if isinstance(data, (list, tuple, set, dict)) and len(data) == 0:
            return

        if isinstance(data, np.ndarray) and data.size == 0:
            return

        if ttl is None:
            # 根据数据类型设置默认TTL
            if isinstance(data, list) and len(data) > 0:
                # 如果是时间序列数据，根据时间间隔设置TTL
                ttl = self.config.cache_ttl.get('hour', 3600)
            else:
                ttl = 300  # 默认5分钟
        
        self.cache_manager.set(key, data, ttl=ttl, sub_dir=self.market_type)
    
    def _get_ccxt_timeframe(self, timeframe: str) -> str:
        """将标准时间间隔转换为CCXT格式"""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }
        return timeframe_map.get(timeframe, timeframe)
    
    @abstractmethod
    def _init_exchange(self):
        """初始化交易所实例（抽象方法）"""
        pass
    
    @abstractmethod
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = "1h",
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None,
                   **kwargs) -> List[OHLCVData]:
        """
        获取K线数据（抽象方法）
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCVData列表
        """
        pass
    
    @abstractmethod
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> Optional[OrderBookData]:
        """
        获取订单簿数据（抽象方法）
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            OrderBookData对象
        """
        pass
    
    @abstractmethod
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取成交数据（抽象方法）
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            TradeData列表
        """
        pass
    
    def fetch_ticker(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        获取行情数据（可选实现）
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            行情数据字典
        """
        if self.ccxt_exchange and hasattr(self.ccxt_exchange, 'fetch_ticker'):
            try:
                return self._retry_request(
                    self.ccxt_exchange.fetch_ticker,
                    symbol,
                    request_type='ticker'
                )
            except Exception as e:
                self.logger.error(f"获取行情数据失败: {e}")
                return {}
        else:
            self.logger.warning("fetch_ticker 方法未实现")
            return {}
    
    def fetch_market_info(self, symbol: str = None) -> Dict[str, Any]:
        """
        获取市场信息（可选实现）
        
        参数:
            symbol: 交易对符号（可选）
            
        返回:
            市场信息字典
        """
        if self.ccxt_exchange:
            try:
                if symbol:
                    markets = self.ccxt_exchange.markets
                    return markets.get(symbol, {})
                else:
                    return self.ccxt_exchange.markets
            except Exception as e:
                self.logger.error(f"获取市场信息失败: {e}")
                return {}
        else:
            self.logger.warning("市场信息获取未实现")
            return {}
    
    def fetch_funding_rate(self, symbol: str, **kwargs) -> Optional[FundingRateData]:
        """
        获取资金费率数据（永续合约专用）
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            FundingRateData对象
        """
        if self.market_type not in ['swap', 'future']:
            self.logger.warning(f"资金费率仅适用于永续合约，当前市场类型: {self.market_type}")
            return None
        
        if self.ccxt_exchange and hasattr(self.ccxt_exchange, 'fetch_funding_rate'):
            try:
                funding_rate_data = self._retry_request(
                    self.ccxt_exchange.fetch_funding_rate,
                    symbol,
                    request_type='funding_rate'
                )
                return FundingRateData(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    exchange=self.exchange,
                    market_type=self.market_type,
                    funding_rate=float(funding_rate_data.get('fundingRate', 0)),
                    funding_time=datetime.fromtimestamp(funding_rate_data.get('fundingTime', 0) / 1000)
                )
            except Exception as e:
                self.logger.error(f"获取资金费率失败: {e}")
                return None
        else:
            self.logger.warning("fetch_funding_rate 方法未实现")
            return None
    
    def fetch_open_interest(self, symbol: str, **kwargs) -> Optional[OpenInterestData]:
        """
        获取未平仓合约数据
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            OpenInterestData对象
        """
        if self.market_type not in ['swap', 'future', 'option']:
            self.logger.warning(f"未平仓合约数据仅适用于衍生品市场，当前市场类型: {self.market_type}")
            return None
        
        if self.ccxt_exchange and hasattr(self.ccxt_exchange, 'fetch_open_interest'):
            try:
                oi_data = self._retry_request(
                    self.ccxt_exchange.fetch_open_interest,
                    symbol,
                    request_type='open_interest'
                )
                return OpenInterestData(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    exchange=self.exchange,
                    market_type=self.market_type,
                    open_interest=float(oi_data.get('openInterest', 0)),
                    open_interest_value=float(oi_data.get('openInterestValue', 0)),
                    volume_24h=float(oi_data.get('volume24h', 0)),
                    turnover_24h=float(oi_data.get('turnover24h', 0))
                )
            except Exception as e:
                self.logger.error(f"获取未平仓合约数据失败: {e}")
                return None
        else:
            self.logger.warning("fetch_open_interest 方法未实现")
            return None
    
    def fetch_ohlcv_bulk(self,
                        symbol: str,
                        start_date: Union[str, datetime],
                        end_date: Union[str, datetime],
                        timeframe: str = "1h",
                        max_bars_per_request: int = 1000,
                        **kwargs) -> pd.DataFrame:
        """
        批量获取K线数据
        
        参数:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间间隔
            max_bars_per_request: 每次请求最大条数
            **kwargs: 额外参数
            
        返回:
            K线数据DataFrame
        """
        # 转换日期格式（先做交易所特定的窗口裁剪，再生成缓存键）
        if isinstance(start_date, str):
            start_dt = pd.Timestamp(start_date)
        else:
            start_dt = pd.Timestamp(start_date)

        if isinstance(end_date, str):
            end_dt = pd.Timestamp(end_date)
        else:
            end_dt = pd.Timestamp(end_date)

        # 统一时区：将 tz-aware 时间戳转换为 UTC 后再去掉 tz 信息（保持 tz-naive）。
        # 否则会触发 pandas 比较异常：Cannot compare tz-naive and tz-aware timestamps
        try:
            if getattr(start_dt, 'tzinfo', None) is not None:
                start_dt = start_dt.tz_convert('UTC').tz_localize(None)
            if getattr(end_dt, 'tzinfo', None) is not None:
                end_dt = end_dt.tz_convert('UTC').tz_localize(None)
        except Exception:
            pass

        # Gate 限制：只能请求“最近 N 根K线”窗口（报错: Candlestick too long ago. Maximum 10000 points ago are allowed）
        exchange_lower = str(self.exchange).lower()
        if exchange_lower == 'gate':
            max_lookback_bars = int(self.config.get('gate_max_ohlcv_lookback_bars', 10000))
            seconds_per_bar = int(calculate_timeframe_seconds(timeframe))
            now_ts = pd.Timestamp.utcnow()
            # 兼容：当前 pandas 版本下 utcnow() 返回 tz-aware(UTC)，与 tz-naive 的 start/end 比较会报错
            try:
                if getattr(now_ts, 'tzinfo', None) is not None:
                    now_ts = now_ts.tz_convert('UTC').tz_localize(None)
            except Exception:
                pass
            earliest_allowed = now_ts - pd.Timedelta(seconds=max_lookback_bars * seconds_per_bar)

            # end 也不能早于允许窗口，否则 Gate 直接拒绝（无论 limit 多小）
            if end_dt < earliest_allowed:
                msg = (
                    f"Gate OHLCV 历史深度限制：仅支持最近 {max_lookback_bars} 根 {timeframe}。"
                    f" 当前请求 end={end_dt} 早于允许最早时间 {earliest_allowed}，无法从 Gate 获取该历史区间。"
                )
                self.logger.error(msg)
                if kwargs.get('raise_error', False):
                    raise ValueError(msg)
                # 返回空 DF，避免无意义重试
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe'])

            if start_dt < earliest_allowed:
                self.logger.warning(
                    f"Gate OHLCV 历史深度限制：start={start_dt} 早于允许最早 {earliest_allowed}，将自动裁剪到允许窗口。"
                )
                start_dt = earliest_allowed

            # 额外：Gate 单次返回上限与稳定性，保守降低每次请求条数
            max_bars_per_request = min(max_bars_per_request, 500)

        # 生成缓存键（使用裁剪后的 start/end）
        cache_key = self._get_cache_key(
            'ohlcv_bulk',
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_dt,
            end_date=end_dt
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取批量K线数据: {symbol} {timeframe}")
            return cached_data
        
        # 分割日期范围
        chunks = split_date_range(
            start_date=start_dt,
            end_date=end_dt,
            timeframe=timeframe,
            max_bars=max_bars_per_request
        )
        
        all_data = []
        chunk_count = 0
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            self.logger.info(
                f"批量获取第 {i + 1}/{len(chunks)} 段数据: "
                f"{chunk_start.strftime('%Y-%m-%d %H:%M')} 到 "
                f"{chunk_end.strftime('%Y-%m-%d %H:%M')}"
            )
            
            try:
                # 获取该时间段的数据
                data_list = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=chunk_start,
                    limit=max_bars_per_request,
                    **kwargs
                )
                
                # 转换为DataFrame并添加到列表
                if data_list:
                    data_dicts = []
                    for idx, data in enumerate(data_list):
                        try:
                            # Check if it's an OHLCVData-like object (using hasattr instead of isinstance)
                            if hasattr(data, 'timestamp') and hasattr(data, 'open') and hasattr(data, 'close'):
                                # Handle OHLCVData object or similar
                                data_dicts.append({
                                    'timestamp': data.timestamp,
                                    'open': data.open,
                                    'high': data.high,
                                    'low': data.low,
                                    'close': data.close,
                                    'volume': data.volume,
                                    'symbol': symbol,
                                    'timeframe': timeframe
                                })
                            elif isinstance(data, dict):
                                # Handle dict format
                                data_dicts.append({
                                    'timestamp': data.get('timestamp'),
                                    'open': data.get('open'),
                                    'high': data.get('high'),
                                    'low': data.get('low'),
                                    'close': data.get('close'),
                                    'volume': data.get('volume', 0),
                                    'symbol': symbol,
                                    'timeframe': timeframe
                                })
                            elif isinstance(data, (list, tuple)) and len(data) >= 6:
                                # Handle CCXT raw format [timestamp, open, high, low, close, volume]
                                data_dicts.append({
                                    'timestamp': pd.Timestamp(data[0], unit='ms'),
                                    'open': float(data[1]),
                                    'high': float(data[2]),
                                    'low': float(data[3]),
                                    'close': float(data[4]),
                                    'volume': float(data[5]),
                                    'symbol': symbol,
                                    'timeframe': timeframe
                                })
                            else:
                                if idx == 0:  # Only log once per chunk to avoid spam
                                    self.logger.warning(f"第 {i + 1} 段：未知数据类型 {type(data)}, 属性: {dir(data)[:5]}")
                        except Exception as convert_error:
                            if idx < 3:  # Only log first few errors
                                self.logger.debug(f"数据转换错误，跳过此条: {convert_error}")
                            continue
                    
                    if data_dicts:
                        try:
                            chunk_df = pd.DataFrame(data_dicts)
                            chunk_df.set_index('timestamp', inplace=True)
                            all_data.append(chunk_df)
                            chunk_count += 1
                        except Exception as df_error:
                            self.logger.error(f"创建DataFrame失败: {df_error}")
                    else:
                        self.logger.warning(f"第 {i + 1} 段数据转换为空（原始数据 {len(data_list)} 条，类型：{type(data_list[0]) if data_list else 'N/A'}）")
                
                # 避免频率限制
                if i < len(chunks) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"获取第 {i + 1} 段数据失败: {e}")
                if kwargs.get('raise_error', False):
                    raise
                continue
        
        # 合并所有数据
        if all_data:
            df = pd.concat(all_data, axis=0)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            
            # 缓存结果 (1天缓存)
            self._cache_set(cache_key, df, ttl=86400)
            
            self.logger.info(f"批量获取完成: {symbol} {timeframe}, 共 {chunk_count} 个时间段，{len(df)} 条数据")
            return df
        else:
            self.logger.warning(f"未获取到数据: {symbol} {timeframe}（尝试了 {len(chunks)} 个时间段）")
            return pd.DataFrame()
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证交易对是否有效
        支持不同交易所的格式：Binance(/或:)、OKX(-)等
        
        参数:
            symbol: 交易对符号
            
        返回:
            是否有效
        """
        # 基础验证
        if not symbol or not isinstance(symbol, str):
            return False
        
        # 检查是否包含必要的分隔符（支持 / 、- 、: 分隔符）
        # Binance: BTC/USDT 或 BTC/USDT:USDT
        # OKX: BTC-USDT 或 BTC-USDT-SWAP
        # Gate: BTC/USDT
        # Coinbase: BTC/USD
        has_separator = '/' in symbol or ':' in symbol or '-' in symbol
        
        if not has_separator:
            self.logger.warning(f"交易对可能格式不正确: {symbol}")
            # 有些交易所可能使用不同格式，这里不严格限制
            return True
        
        return True
    
    def format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号为交易所标准格式
        
        参数:
            symbol: 原始交易对符号
            
        返回:
            格式化后的交易对符号
        """
        # 基础格式化：去除空格，转为大写
        formatted = symbol.strip().upper()
        
        # 确保使用正斜杠分隔
        if '-' in formatted and '/' not in formatted:
            formatted = formatted.replace('-', '/')
        
        return formatted
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的交易对列表
        
        返回:
            交易对列表
        """
        if self.ccxt_exchange:
            try:
                markets = self.fetch_market_info()
                symbols = []
                
                for symbol, market_info in markets.items():
                    # 过滤活跃的交易对
                    if market_info.get('active', False):
                        symbols.append(symbol)
                
                return sorted(symbols)
            except Exception as e:
                self.logger.error(f"获取交易对列表失败: {e}")
                return []
        else:
            self.logger.warning("获取交易对列表未实现")
            return []
    
    def test_connection(self) -> bool:
        """
        测试交易所连接
        
        返回:
            连接是否成功
        """
        try:
            # 尝试获取一个常见交易对的信息
            test_symbol = "BTC/USDT"
            if self.validate_symbol(test_symbol):
                ticker = self.fetch_ticker(test_symbol)
                return ticker is not None and 'last' in ticker
            return False
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取获取器状态
        
        返回:
            状态信息字典
        """
        return {
            'exchange': self.exchange,
            'market_type': self.market_type,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'config': {
                'rate_limit': self.config.rate_limit,
                'timeout': self.config.timeout,
                'retry_count': self.config.retry_count,
                'enable_cache': self.config.enable_cache,
            },
            'request_stats': self.request_stats.to_dict(),
            'error_count': self.error_count,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'consecutive_errors': self.consecutive_errors
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.request_stats.reset()
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        self.logger.info("统计信息已重置")
    
    def initialize(self):
        """初始化获取器"""
        try:
            self.logger.info(f"初始化 {self.exchange} {self.market_type} 获取器...")
            
            # 初始化交易所实例
            self._init_exchange()
            
            # 测试连接
            if self.test_connection():
                self.is_initialized = True
                self.is_running = True
                self.logger.info(f"{self.exchange} {self.market_type} 获取器初始化成功")
                return True
            else:
                self.logger.error(f"{self.exchange} {self.market_type} 获取器初始化失败: 连接测试未通过")
                return False
                
        except Exception as e:
            self.logger.error(f"{self.exchange} {self.market_type} 获取器初始化失败: {e}")
            return False
    
    def close(self):
        """关闭获取器，释放资源"""
        self.is_running = False
        self.is_initialized = False
        
        # 关闭CCXT连接
        if self.ccxt_exchange:
            try:
                if hasattr(self.ccxt_exchange, 'close'):
                    self.ccxt_exchange.close()
            except Exception as e:
                self.logger.error(f"关闭CCXT连接失败: {e}")
        
        self.logger.info(f"{self.exchange} {self.market_type} 获取器已关闭")
    
    def __enter__(self):
        """上下文管理器进入"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
    
    def __repr__(self):
        """字符串表示"""
        return f"BaseFetcher(exchange='{self.exchange}', market_type='{self.market_type}')"


# ==================== 异步数据获取器 ====================

class AsyncFetcher(ABC):
    """
    异步数据获取器基类
    定义所有异步数据获取器的公共接口和功能
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 market_type: str = "spot",
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化异步数据获取器
        
        参数:
            exchange: 交易所名称
            market_type: 市场类型
            config: 配置字典
            cache_manager: 缓存管理器
            loop: 事件循环
        """
        self.exchange = exchange.lower()
        self.market_type = market_type.lower()
        self.config = self._load_config(config)
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 缓存管理器
        self.cache_manager = cache_manager or CacheManager()
        
        # 事件循环
        self.loop = loop or asyncio.get_event_loop()
        
        # 请求统计
        self.request_stats = RequestStats()
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        
        # 请求锁和频率限制
        self.request_lock = asyncio.Lock()
        self.last_request_time = 0
        self.min_request_interval = self.config.rate_limit / 1000  # 转换为秒
        
        # 状态标记
        self.is_initialized = False
        self.is_running = False
        
        # CCXT异步实例
        self.ccxt_exchange = None
        
        self.logger.info(f"初始化异步数据获取器: 交易所={self.exchange}, 市场类型={self.market_type}")
    
    def _load_config(self, config: Optional[Dict]) -> FetcherConfig:
        """加载配置"""
        if config is None:
            config = {}
        
        return FetcherConfig(
            name=config.get('name', f"{self.exchange}_{self.market_type}_async_fetcher"),
            exchange=config.get('exchange', self.exchange),
            market_type=config.get('market_type', self.market_type),
            rate_limit=config.get('rate_limit', 1000),
            timeout=config.get('timeout', 30000),
            retry_count=config.get('retry_count', 3),
            max_retry_delay=config.get('max_retry_delay', 30),
            proxy_url=config.get('proxy_url'),
            enable_cache=config.get('enable_cache', True),
            cache_ttl=config.get('cache_ttl'),
            verbose=config.get('verbose', False)
        )
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志器"""
        logger_name = f"async_fetcher.{self.exchange}.{self.market_type}"
        return get_logger(logger_name)
    
    async def _rate_limit(self):
        """异步频率限制控制"""
        async with self.request_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                if self.config.verbose:
                    self.logger.debug(f"频率限制: 等待 {sleep_time:.3f} 秒")
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _record_request(self, request_type: str, success: bool, start_time: float):
        """记录请求统计"""
        response_time = time.time() - start_time
        self.request_stats.add_request(request_type, success, response_time)
        
        if success:
            self.consecutive_errors = 0
        else:
            self.error_count += 1
            self.last_error_time = datetime.now()
            self.consecutive_errors += 1
    
    async def _retry_request(self, func: Callable, *args, request_type: str = "unknown", **kwargs):
        """
        带重试的异步请求执行
        
        参数:
            func: 要执行的异步函数
            *args: 函数参数
            request_type: 请求类型
            **kwargs: 函数关键字参数
            
        返回:
            函数执行结果
        """
        max_retries = self.config.retry_count
        max_retry_delay = self.config.max_retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                await self._rate_limit()
                start_time = time.time()
                
                result = await func(*args, **kwargs)
                
                self._record_request(request_type, True, start_time)
                return result
                
            except RateLimitError as e:
                # 频率限制错误，增加等待时间
                self.logger.warning(f"频率限制错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, max_retry_delay)  # 指数退避，有上限
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    self._record_request(request_type, False, start_time)
                    raise
                    
            except (ConnectionError, TimeoutError) as e:
                # 连接或超时错误，重试
                self.logger.warning(f"连接错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait_time = min(1 + attempt, max_retry_delay)
                    await asyncio.sleep(wait_time)
                else:
                    self._record_request(request_type, False, start_time)
                    raise
                    
            except AuthenticationError as e:
                # 认证错误，不重试
                self.logger.error(f"认证错误: {e}")
                self._record_request(request_type, False, start_time)
                raise
                
            except Exception as e:
                # 其他错误
                self.logger.error(f"请求错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait_time = min(1 + attempt, max_retry_delay)
                    await asyncio.sleep(wait_time)
                else:
                    self._record_request(request_type, False, start_time)
                    raise
        
        # 所有重试都失败
        raise DataFetcherError(f"请求失败，已重试 {max_retries} 次")
    
    def _get_cache_key(self, 
                      method: str, 
                      symbol: str = None, 
                      timeframe: str = None,
                      **params) -> str:
        """生成缓存键"""
        key_parts = [self.exchange, self.market_type, method]
        
        if symbol:
            key_parts.append(symbol.replace('/', '_'))
        
        if timeframe:
            key_parts.append(timeframe)
        
        # 添加参数哈希
        if params:
            import hashlib
            param_str = str(sorted(params.items()))
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_parts.append(param_hash)
        
        return "_".join(key_parts)
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.config.enable_cache:
            return None
        
        return self.cache_manager.get(key, sub_dir=self.market_type)
    
    def _cache_set(self, key: str, data: Any, ttl: int = None):
        """设置缓存数据"""
        if not self.config.enable_cache or data is None:
            return
        
        # Avoid persisting empty placeholders in cache
        if isinstance(data, pd.DataFrame) and data.empty:
            return

        if isinstance(data, (list, tuple, set, dict)) and len(data) == 0:
            return

        if isinstance(data, np.ndarray) and data.size == 0:
            return

        if ttl is None:
            # 根据数据类型设置默认TTL
            if isinstance(data, list) and len(data) > 0:
                ttl = self.config.cache_ttl.get('hour', 3600)
            else:
                ttl = 300  # 默认5分钟
        
        self.cache_manager.set(key, data, ttl=ttl, sub_dir=self.market_type)
    
    @abstractmethod
    async def _init_exchange_async(self):
        """异步初始化交易所实例（抽象方法）"""
        pass
    
    @abstractmethod
    async def fetch_ohlcv_async(self, 
                              symbol: str, 
                              timeframe: str = "1h",
                              since: Optional[Union[int, datetime, str]] = None,
                              limit: Optional[int] = None,
                              **kwargs) -> List[OHLCVData]:
        """
        异步获取K线数据（抽象方法）
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCVData列表
        """
        pass
    
    @abstractmethod
    async def fetch_orderbook_async(self, 
                                  symbol: str,
                                  limit: Optional[int] = None,
                                  **kwargs) -> Optional[OrderBookData]:
        """
        异步获取订单簿数据（抽象方法）
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            OrderBookData对象
        """
        pass
    
    @abstractmethod
    async def fetch_trades_async(self, 
                               symbol: str,
                               since: Optional[Union[int, datetime, str]] = None,
                               limit: Optional[int] = None,
                               **kwargs) -> List[TradeData]:
        """
        异步获取成交数据（抽象方法）
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            TradeData列表
        """
        pass
    
    async def fetch_ticker_async(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        异步获取行情数据（可选实现）
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            行情数据字典
        """
        if self.ccxt_exchange and hasattr(self.ccxt_exchange, 'fetch_ticker'):
            try:
                return await self._retry_request(
                    self.ccxt_exchange.fetch_ticker,
                    symbol,
                    request_type='ticker'
                )
            except Exception as e:
                self.logger.error(f"获取行情数据失败: {e}")
                return {}
        else:
            self.logger.warning("fetch_ticker_async 方法未实现")
            return {}
    
    async def fetch_market_info_async(self, symbol: str = None) -> Dict[str, Any]:
        """
        异步获取市场信息（可选实现）
        
        参数:
            symbol: 交易对符号（可选）
            
        返回:
            市场信息字典
        """
        if self.ccxt_exchange:
            try:
                if symbol:
                    markets = self.ccxt_exchange.markets
                    return markets.get(symbol, {})
                else:
                    return self.ccxt_exchange.markets
            except Exception as e:
                self.logger.error(f"获取市场信息失败: {e}")
                return {}
        else:
            self.logger.warning("市场信息获取未实现")
            return {}
    
    async def fetch_funding_rate_async(self, symbol: str, **kwargs) -> Optional[FundingRateData]:
        """
        异步获取资金费率数据（永续合约专用）
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            FundingRateData对象
        """
        if self.market_type not in ['swap', 'future']:
            self.logger.warning(f"资金费率仅适用于永续合约，当前市场类型: {self.market_type}")
            return None
        
        if self.ccxt_exchange and hasattr(self.ccxt_exchange, 'fetch_funding_rate'):
            try:
                funding_rate_data = await self._retry_request(
                    self.ccxt_exchange.fetch_funding_rate,
                    symbol,
                    request_type='funding_rate'
                )
                return FundingRateData(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    exchange=self.exchange,
                    market_type=self.market_type,
                    funding_rate=float(funding_rate_data.get('fundingRate', 0)),
                    funding_time=datetime.fromtimestamp(funding_rate_data.get('fundingTime', 0) / 1000)
                )
            except Exception as e:
                self.logger.error(f"获取资金费率失败: {e}")
                return None
        else:
            self.logger.warning("fetch_funding_rate_async 方法未实现")
            return None
    
    async def fetch_ohlcv_bulk_async(self,
                                   symbol: str,
                                   start_date: Union[str, datetime],
                                   end_date: Union[str, datetime],
                                   timeframe: str = "1h",
                                   max_bars_per_request: int = 1000,
                                   **kwargs) -> pd.DataFrame:
        """
        异步批量获取K线数据
        
        参数:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间间隔
            max_bars_per_request: 每次请求最大条数
            **kwargs: 额外参数
            
        返回:
            K线数据DataFrame
        """
        # 转换日期格式（先做交易所特定的窗口裁剪，再生成缓存键）
        if isinstance(start_date, str):
            start_dt = pd.Timestamp(start_date)
        else:
            start_dt = pd.Timestamp(start_date)

        if isinstance(end_date, str):
            end_dt = pd.Timestamp(end_date)
        else:
            end_dt = pd.Timestamp(end_date)

        # 统一时区：将 tz-aware 时间戳转换为 UTC 后再去掉 tz 信息（保持 tz-naive）。
        try:
            if getattr(start_dt, 'tzinfo', None) is not None:
                start_dt = start_dt.tz_convert('UTC').tz_localize(None)
            if getattr(end_dt, 'tzinfo', None) is not None:
                end_dt = end_dt.tz_convert('UTC').tz_localize(None)
        except Exception:
            pass

        # Gate 限制：只能请求“最近 N 根K线”窗口
        exchange_lower = str(self.exchange).lower()
        if exchange_lower == 'gate':
            max_lookback_bars = int(self.config.get('gate_max_ohlcv_lookback_bars', 10000))
            seconds_per_bar = int(calculate_timeframe_seconds(timeframe))
            now_ts = pd.Timestamp.utcnow()
            try:
                if getattr(now_ts, 'tzinfo', None) is not None:
                    now_ts = now_ts.tz_convert('UTC').tz_localize(None)
            except Exception:
                pass
            earliest_allowed = now_ts - pd.Timedelta(seconds=max_lookback_bars * seconds_per_bar)

            if end_dt < earliest_allowed:
                msg = (
                    f"Gate OHLCV 历史深度限制：仅支持最近 {max_lookback_bars} 根 {timeframe}。"
                    f" 当前请求 end={end_dt} 早于允许最早时间 {earliest_allowed}，无法从 Gate 获取该历史区间。"
                )
                self.logger.error(msg)
                if kwargs.get('raise_error', False):
                    raise ValueError(msg)
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe'])

            if start_dt < earliest_allowed:
                self.logger.warning(
                    f"Gate OHLCV 历史深度限制：start={start_dt} 早于允许最早 {earliest_allowed}，将自动裁剪到允许窗口。"
                )
                start_dt = earliest_allowed

            max_bars_per_request = min(max_bars_per_request, 500)

        # 生成缓存键（使用裁剪后的 start/end）
        cache_key = self._get_cache_key(
            'ohlcv_bulk_async',
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_dt,
            end_date=end_dt
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取批量K线数据: {symbol} {timeframe}")
            return cached_data
        
        # 分割日期范围
        chunks = split_date_range(
            start_date=start_dt,
            end_date=end_dt,
            timeframe=timeframe,
            max_bars=max_bars_per_request
        )
        
        all_data = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            self.logger.info(
                f"异步批量获取第 {i + 1}/{len(chunks)} 段数据: "
                f"{chunk_start.strftime('%Y-%m-%d %H:%M')} 到 "
                f"{chunk_end.strftime('%Y-%m-%d %H:%M')}"
            )
            
            try:
                # 获取该时间段的数据
                data_list = await self.fetch_ohlcv_async(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=chunk_start,
                    limit=max_bars_per_request,
                    **kwargs
                )
                
                # 转换为DataFrame并添加到列表
                if data_list:
                    data_dicts = []
                    for data in data_list:
                        if isinstance(data, OHLCVData):
                            data_dicts.append({
                                'timestamp': data.timestamp,
                                'open': data.open,
                                'high': data.high,
                                'low': data.low,
                                'close': data.close,
                                'volume': data.volume,
                                'symbol': symbol,
                                'timeframe': timeframe
                            })
                    if data_dicts:
                        chunk_df = pd.DataFrame(data_dicts)
                        chunk_df.set_index('timestamp', inplace=True)
                        all_data.append(chunk_df)
                
                # 避免频率限制
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"获取第 {i + 1} 段数据失败: {e}")
                if kwargs.get('raise_error', False):
                    raise
                continue
        
        # 合并所有数据
        if all_data:
            df = pd.concat(all_data, axis=0)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            
            # 缓存结果 (1天缓存)
            self._cache_set(cache_key, df, ttl=86400)
            
            self.logger.info(f"异步批量获取完成: {symbol} {timeframe}, 共 {len(df)} 条数据")
            return df
        else:
            self.logger.warning(f"未获取到数据: {symbol} {timeframe}")
            return pd.DataFrame()
    
    async def test_connection_async(self) -> bool:
        """
        异步测试交易所连接
        
        返回:
            连接是否成功
        """
        try:
            # 尝试获取一个常见交易对的信息
            test_symbol = "BTC/USDT"
            if self.validate_symbol(test_symbol):
                ticker = await self.fetch_ticker_async(test_symbol)
                return ticker is not None and 'last' in ticker
            return False
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False
    
    async def initialize_async(self):
        """异步初始化获取器"""
        try:
            self.logger.info(f"异步初始化 {self.exchange} {self.market_type} 获取器...")
            
            # 初始化交易所实例
            await self._init_exchange_async()
            
            # 测试连接
            if await self.test_connection_async():
                self.is_initialized = True
                self.is_running = True
                self.logger.info(f"{self.exchange} {self.market_type} 异步获取器初始化成功")
                return True
            else:
                self.logger.error(f"{self.exchange} {self.market_type} 异步获取器初始化失败: 连接测试未通过")
                return False
                
        except Exception as e:
            self.logger.error(f"{self.exchange} {self.market_type} 异步获取器初始化失败: {e}")
            return False
    
    async def close_async(self):
        """异步关闭获取器，释放资源"""
        self.is_running = False
        self.is_initialized = False
        
        # 关闭CCXT连接
        if self.ccxt_exchange:
            try:
                if hasattr(self.ccxt_exchange, 'close'):
                    await self.ccxt_exchange.close()
            except Exception as e:
                self.logger.error(f"关闭CCXT连接失败: {e}")
        
        self.logger.info(f"{self.exchange} {self.market_type} 异步获取器已关闭")
    
    async def __aenter__(self):
        """异步上下文管理器进入"""
        await self.initialize_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close_async()
    
    def __repr__(self):
        """字符串表示"""
        return f"AsyncFetcher(exchange='{self.exchange}', market_type='{self.market_type}')"


# ==================== 多市场数据获取器 ====================

class MultiMarketFetcher:
    """
    多市场数据获取器
    管理多个市场类型的数据获取
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化多市场数据获取器
        
        参数:
            exchange: 交易所名称
            config: 配置字典
            cache_manager: 缓存管理器
        """
        self.exchange = exchange
        self.config = config or {}
        self.cache_manager = cache_manager
        
        # 初始化日志
        self.logger = get_logger(f"multi_market_fetcher.{exchange}")
        
        # 市场获取器字典
        self.fetchers = {
            'spot': None,
            'margin': None,
            'swap': None,
            'future': None,
            'option': None,
            'onchain': None,
            'social': None
        }
        
        # 初始化状态
        self.is_initialized = False
        
        self.logger.info(f"初始化多市场数据获取器: {exchange}")
    
    def get_fetcher(self, market_type: str) -> Optional[BaseFetcher]:
        """
        获取指定市场类型的获取器
        
        参数:
            market_type: 市场类型
            
        返回:
            数据获取器
        """
        market_type = market_type.lower()
        
        if market_type not in self.fetchers:
            self.logger.error(f"不支持的市场类型: {market_type}")
            return None
        
        return self.fetchers[market_type]
    
    def init_fetcher(self, market_type: str, **kwargs):
        """
        初始化指定市场类型的获取器
        
        参数:
            market_type: 市场类型
            **kwargs: 获取器参数
        """
        market_type = market_type.lower()
        
        try:
            if market_type == 'spot':
                from .spot_fetcher import CCXTSpotFetcher
                self.fetchers[market_type] = CCXTSpotFetcher(
                    exchange=self.exchange,
                    config={**self.config, **kwargs},
                    cache_manager=self.cache_manager
                )
                
            elif market_type == 'margin':
                from .margin_fetcher import CCXTMarginFetcher
                self.fetchers[market_type] = CCXTMarginFetcher(
                    exchange=self.exchange,
                    config={**self.config, **kwargs},
                    cache_manager=self.cache_manager
                )
                
            elif market_type == 'swap':
                from .swap_fetcher import CCXTSwapFetcher
                self.fetchers[market_type] = CCXTSwapFetcher(
                    exchange=self.exchange,
                    config={**self.config, **kwargs},
                    cache_manager=self.cache_manager
                )
                
            elif market_type == 'future':
                from .future_fetcher import CCXTFutureFetcher
                self.fetchers[market_type] = CCXTFutureFetcher(
                    exchange=self.exchange,
                    config={**self.config, **kwargs},
                    cache_manager=self.cache_manager
                )
                
            elif market_type == 'option':
                from .option_fetcher import CCXTOptionFetcher
                self.fetchers[market_type] = CCXTOptionFetcher(
                    exchange=self.exchange,
                    config={**self.config, **kwargs},
                    cache_manager=self.cache_manager
                )
                
            elif market_type == 'onchain':
                from .onchain_fetcher import OnchainFetcher
                self.fetchers[market_type] = OnchainFetcher(
                    exchange=self.exchange,
                    config={**self.config, **kwargs},
                    cache_manager=self.cache_manager
                )
                
            elif market_type == 'social':
                from .social_fetcher import SocialFetcher
                self.fetchers[market_type] = SocialFetcher(
                    exchange=self.exchange,
                    config={**self.config, **kwargs},
                    cache_manager=self.cache_manager
                )
            
            self.logger.info(f"初始化 {market_type} 市场获取器成功")
            
        except ImportError as e:
            self.logger.error(f"导入 {market_type} 市场获取器失败: {e}")
        except Exception as e:
            self.logger.error(f"初始化 {market_type} 市场获取器失败: {e}")
    
    def init_all_fetchers(self, market_types: List[str] = None):
        """
        初始化所有市场类型的获取器
        
        参数:
            market_types: 要初始化的市场类型列表，为None则初始化所有
        """
        self.logger.info(f"初始化{self.exchange}的市场获取器...")
        
        if market_types is None:
            market_types = list(self.fetchers.keys())
        
        for market_type in market_types:
            self.init_fetcher(market_type)
        
        self.is_initialized = True
        self.logger.info(f"市场获取器初始化完成: {len([f for f in self.fetchers.values() if f])}/{len(market_types)}个")
    
    def fetch_multi_market_data(self, 
                               symbol: str,
                               market_types: List[str] = None,
                               data_type: str = 'ohlcv',
                               **kwargs) -> Dict[str, Any]:
        """
        获取多个市场的数据
        
        参数:
            symbol: 交易对符号
            market_types: 市场类型列表
            data_type: 数据类型 (ohlcv, orderbook, trades, ticker, funding_rate, open_interest)
            **kwargs: 额外参数
            
        返回:
            多市场数据字典
        """
        if market_types is None:
            market_types = ['spot', 'swap']
        
        results = {}
        
        for market_type in market_types:
            fetcher = self.get_fetcher(market_type)
            
            if not fetcher:
                self.logger.warning(f"{market_type} 市场获取器未初始化")
                results[market_type] = None
                continue
            
            try:
                if data_type == 'ohlcv':
                    data = fetcher.fetch_ohlcv(symbol, **kwargs)
                elif data_type == 'orderbook':
                    data = fetcher.fetch_orderbook(symbol, **kwargs)
                elif data_type == 'trades':
                    data = fetcher.fetch_trades(symbol, **kwargs)
                elif data_type == 'ticker':
                    data = fetcher.fetch_ticker(symbol, **kwargs)
                elif data_type == 'funding_rate':
                    data = fetcher.fetch_funding_rate(symbol, **kwargs)
                elif data_type == 'open_interest':
                    data = fetcher.fetch_open_interest(symbol, **kwargs)
                else:
                    self.logger.error(f"不支持的数据类型: {data_type}")
                    results[market_type] = None
                    continue
                
                results[market_type] = data
                
            except Exception as e:
                self.logger.error(f"获取 {market_type} 市场数据失败: {e}")
                results[market_type] = None
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取多市场获取器状态
        
        返回:
            状态信息字典
        """
        status = {
            'exchange': self.exchange,
            'is_initialized': self.is_initialized,
            'fetchers': {}
        }
        
        for market_type, fetcher in self.fetchers.items():
            if fetcher:
                status['fetchers'][market_type] = {
                    'initialized': fetcher.is_initialized,
                    'running': fetcher.is_running,
                    'request_stats': fetcher.request_stats.to_dict()
                }
            else:
                status['fetchers'][market_type] = None
        
        return status
    
    def close_all(self):
        """关闭所有获取器"""
        self.logger.info("关闭所有市场获取器...")
        
        for market_type, fetcher in self.fetchers.items():
            if fetcher:
                try:
                    fetcher.close()
                    self.logger.info(f"关闭 {market_type} 市场获取器成功")
                except Exception as e:
                    self.logger.error(f"关闭 {market_type} 市场获取器失败: {e}")
        
        self.is_initialized = False
        self.logger.info("所有市场获取器已关闭")
    
    def __enter__(self):
        """上下文管理器进入"""
        self.init_all_fetchers()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close_all()
    
    def __repr__(self):
        """字符串表示"""
        return f"MultiMarketFetcher(exchange='{self.exchange}')"


# ==================== 工具函数 ====================

def create_fetcher(exchange: str = "binance", 
                  market_type: str = "spot",
                  api_key: Optional[str] = None,
                  api_secret: Optional[str] = None,
                  **kwargs) -> BaseFetcher:
    """
    创建数据获取器工厂函数
    
    参数:
        exchange: 交易所名称
        market_type: 市场类型
        api_key: API密钥
        api_secret: API密钥
        **kwargs: 额外参数
        
    返回:
        数据获取器实例
    """
    config = kwargs.pop('config', {})
    config.update({
        'api_key': api_key,
        'api_secret': api_secret
    })
    
    market_type = market_type.lower()
    
    try:
        # 尝试相对导入（包导入方式）
        if market_type == 'spot':
            from .spot_fetcher import CCXTSpotFetcher
            return CCXTSpotFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'margin':
            from .margin_fetcher import CCXTMarginFetcher
            return CCXTMarginFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'swap':
            from .swap_fetcher import CCXTSwapFetcher
            return CCXTSwapFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'future':
            from .future_fetcher import CCXTFutureFetcher
            return CCXTFutureFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'option':
            from .option_fetcher import CCXTOptionFetcher
            return CCXTOptionFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'onchain':
            from .onchain_fetcher import OnchainFetcher
            return OnchainFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'social':
            from .social_fetcher import SocialFetcher
            return SocialFetcher(exchange=exchange, config=config, **kwargs)
        
        else:
            raise ValueError(f"不支持的市场类型: {market_type}")
    
    except (ImportError, ModuleNotFoundError):
        # 备用：绝对导入方式（脚本直接运行时）
        import sys
        import os
        fetchers_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, fetchers_dir)
        
        if market_type == 'spot':
            from spot_fetcher import CCXTSpotFetcher
            return CCXTSpotFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'margin':
            from margin_fetcher import CCXTMarginFetcher
            return CCXTMarginFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'swap':
            from swap_fetcher import CCXTSwapFetcher
            return CCXTSwapFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'future':
            from future_fetcher import CCXTFutureFetcher
            return CCXTFutureFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'option':
            from option_fetcher import CCXTOptionFetcher
            return CCXTOptionFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'onchain':
            from onchain_fetcher import OnchainFetcher
            return OnchainFetcher(exchange=exchange, config=config, **kwargs)
        
        elif market_type == 'social':
            from social_fetcher import SocialFetcher
            return SocialFetcher(exchange=exchange, config=config, **kwargs)
        
        else:
            raise ValueError(f"不支持的市场类型: {market_type}")


def create_async_fetcher(exchange: str = "binance", 
                        market_type: str = "spot",
                        api_key: Optional[str] = None,
                        api_secret: Optional[str] = None,
                        **kwargs) -> AsyncFetcher:
    """
    创建异步数据获取器工厂函数
    
    参数:
        exchange: 交易所名称
        market_type: 市场类型
        api_key: API密钥
        api_secret: API密钥
        **kwargs: 额外参数
        
    返回:
        异步数据获取器实例
    """
    config = kwargs.pop('config', {})
    config.update({
        'api_key': api_key,
        'api_secret': api_secret
    })
    
    market_type = market_type.lower()
    
    # 异步获取器需要特定的实现
    # 这里先返回基础异步类，实际使用中需要具体实现
    # 注意：异步获取器的具体实现需要根据对应的市场类型来确定
    print(f"注意: 异步获取器 {market_type} 需要具体实现，当前返回基础AsyncFetcher类")
    return AsyncFetcher(exchange=exchange, market_type=market_type, config=config, **kwargs)


# ==================== 测试函数 ====================

def test_base_fetcher():
    """测试基础获取器"""
    print("=" * 60)
    print("基础获取器模块测试")
    print("=" * 60)
    
    # 测试配置类
    print("\n1. 测试FetcherConfig:")
    config = FetcherConfig(
        name="test_fetcher",
        exchange="binance",
        market_type="spot",
        rate_limit=1000,
        timeout=30000
    )
    print(f"✅ 配置创建成功: {config}")
    
    # 测试请求统计
    print("\n2. 测试RequestStats:")
    stats = RequestStats()
    stats.add_request('ohlcv', True, 0.5)
    stats.add_request('orderbook', False, 0.3)
    stats.add_request('ticker', True, 0.7)
    
    print(f"✅ 总请求数: {stats.total_requests}")
    print(f"✅ 成功请求数: {stats.successful_requests}")
    print(f"✅ 失败请求数: {stats.failed_requests}")
    print(f"✅ 成功率: {stats.success_rate:.2f}%")
    print(f"✅ 平均响应时间: {stats.average_response_time:.3f}秒")
    print(f"✅ 请求类型分布: {stats.requests_by_type}")
    
    # 测试异常类
    print("\n3. 测试异常类:")
    try:
        raise ConnectionError("测试连接异常", Exception("原始异常"))
    except ConnectionError as e:
        print(f"✅ 异常捕获成功: {e.message}, 原始异常: {e.original_error}")
    
    # 测试基础获取器类（模拟）
    print("\n4. 测试BaseFetcher基类（模拟）:")
    
    class TestFetcher(BaseFetcher):
        def _init_exchange(self):
            self.logger.info("模拟初始化交易所")
            self.ccxt_exchange = type('MockExchange', (), {})()
        
        def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=None, **kwargs):
            return []
        
        def fetch_orderbook(self, symbol, limit=None, **kwargs):
            return None
        
        def fetch_trades(self, symbol, since=None, limit=None, **kwargs):
            return []
    
    try:
        fetcher = TestFetcher(exchange="binance", market_type="spot")
        print(f"✅ 基础获取器创建成功: {fetcher}")
        
        # 测试验证交易对
        test_symbol = "BTC/USDT"
        is_valid = fetcher.validate_symbol(test_symbol)
        print(f"✅ 交易对验证: {test_symbol} -> {is_valid}")
        
        # 测试格式化交易对
        formatted = fetcher.format_symbol("btc-usdt")
        print(f"✅ 交易对格式化: btc-usdt -> {formatted}")
        
        # 测试获取状态
        status = fetcher.get_status()
        print(f"✅ 获取器状态: {status['exchange']}, 市场类型: {status['market_type']}")
        
        # 测试重置统计
        fetcher.reset_stats()
        print("✅ 统计信息已重置")
        
        # 测试关闭获取器
        fetcher.close()
        print("✅ 获取器关闭成功")
        
    except Exception as e:
        print(f"❌ 基础获取器测试失败: {e}")
    
    # 测试多市场获取器
    print("\n5. 测试MultiMarketFetcher:")
    try:
        multi_fetcher = MultiMarketFetcher(exchange="binance")
        print(f"✅ 多市场获取器创建成功: {multi_fetcher.exchange}")
        
        # 测试获取状态
        status = multi_fetcher.get_status()
        print(f"✅ 多市场获取器状态: 已初始化={status['is_initialized']}, 获取器数量={len(status['fetchers'])}")
        
    except Exception as e:
        print(f"❌ 多市场获取器测试失败: {e}")
    
    # 测试工厂函数
    print("\n6. 测试工厂函数:")
    try:
        # 测试创建现货获取器（模拟）
        spot_fetcher = create_fetcher(exchange="binance", market_type="spot")
        print(f"✅ 现货获取器创建成功: {spot_fetcher}")
        
        print("⚠️  具体功能测试需要网络连接和API密钥")
        
    except Exception as e:
        print(f"❌ 工厂函数测试失败: {e}")
    
    print("\n✅ 基础获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_base_fetcher()