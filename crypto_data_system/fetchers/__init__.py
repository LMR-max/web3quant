"""
基础数据获取器模块
提供所有数据获取器的基类和通用功能
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 导入工具模块
try:
    from utils.logger import get_logger, log_execution_time, log_errors
    from utils.cache import CacheManager, cached
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from data_models import BaseData, OHLCVData, OrderBookData, TradeData
except ImportError:
    # 如果直接运行，使用简单导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import get_logger, log_execution_time, log_errors
    from utils.cache import CacheManager, cached
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from data_models import BaseData, OHLCVData, OrderBookData, TradeData


# ==================== 基础类 ====================

class BaseFetcher(ABC):
    """
    基础数据获取器抽象基类
    
    所有数据获取器都应该继承这个类，并实现必要的方法
    """
    
    def __init__(self, 
                 name: str = "base_fetcher",
                 exchange: str = "unknown",
                 market_type: str = "unknown",
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化基础获取器
        
        参数:
            name: 获取器名称
            exchange: 交易所名称
            market_type: 市场类型
            config: 配置字典
            cache_manager: 缓存管理器
        """
        self.name = name
        self.exchange = exchange
        self.market_type = market_type
        self.config = config or {}
        
        # 设置日志器
        self.logger = get_logger(f"fetcher.{name}")
        
        # 设置缓存管理器
        self.cache_manager = cache_manager or CacheManager()
        
        # 初始化状态
        self.is_initialized = False
        self.last_fetch_time = None
        self.fetch_count = 0
        self.error_count = 0
        
        # 请求统计
        self.request_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'cached': 0,
            'total_time': 0.0
        }
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化获取器"""
        try:
            self.logger.info(f"初始化获取器: {self.name} (exchange={self.exchange}, market={self.market_type})")
            
            # 设置默认配置
            default_config = {
                'rate_limit': 10,  # 每秒请求数限制
                'retry_count': 3,  # 重试次数
                'timeout': 30,  # 超时时间（秒）
                'cache_ttl': 300,  # 缓存时间（秒）
                'max_retry_delay': 60,  # 最大重试延迟（秒）
                'use_proxy': True,  # 是否使用代理
                'proxy_url': None,  # 代理URL
                'user_agent': f"CryptoDataFetcher/{self.name}",
                'enable_cache': True,  # 是否启用缓存
                'verify_ssl': True,  # 是否验证SSL证书
            }
            
            # 合并配置
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
            
            self.is_initialized = True
            self.logger.info(f"获取器初始化完成: {self.name}")
            
        except Exception as e:
            self.logger.error(f"获取器初始化失败: {e}")
            raise
    
    @abstractmethod
    def fetch_ohlcv(self, 
                    symbol: str, 
                    timeframe: str = "1h",
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[OHLCVData]:
        """
        获取K线数据（抽象方法，子类必须实现）
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间（时间戳、datetime对象或字符串）
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        pass
    
    @abstractmethod
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> OrderBookData:
        """
        获取订单簿数据（抽象方法，子类必须实现）
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        pass
    
    @abstractmethod
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取成交数据（抽象方法，子类必须实现）
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        pass
    
    def fetch_ohlcv_bulk(self,
                        symbol: str,
                        start_date: Union[datetime, str],
                        end_date: Union[datetime, str],
                        timeframe: str = "1h",
                        max_bars_per_request: int = 1000,
                        **kwargs) -> List[OHLCVData]:
        """
        批量获取K线数据（自动处理API限制）
        
        参数:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间间隔，默认为"1h"
            max_bars_per_request: 每次请求最大K线数量，默认为1000
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        self.logger.info(f"批量获取K线数据: {symbol}, {timeframe}, {start_date} 到 {end_date}")
        
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).to_pydatetime()
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date).to_pydatetime()
        
        # 分割日期范围
        date_chunks = split_date_range(
            start_date, end_date, timeframe, max_bars_per_request
        )
        
        all_ohlcv = []
        
        for i, (chunk_start, chunk_end) in enumerate(date_chunks):
            self.logger.info(f"获取分块 {i+1}/{len(date_chunks)}: {chunk_start} 到 {chunk_end}")
            
            try:
                # 获取分块数据
                chunk_data = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=chunk_start,
                    limit=max_bars_per_request,
                    **kwargs
                )
                
                if chunk_data:
                    all_ohlcv.extend(chunk_data)
                    self.logger.info(f"分块 {i+1} 获取成功: {len(chunk_data)} 条数据")
                else:
                    self.logger.warning(f"分块 {i+1} 未获取到数据")
                
                # 限制请求频率
                if i < len(date_chunks) - 1:
                    time.sleep(self.config.get('rate_limit_delay', 1.0))
                    
            except Exception as e:
                self.logger.error(f"分块 {i+1} 获取失败: {e}")
                self.error_count += 1
                
                # 如果连续失败过多，停止获取
                if self.error_count > self.config.get('max_errors', 10):
                    self.logger.error("错误过多，停止获取")
                    break
        
        self.logger.info(f"批量获取完成: 总共获取 {len(all_ohlcv)} 条数据")
        return all_ohlcv
    
    def fetch_ticker(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        获取行情数据
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            行情数据字典
        """
        try:
            # 获取最新的K线数据
            ohlcv_list = self.fetch_ohlcv(symbol=symbol, timeframe="1m", limit=1, **kwargs)
            
            if ohlcv_list:
                latest = ohlcv_list[0]
                return {
                    'symbol': symbol,
                    'timestamp': latest.timestamp,
                    'open': latest.open,
                    'high': latest.high,
                    'low': latest.low,
                    'close': latest.close,
                    'volume': latest.volume,
                    'exchange': self.exchange,
                    'market_type': self.market_type
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"获取行情数据失败: {e}")
            return {}
    
    def fetch_multiple_symbols(self,
                              symbols: List[str],
                              fetch_func: Callable,
                              **kwargs) -> Dict[str, Any]:
        """
        获取多个交易对的数据
        
        参数:
            symbols: 交易对列表
            fetch_func: 获取函数
            **kwargs: 传递给获取函数的参数
            
        返回:
            按交易对组织的数据字典
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            self.logger.info(f"获取交易对 {i+1}/{len(symbols)}: {symbol}")
            
            try:
                result = fetch_func(symbol=symbol, **kwargs)
                results[symbol] = result
                
                # 限制请求频率
                if i < len(symbols) - 1:
                    time.sleep(self.config.get('rate_limit_delay', 0.5))
                    
            except Exception as e:
                self.logger.error(f"获取交易对 {symbol} 失败: {e}")
                results[symbol] = None
        
        return results
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证交易对符号是否有效
        
        参数:
            symbol: 交易对符号
            
        返回:
            是否有效
        """
        # 基本验证
        if not symbol or not isinstance(symbol, str):
            return False
        
        # 检查是否包含斜杠（标准格式）
        if '/' not in symbol:
            self.logger.warning(f"交易对符号可能不规范: {symbol}")
            # 仍然返回True，因为某些交易所可能使用不同格式
        
        return True
    
    def format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号
        
        参数:
            symbol: 原始交易对符号
            
        返回:
            格式化后的交易对符号
        """
        # 去除空格
        symbol = symbol.strip()
        
        # 统一分隔符为斜杠
        symbol = symbol.replace('-', '/')
        
        return symbol.upper()
    
    def parse_timestamp(self, timestamp: Union[int, str, datetime]) -> datetime:
        """
        解析时间戳
        
        参数:
            timestamp: 时间戳（毫秒、秒、字符串或datetime对象）
            
        返回:
            datetime对象
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        elif isinstance(timestamp, (int, float)):
            # 判断是毫秒还是秒
            if timestamp > 1e12:  # 毫秒
                return datetime.fromtimestamp(timestamp / 1000)
            else:  # 秒
                return datetime.fromtimestamp(timestamp)
        
        elif isinstance(timestamp, str):
            try:
                # 尝试解析ISO格式
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # 尝试解析时间戳字符串
                    ts = float(timestamp)
                    return self.parse_timestamp(ts)
                except ValueError:
                    raise ValueError(f"无法解析时间戳: {timestamp}")
        
        else:
            raise TypeError(f"不支持的时间戳类型: {type(timestamp)}")
    
    @log_execution_time
    def safe_fetch(self, fetch_func: Callable, *args, **kwargs) -> Any:
        """
        安全执行获取函数（带重试和错误处理）
        
        参数:
            fetch_func: 获取函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        返回:
            函数执行结果
        """
        max_retries = self.config.get('retry_count', 3)
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = fetch_func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # 更新统计
                self.request_stats['total'] += 1
                self.request_stats['success'] += 1
                self.request_stats['total_time'] += elapsed
                self.last_fetch_time = datetime.now()
                self.fetch_count += 1
                
                self.logger.debug(f"请求成功 (尝试 {attempt+1}): {elapsed:.2f}秒")
                return result
                
            except Exception as e:
                self.logger.warning(f"请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
                self.error_count += 1
                self.request_stats['failed'] += 1
                
                # 最后一次尝试，抛出异常
                if attempt == max_retries - 1:
                    self.logger.error(f"请求最终失败: {e}")
                    raise
                
                # 等待后重试
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.config.get('max_retry_delay', 60))
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取获取器状态
        
        返回:
            状态字典
        """
        return {
            'name': self.name,
            'exchange': self.exchange,
            'market_type': self.market_type,
            'is_initialized': self.is_initialized,
            'last_fetch_time': self.last_fetch_time,
            'fetch_count': self.fetch_count,
            'error_count': self.error_count,
            'request_stats': self.request_stats,
            'cache_enabled': self.config.get('enable_cache', True),
            'rate_limit': self.config.get('rate_limit', 10)
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.fetch_count = 0
        self.error_count = 0
        self.request_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'cached': 0,
            'total_time': 0.0
        }
        self.logger.info("统计信息已重置")
    
    def close(self):
        """关闭获取器，释放资源"""
        self.logger.info(f"关闭获取器: {self.name}")
        # 子类可以在这里实现资源清理
        self.is_initialized = False
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
    
    def __str__(self):
        return f"BaseFetcher(name={self.name}, exchange={self.exchange}, market={self.market_type})"
    
    def __repr__(self):
        return self.__str__()


# ==================== 同步数据获取器 ====================

class DataFetcher(BaseFetcher):
    """
    同步数据获取器
    
    提供同步方式获取数据的实现
    """
    
    def __init__(self, name: str = "data_fetcher", **kwargs):
        super().__init__(name=name, **kwargs)
    
    @log_errors(reraise=True)
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", 
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None, **kwargs) -> List[OHLCVData]:
        """
        同步获取K线数据（需要子类实现具体逻辑）
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        # 子类应该实现这个方法
        raise NotImplementedError("子类必须实现 fetch_ohlcv 方法")
    
    @log_errors(reraise=True)
    def fetch_orderbook(self, symbol: str, limit: Optional[int] = None, 
                       **kwargs) -> OrderBookData:
        """
        同步获取订单簿数据（需要子类实现具体逻辑）
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        # 子类应该实现这个方法
        raise NotImplementedError("子类必须实现 fetch_orderbook 方法")
    
    @log_errors(reraise=True)
    def fetch_trades(self, symbol: str, 
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None, **kwargs) -> List[TradeData]:
        """
        同步获取成交数据（需要子类实现具体逻辑）
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        # 子类应该实现这个方法
        raise NotImplementedError("子类必须实现 fetch_trades 方法")


# ==================== 异步数据获取器 ====================

class AsyncFetcher(BaseFetcher):
    """
    异步数据获取器
    
    提供异步方式获取数据的实现
    """
    
    def __init__(self, name: str = "async_fetcher", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loop = None
    
    async def fetch_ohlcv_async(self, symbol: str, timeframe: str = "1h", 
                               since: Optional[Union[int, datetime, str]] = None,
                               limit: Optional[int] = None, **kwargs) -> List[OHLCVData]:
        """
        异步获取K线数据（需要子类实现具体逻辑）
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        # 子类应该实现这个方法
        raise NotImplementedError("子类必须实现 fetch_ohlcv_async 方法")
    
    async def fetch_orderbook_async(self, symbol: str, limit: Optional[int] = None, 
                                   **kwargs) -> OrderBookData:
        """
        异步获取订单簿数据（需要子类实现具体逻辑）
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        # 子类应该实现这个方法
        raise NotImplementedError("子类必须实现 fetch_orderbook_async 方法")
    
    async def fetch_trades_async(self, symbol: str, 
                                since: Optional[Union[int, datetime, str]] = None,
                                limit: Optional[int] = None, **kwargs) -> List[TradeData]:
        """
        异步获取成交数据（需要子类实现具体逻辑）
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        # 子类应该实现这个方法
        raise NotImplementedError("子类必须实现 fetch_trades_async 方法")
    
    # 同步方法包装异步方法
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", 
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None, **kwargs) -> List[OHLCVData]:
        """
        同步获取K线数据（包装异步方法）
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        return self.loop.run_until_complete(
            self.fetch_ohlcv_async(symbol, timeframe, since, limit, **kwargs)
        )
    
    def fetch_orderbook(self, symbol: str, limit: Optional[int] = None, 
                       **kwargs) -> OrderBookData:
        """
        同步获取订单簿数据（包装异步方法）
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        return self.loop.run_until_complete(
            self.fetch_orderbook_async(symbol, limit, **kwargs)
        )
    
    def fetch_trades(self, symbol: str, 
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None, **kwargs) -> List[TradeData]:
        """
        同步获取成交数据（包装异步方法）
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        return self.loop.run_until_complete(
            self.fetch_trades_async(symbol, since, limit, **kwargs)
        )
    
    async def fetch_multiple_async(self, symbols: List[str], 
                                  fetch_func: Callable, **kwargs) -> Dict[str, Any]:
        """
        异步获取多个交易对的数据
        
        参数:
            symbols: 交易对列表
            fetch_func: 异步获取函数
            **kwargs: 传递给获取函数的参数
            
        返回:
            按交易对组织的数据字典
        """
        import asyncio
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(fetch_func(symbol=symbol, **kwargs))
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                result = await task
                results[symbol] = result
            except Exception as e:
                self.logger.error(f"异步获取交易对 {symbol} 失败: {e}")
                results[symbol] = None
        
        return results
    
    def close(self):
        """关闭获取器，释放资源"""
        super().close()
        if self.loop is not None and not self.loop.is_closed():
            self.loop.close()
            self.loop = None


# ==================== 测试函数 ====================

class TestFetcher(DataFetcher):
    """测试获取器"""
    
    def __init__(self, **kwargs):
        super().__init__(name="test_fetcher", **kwargs)
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", 
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None, **kwargs) -> List[OHLCVData]:
        """模拟获取K线数据"""
        from datetime import datetime, timedelta
        
        # 生成测试数据
        data = []
        now = datetime.now()
        
        for i in range(limit or 10):
            timestamp = now - timedelta(hours=i)
            ohlcv = OHLCVData(
                timestamp=timestamp,
                symbol=symbol,
                exchange=self.exchange,
                market_type=self.market_type,
                timeframe=timeframe,
                open=50000 + i * 10,
                high=51000 + i * 10,
                low=49000 + i * 10,
                close=50500 + i * 10,
                volume=1000 + i * 100
            )
            data.append(ohlcv)
        
        return data
    
    def fetch_orderbook(self, symbol: str, limit: Optional[int] = None, 
                       **kwargs) -> OrderBookData:
        """模拟获取订单簿数据"""
        from datetime import datetime
        
        bids = [(50500 - i * 100, 1.0 + i * 0.1) for i in range(limit or 5)]
        asks = [(50500 + i * 100, 1.0 + i * 0.1) for i in range(limit or 5)]
        
        return OrderBookData(
            timestamp=datetime.now(),
            symbol=symbol,
            exchange=self.exchange,
            market_type=self.market_type,
            bids=bids,
            asks=asks,
            spread=200.0
        )
    
    def fetch_trades(self, symbol: str, 
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None, **kwargs) -> List[TradeData]:
        """模拟获取成交数据"""
        from datetime import datetime, timedelta
        
        trades = []
        now = datetime.now()
        
        for i in range(limit or 5):
            trade = TradeData(
                timestamp=now - timedelta(minutes=i),
                symbol=symbol,
                exchange=self.exchange,
                market_type=self.market_type,
                trade_id=f"test_{i}",
                price=50500 + i * 10,
                amount=1.0 + i * 0.1,
                side="buy" if i % 2 == 0 else "sell"
            )
            trades.append(trade)
        
        return trades


def test_base_fetcher():
    """测试基础获取器"""
    print("=" * 60)
    print("基础获取器模块测试")
    print("=" * 60)
    
    # 测试基础功能
    print("\n1. 测试基础获取器:")
    try:
        fetcher = TestFetcher(exchange="test", market_type="spot")
        print(f"✅ 获取器创建成功: {fetcher}")
        
        # 测试状态获取
        status = fetcher.get_status()
        print(f"✅ 状态获取成功: {status['name']}")
        
        # 测试K线数据获取
        ohlcv_data = fetcher.fetch_ohlcv("BTC/USDT", limit=5)
        print(f"✅ K线数据获取成功: {len(ohlcv_data)} 条")
        
        # 测试订单簿数据获取
        orderbook = fetcher.fetch_orderbook("BTC/USDT")
        print(f"✅ 订单簿数据获取成功: {len(orderbook.bids)} 个买盘")
        
        # 测试成交数据获取
        trades = fetcher.fetch_trades("BTC/USDT", limit=3)
        print(f"✅ 成交数据获取成功: {len(trades)} 条")
        
        # 测试批量获取
        bulk_data = fetcher.fetch_ohlcv_bulk(
            symbol="BTC/USDT",
            start_date="2024-01-01",
            end_date="2024-01-02",
            timeframe="1h",
            max_bars_per_request=10
        )
        print(f"✅ 批量获取成功: {len(bulk_data)} 条")
        
        # 测试多个交易对获取
        symbols = ["BTC/USDT", "ETH/USDT"]
        results = fetcher.fetch_multiple_symbols(
            symbols=symbols,
            fetch_func=fetcher.fetch_ticker
        )
        print(f"✅ 多交易对获取成功: {len(results)} 个交易对")
        
        # 关闭获取器
        fetcher.close()
        print("✅ 获取器关闭成功")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试异步获取器
    print("\n2. 测试异步获取器:")
    try:
        class TestAsyncFetcher(AsyncFetcher):
            async def fetch_ohlcv_async(self, symbol: str, timeframe: str = "1h", 
                                       since=None, limit=None, **kwargs):
                # 模拟异步获取
                import asyncio
                await asyncio.sleep(0.1)
                
                data = []
                now = datetime.now()
                for i in range(limit or 3):
                    timestamp = now - timedelta(hours=i)
                    data.append(OHLCVData(
                        timestamp=timestamp,
                        symbol=symbol,
                        exchange=self.exchange,
                        market_type=self.market_type,
                        timeframe=timeframe,
                        open=50000,
                        high=51000,
                        low=49000,
                        close=50500,
                        volume=1000
                    ))
                return data
            
            async def fetch_orderbook_async(self, symbol: str, limit=None, **kwargs):
                import asyncio
                await asyncio.sleep(0.1)
                return OrderBookData(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    exchange=self.exchange,
                    market_type=self.market_type,
                    bids=[(50000, 1.0)],
                    asks=[(51000, 1.0)],
                    spread=1000.0
                )
            
            async def fetch_trades_async(self, symbol: str, since=None, limit=None, **kwargs):
                import asyncio
                await asyncio.sleep(0.1)
                return []
        
        async_fetcher = TestAsyncFetcher(exchange="test", market_type="spot")
        
        # 同步调用异步方法
        ohlcv = async_fetcher.fetch_ohlcv("BTC/USDT", limit=2)
        print(f"✅ 异步获取器同步调用成功: {len(ohlcv)} 条K线")
        
        async_fetcher.close()
        
    except Exception as e:
        print(f"❌ 异步获取器测试失败: {e}")
    
    print("\n✅ 基础获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    test_base_fetcher()