"""
现货数据获取器模块
提供从交易所获取现货市场数据的功能
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
    from .base_fetcher import BaseFetcher, AsyncFetcher, MultiMarketFetcher, create_fetcher
    from ..data_models import OHLCVData, OrderBookData, TradeData, TimeSeriesData
    from ..utils.logger import get_logger
    from ..utils.cache import CacheManager
    from ..utils.date_utils import split_date_range, calculate_timeframe_seconds
    from ..config import get_exchange_config, ExchangeSymbolFormats, PresetConfigs
except ImportError:
    # 如果直接运行，使用简单导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fetchers.base_fetcher import BaseFetcher, AsyncFetcher, MultiMarketFetcher, create_fetcher
    from data_models import OHLCVData, OrderBookData, TradeData, TimeSeriesData
    from utils.logger import get_logger
    from utils.cache import CacheManager
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from config import get_exchange_config, ExchangeSymbolFormats, PresetConfigs


# ==================== CCXT现货获取器 ====================

class CCXTSpotFetcher(BaseFetcher):
    """
    CCXT现货数据获取器
    
    使用CCXT库获取现货数据
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化CCXT现货获取器
        
        参数:
            exchange: 交易所名称
            api_key: API密钥
            api_secret: API密钥
            config: 配置字典
            cache_manager: 缓存管理器
        """
        # 更新配置
        if config is None:
            config = {}
        
        config.update({
            'api_key': api_key,
            'api_secret': api_secret
        })
        
        super().__init__(
            exchange=exchange,
            market_type="spot",
            config=config,
            cache_manager=cache_manager
        )
        
        # 加载交易所配置
        self.exchange_config = get_exchange_config(exchange)
        
        # 预设配置
        self.market_config = PresetConfigs.get_spot_config()
        
        # 市场信息
        self.spot_markets = {}
        self.markets_loaded = False
        
        # 初始化交易所连接
        self._init_exchange()
    
    def format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号为交易所标准格式
        覆盖 BaseFetcher 的实现，使用 ExchangeSymbolFormats 处理每个交易所的特定格式
        
        参数:
            symbol: 原始交易对符号（如 BTC/USDT、BTC-USDT）
            
        返回:
            格式化后的交易对符号
        """
        # 使用 ExchangeSymbolFormats 处理交易所特定格式
        try:
            formatted = ExchangeSymbolFormats.format_symbol(
                symbol=symbol, 
                exchange_id=self.exchange,
                market_type='spot'
            )
            return formatted
        except Exception:
            # 如果 ExchangeSymbolFormats 处理失败，回退到基础处理
            formatted = symbol.strip().upper()
            if '-' in formatted and '/' not in formatted:
                formatted = formatted.replace('-', '/')
            return formatted
    
    def _init_exchange(self):
        """初始化交易所实例"""
        try:
            self.logger.info(f"初始化交易所连接: {self.exchange}")
            
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
            
            # 合并预设配置
            ccxt_config.update(self.market_config.get('ccxt_options', {}))
            
            # 如果有API密钥，添加
            if self.config.api_key and self.config.api_secret:
                ccxt_config['apiKey'] = self.config.api_key
                ccxt_config['secret'] = self.config.api_secret
            
            # 如果有代理配置，添加
            if self.config.proxy_url:
                if self.config.verbose:
                    self.logger.info(f"使用代理: {self.config.proxy_url}")
                ccxt_config['proxies'] = {
                    'http': self.config.proxy_url,
                    'https': self.config.proxy_url,
                }
            
            # 创建交易所实例
            self.ccxt_exchange = exchange_class(ccxt_config)
            
            # 加载市场信息
            self._load_markets()

            if self.markets_loaded:
                self.logger.info(
                    f"交易所连接初始化成功: {self.exchange}（spot markets={len(self.spot_markets)}）"
                )
            else:
                self.logger.warning(
                    f"交易所连接初始化完成: {self.exchange}，但市场信息未加载；"
                    f"get_available_symbols 可能为空（可配置 proxy_url 或稍后重试）"
                )
            
        except Exception as e:
            self.logger.error(f"交易所连接初始化失败: {e}")
            self.ccxt_exchange = None
    
    def _load_markets(self):
        """加载市场信息"""
        if not self.ccxt_exchange:
            return
        
        try:
            self.logger.info(f"加载 {self.exchange} 市场信息...")
            
            # 加载市场
            self.ccxt_exchange.load_markets()
            
            # 筛选现货交易对
            self.spot_markets = {}
            for symbol, market in self.ccxt_exchange.markets.items():
                if market.get('spot', False) and market.get('active', False):
                    self.spot_markets[symbol] = market
            
            self.markets_loaded = True
            self.logger.info(f"加载 {len(self.spot_markets)} 个现货交易对")
            
            # 缓存市场信息
            if self.cache_manager:
                self.cache_manager.set(
                    key=f"{self.exchange}_spot_markets",
                    data=self.spot_markets,
                    ttl=3600,  # 缓存1小时
                    sub_dir='spot'
                )
            
        except Exception as e:
            self.logger.error(
                f"加载市场信息失败: {type(e).__name__}: {e!r}"
            )

            # OKX 在部分网络环境下可能无法直连；给出可操作提示
            if str(self.exchange).lower() == 'okx' and not getattr(self.config, 'proxy_url', None):
                self.logger.warning(
                    "OKX 市场信息加载失败，可能是网络/地区限制导致；"
                    "如需访问 OKX，可在配置中设置 proxy_url（http/https 代理）。"
                )

            # 尝试从缓存恢复（允许在无网络情况下仍能返回 symbols）
            recovered = False
            if self.cache_manager:
                try:
                    cached = self.cache_manager.get(
                        key=f"{self.exchange}_spot_markets",
                        sub_dir='spot'
                    )
                    if isinstance(cached, dict) and cached:
                        self.spot_markets = cached
                        self.markets_loaded = True
                        recovered = True
                        self.logger.info(
                            f"已从缓存恢复 {len(self.spot_markets)} 个现货交易对: {self.exchange}"
                        )
                except Exception as cache_error:
                    self.logger.warning(
                        f"从缓存恢复市场信息失败: {type(cache_error).__name__}: {cache_error!r}"
                    )

            if not recovered:
                self.spot_markets = {}
                self.markets_loaded = False
    
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = "1h",
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None,
                   **kwargs) -> List[OHLCVData]:
        """
        获取现货K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 验证交易对
        if not self.validate_symbol(formatted_symbol):
            self.logger.warning(f"交易对可能无效: {formatted_symbol}")
        
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

        # Gate 限制：仅允许请求“最近 N 根K线”的窗口。
        # 如果 since 太久远，Gate 会直接报错（Candlestick too long ago...）。
        if str(self.exchange).lower() == 'gate' and since_timestamp is not None:
            try:
                max_lookback_bars = int(self.config.get('gate_max_ohlcv_lookback_bars', 10000))
            except Exception:
                max_lookback_bars = 10000
            try:
                seconds_per_bar = int(calculate_timeframe_seconds(timeframe))
            except Exception:
                seconds_per_bar = 60
            now_ms = int(time.time() * 1000)
            earliest_ms = now_ms - (max_lookback_bars * seconds_per_bar * 1000)
            if int(since_timestamp) < int(earliest_ms):
                self.logger.warning(
                    f"Gate OHLCV 历史深度限制：since={since_timestamp} 早于允许最早 {earliest_ms}，将自动裁剪到允许窗口。"
                )
                since_timestamp = int(earliest_ms)
        
        # 设置默认限制（避免超出交易所/CCXT 单次上限）
        # Gate API 限制：最多 10000 个数据点的历史深度
        # 对于 Gate，我们需要更严格的限制以避免请求过于久远的数据
        if limit is None:
            if str(self.exchange).lower() == 'gate':
                # Gate 较为保守：单次最多 1000 条，但要考虑历史深度限制
                limit = 500
            else:
                limit = 1000

        # 是否拉取扩展字段（quoteVolume/numberOfTrades/takerBuy...）以及是否计算指标
        use_extended = kwargs.pop('extended', True)
        with_indicators = kwargs.pop('with_indicators', True)
        
        self.logger.info(
            f"获取现货K线数据: {formatted_symbol}, "
            f"时间间隔: {timeframe}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'ohlcv',
            symbol=formatted_symbol,
            timeframe=timeframe,
            since=since_timestamp,
            limit=limit
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取K线数据: {formatted_symbol} {timeframe}")
            return cached_data
        
        try:
            params = dict(kwargs.get('params', {}) or {})

            # Binance spot：优先用 /api/v3/klines 获取扩展字段
            ohlcv_list = None
            if use_extended and str(self.exchange).lower() == 'binance':
                try:
                    market = self.ccxt_exchange.market(formatted_symbol)
                    market_id = market.get('id') or formatted_symbol.replace('/', '')
                    req = {
                        'symbol': market_id,
                        'interval': timeframe,
                        'limit': limit,
                        **params,
                    }
                    if since_timestamp is not None:
                        req['startTime'] = int(since_timestamp)

                    if hasattr(self.ccxt_exchange, 'publicGetKlines'):
                        ohlcv_list = self._retry_request(
                            self.ccxt_exchange.publicGetKlines,
                            params=req,
                            request_type='ohlcv'
                        )
                except Exception as e:
                    self.logger.warning(f"Binance 扩展K线获取失败，回退到 fetch_ohlcv: {e}")
                    ohlcv_list = None

            # fallback：标准 CCXT
            if ohlcv_list is None:
                ohlcv_list = self._retry_request(
                    self.ccxt_exchange.fetch_ohlcv,
                    symbol=formatted_symbol,
                    timeframe=timeframe,
                    since=since_timestamp,
                    limit=limit,
                    params=params,
                    request_type='ohlcv'
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
                        market_type="spot"
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换OHLCV数据失败: {e}")
                    continue
            
            # 计算指标（就地写回到 OHLCVData 对象）
            if with_indicators and data_models:
                try:
                    self._apply_indicators_to_ohlcv(data_models)
                except Exception as ind_err:
                    self.logger.warning(f"计算指标失败（将返回不含指标的数据）: {ind_err}")

            self.logger.info(f"获取到 {len(data_models)} 条K线数据")
            
            # 缓存结果 (1小时缓存)
            self._cache_set(cache_key, data_models, ttl=3600)
            
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            # 如果失败，返回空列表而不是抛出异常
            if kwargs.get('raise_error', False):
                raise
            return []

    def fetch_order_book(self, symbol: str, limit: Optional[int] = 20, **kwargs) -> Optional[OrderBookData]:
        """
        获取现货盘口数据
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            OrderBookData对象
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 验证交易对
        if not self.validate_symbol(formatted_symbol):
            self.logger.warning(f"交易对可能无效: {formatted_symbol}")

        self.logger.info(f"获取现货盘口数据: {formatted_symbol}, 限制: {limit}")
        
        # 盘口数据实时性强，通常不缓存，或者缓存时间极短
        # 但为了接口一致性，依然可以使用极短ttl
        cache_key = self._get_cache_key('order_book', symbol=formatted_symbol, limit=limit)
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
             # 判断是否足够"新鲜"，例如1秒内
             if isinstance(cached_data, OrderBookData):
                 if (pd.Timestamp.now() - cached_data.timestamp).total_seconds() < 1:
                    return cached_data

        try:
            params = dict(kwargs.get('params', {}) or {})
            order_book = self._retry_request(
                self.ccxt_exchange.fetch_order_book,
                symbol=formatted_symbol,
                limit=limit,
                params=params,
                request_type='order_book'
            )
            
            data_model = OrderBookData.from_ccxt(
                orderbook=order_book,
                symbol=formatted_symbol,
                exchange=self.exchange
            )
            
            # 缓存极短时间，主要防止并发重复请求
            self._cache_set(cache_key, data_model, ttl=1)
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取盘口数据失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None

    def fetch_orderbook(self, *args, **kwargs):
        """兼容性别名"""
        return self.fetch_order_book(*args, **kwargs)

    def _apply_indicators_to_ohlcv(self, ohlcv_list: List[OHLCVData]) -> None:
        """对 OHLCV 列表计算常用技术指标并写回对象。

        计算项：RSI(14)、SMA(20/50/200)、EMA(12/26)、MACD(12,26,9)、Bollinger(20,2)
        """
        if not ohlcv_list:
            return

        # 确保按时间升序
        ohlcv_list.sort(key=lambda x: x.timestamp)

        closes = pd.Series([float(x.close) for x in ohlcv_list], dtype='float64')

        # RSI(14) - 标准 Wilder's RSI (EMA)
        delta = closes.diff()
        # wildcard smoothing which is equivalent to exponential smoothing with alpha = 1/N
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        sma_20 = closes.rolling(window=20, min_periods=20).mean()
        sma_50 = closes.rolling(window=50, min_periods=50).mean()
        sma_200 = closes.rolling(window=200, min_periods=200).mean()

        ema_12 = closes.ewm(span=12, adjust=False).mean()
        ema_26 = closes.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        bb_middle = sma_20
        bb_std = closes.rolling(window=20, min_periods=20).std(ddof=0)
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std

        for i, item in enumerate(ohlcv_list):
            # vwap（若已从交易所提供 quote_volume）
            try:
                if getattr(item, 'vwap', None) is None:
                    qv = float(getattr(item, 'quote_volume', 0.0) or 0.0)
                    vol = float(getattr(item, 'volume', 0.0) or 0.0)
                    if vol > 0 and qv > 0:
                        item.vwap = qv / vol
            except Exception:
                pass

            def _set_if_finite(attr: str, value):
                try:
                    if value is None:
                        return
                    if pd.isna(value):
                        return
                    setattr(item, attr, float(value))
                except Exception:
                    return

            _set_if_finite('rsi', rsi.iloc[i])
            _set_if_finite('sma_20', sma_20.iloc[i])
            _set_if_finite('sma_50', sma_50.iloc[i])
            _set_if_finite('sma_200', sma_200.iloc[i])
            _set_if_finite('ema_12', ema_12.iloc[i])
            _set_if_finite('ema_26', ema_26.iloc[i])
            _set_if_finite('macd', macd.iloc[i])
            _set_if_finite('macd_signal', macd_signal.iloc[i])
            _set_if_finite('macd_histogram', macd_hist.iloc[i])
            _set_if_finite('bollinger_upper', bb_upper.iloc[i])
            _set_if_finite('bollinger_middle', bb_middle.iloc[i])
            _set_if_finite('bollinger_lower', bb_lower.iloc[i])
    
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> Optional[OrderBookData]:
        """
        获取现货订单簿数据
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 设置默认限制
        if limit is None:
            limit = 20
        
        self.logger.info(f"获取现货订单簿: {formatted_symbol}, 深度: {limit}")
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'orderbook',
            symbol=formatted_symbol,
            limit=limit
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取订单簿: {formatted_symbol}")
            return cached_data
        
        try:
            # 使用重试机制获取订单簿
            orderbook = self._retry_request(
                self.ccxt_exchange.fetch_order_book,
                symbol=formatted_symbol,
                limit=limit,
                request_type='orderbook'
            )
            
            # 转换为数据模型
            data_model = OrderBookData.from_ccxt(
                orderbook=orderbook,
                symbol=formatted_symbol,
                exchange=self.exchange
            )
            
            self.logger.info(
                f"订单簿获取成功: 买盘 {len(data_model.bids)} 个, "
                f"卖盘 {len(data_model.asks)} 个, "
                f"价差: {data_model.spread:.2f}"
            )
            
            # 缓存结果（订单簿变化快，缓存时间短）
            self._cache_set(cache_key, data_model, ttl=30)
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取现货成交数据
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        if not self.ccxt_exchange:
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

        # Gate 限制：仅允许请求“最近 N 根K线”的窗口。
        if str(self.exchange).lower() == 'gate' and since_timestamp is not None:
            try:
                max_lookback_bars = int(self.config.get('gate_max_ohlcv_lookback_bars', 10000))
            except Exception:
                max_lookback_bars = 10000
            try:
                tf = kwargs.get('timeframe', '1m')
                seconds_per_bar = int(calculate_timeframe_seconds(tf))
            except Exception:
                seconds_per_bar = 60
            now_ms = int(time.time() * 1000)
            earliest_ms = now_ms - (max_lookback_bars * seconds_per_bar * 1000)
            if int(since_timestamp) < int(earliest_ms):
                self.logger.warning(
                    f"Gate OHLCV 历史深度限制：since={since_timestamp} 早于允许最早 {earliest_ms}，将自动裁剪到允许窗口。"
                )
                since_timestamp = int(earliest_ms)
        
        # 设置默认限制
        if limit is None:
            limit = 100
        
        self.logger.info(
            f"获取现货成交数据: {formatted_symbol}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'trades',
            symbol=formatted_symbol,
            since=since_timestamp,
            limit=limit
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取成交数据: {formatted_symbol}")
            return cached_data
        
        try:
            # 使用重试机制获取成交数据
            trades = self._retry_request(
                self.ccxt_exchange.fetch_trades,
                symbol=formatted_symbol,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {}),
                request_type='trades'
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
                    self.logger.warning(f"转换成交数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条成交数据")
            
            # 缓存结果
            self._cache_set(cache_key, data_models, ttl=60)
            
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取成交数据失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []
    
    def fetch_ticker(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        获取现货行情数据
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            行情数据字典
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"获取现货行情: {formatted_symbol}")
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'ticker',
            symbol=formatted_symbol
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取行情: {formatted_symbol}")
            return cached_data
        
        try:
            # 使用重试机制获取ticker
            ticker = self._retry_request(
                self.ccxt_exchange.fetch_ticker,
                formatted_symbol,
                request_type='ticker'
            )
            
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
                f"行情获取成功: 最新价 {result.get('last')}, "
                f"24h成交量 {result.get('baseVolume')}"
            )
            
            # 缓存结果
            self._cache_set(cache_key, result, ttl=30)
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取行情数据失败: {e}")
            # 如果失败，尝试从父类获取
            return super().fetch_ticker(symbol, **kwargs)

    def fetch_tickers(self, symbols: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """获取多个/全部交易对的行情。

        - 优先使用 CCXT 的 `fetch_tickers`（如果交易所支持）。
        - 不支持时退化为循环调用 `fetch_ticker`。

        参数:
            symbols: 交易对列表；为空时尝试获取全部（若交易所支持）
            **kwargs: 额外参数（会透传给 ccxt.fetch_tickers 的 params）

        返回:
            dict: symbol -> ticker（原始 CCXT 返回结构）
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        formatted_symbols: Optional[List[str]] = None
        if symbols:
            formatted_symbols = [self.format_symbol(s) for s in symbols]

        cache_key = self._get_cache_key(
            'tickers',
            symbols=','.join(formatted_symbols) if formatted_symbols else 'ALL'
        )

        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if hasattr(self.ccxt_exchange, 'fetch_tickers') and getattr(self.ccxt_exchange, 'has', {}).get('fetchTickers', False):
                try:
                    tickers = self._retry_request(
                        self.ccxt_exchange.fetch_tickers,
                        formatted_symbols,
                        params=kwargs.get('params', {}),
                        request_type='ticker'
                    )
                except TypeError:
                    tickers = self._retry_request(
                        self.ccxt_exchange.fetch_tickers,
                        params=kwargs.get('params', {}),
                        request_type='ticker'
                    )

                self._cache_set(cache_key, tickers or {}, ttl=30)
                return tickers or {}

            # fallback: loop
            tickers: Dict[str, Any] = {}
            if not formatted_symbols:
                formatted_symbols = self.get_available_symbols()
            for s in formatted_symbols:
                try:
                    tickers[s] = self.fetch_ticker(s)
                except Exception:
                    tickers[s] = {}

            self._cache_set(cache_key, tickers, ttl=30)
            return tickers

        except Exception as e:
            self.logger.error(f"获取多交易对行情失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return {}

    def fetch_currencies(self, **kwargs) -> Dict[str, Any]:
        """获取币种/资产信息（若交易所支持）。"""
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        cache_key = self._get_cache_key('currencies')
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if hasattr(self.ccxt_exchange, 'fetch_currencies') and getattr(self.ccxt_exchange, 'has', {}).get('fetchCurrencies', False):
                currencies = self._retry_request(
                    self.ccxt_exchange.fetch_currencies,
                    params=kwargs.get('params', {}),
                    request_type='market'
                )
                self._cache_set(cache_key, currencies or {}, ttl=3600)
                return currencies or {}
            return {}
        except Exception as e:
            self.logger.error(f"获取币种信息失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return {}

    def fetch_status(self, **kwargs) -> Dict[str, Any]:
        """获取交易所状态（若交易所支持）。"""
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        cache_key = self._get_cache_key('status')
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if hasattr(self.ccxt_exchange, 'fetch_status') and getattr(self.ccxt_exchange, 'has', {}).get('fetchStatus', False):
                status = self._retry_request(
                    self.ccxt_exchange.fetch_status,
                    params=kwargs.get('params', {}),
                    request_type='market'
                )
                self._cache_set(cache_key, status or {}, ttl=60)
                return status or {}
            return {}
        except Exception as e:
            self.logger.error(f"获取交易所状态失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return {}

    def fetch_time(self, **kwargs) -> Optional[int]:
        """获取交易所服务器时间（毫秒时间戳，若支持）。"""
        if not self.ccxt_exchange:
            raise RuntimeError(f"交易所 {self.exchange} 未初始化")

        cache_key = self._get_cache_key('time')
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if hasattr(self.ccxt_exchange, 'fetch_time') and getattr(self.ccxt_exchange, 'has', {}).get('fetchTime', True):
                ts = self._retry_request(
                    self.ccxt_exchange.fetch_time,
                    request_type='test'
                )
                # 服务器时间变化快，短缓存
                self._cache_set(cache_key, ts, ttl=10)
                return ts
            return None
        except Exception as e:
            self.logger.error(f"获取服务器时间失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    def fetch_market_info(self, symbol: str = None) -> Dict[str, Any]:
        """
        获取市场信息
        
        参数:
            symbol: 交易对符号（可选）
            
        返回:
            市场信息字典
        """
        if not self.ccxt_exchange:
            return {}
        
        if not self.markets_loaded:
            self._load_markets()
        
        if symbol:
            # 格式化交易对
            formatted_symbol = self.format_symbol(symbol)
            
            if formatted_symbol not in self.spot_markets:
                return {}
            
            market = self.spot_markets[formatted_symbol]
            
            info = {
                'symbol': formatted_symbol,
                'base': market.get('base'),
                'quote': market.get('quote'),
                'baseId': market.get('baseId'),
                'quoteId': market.get('quoteId'),
                'active': market.get('active'),
                'precision': {
                    'amount': market.get('precision', {}).get('amount'),
                    'price': market.get('precision', {}).get('price'),
                    'base': market.get('precision', {}).get('base'),
                    'quote': market.get('precision', {}).get('quote')
                },
                'limits': {
                    'amount': market.get('limits', {}).get('amount'),
                    'price': market.get('limits', {}).get('price'),
                    'cost': market.get('limits', {}).get('cost')
                },
                'taker': market.get('taker'),
                'maker': market.get('maker'),
                'percentage': market.get('percentage'),
                'tierBased': market.get('tierBased'),
                'feeSide': market.get('feeSide')
            }
            
            return info
        else:
            # 返回所有市场
            return self.spot_markets

    def fetch_market_snapshot(
        self,
        symbol: str,
        timeframe: str = "1h",
        ohlcv_limit: int = 200,
        trades_limit: int = 200,
        orderbook_limit: int = 50,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取现货市场“尽可能全”的快照。

        这是一个聚合入口，方便一次性拿到某个现货交易对的常用公开数据。

        include 可选项：
            - ohlcv
            - ticker
            - stats_24h
            - orderbook
            - trades
            - market_info
            - time
            - status

        返回：dict，键为上述数据类型。
        """
        if include is None:
            include = ['ticker', 'stats_24h', 'orderbook', 'trades', 'market_info']

        formatted_symbol = self.format_symbol(symbol)
        snapshot: Dict[str, Any] = {
            'exchange': self.exchange,
            'market_type': 'spot',
            'symbol': formatted_symbol,
        }

        if 'time' in include:
            snapshot['time'] = self.fetch_time()
        if 'status' in include:
            snapshot['status'] = self.fetch_status()
        if 'market_info' in include:
            snapshot['market_info'] = self.fetch_market_info(formatted_symbol)
        if 'ticker' in include:
            snapshot['ticker'] = self.fetch_ticker(formatted_symbol)
        if 'stats_24h' in include:
            snapshot['stats_24h'] = self.fetch_24h_stats(formatted_symbol)
        if 'orderbook' in include:
            snapshot['orderbook'] = self.fetch_orderbook(formatted_symbol, limit=orderbook_limit)
        if 'trades' in include:
            snapshot['trades'] = self.fetch_trades(formatted_symbol, limit=trades_limit)
        if 'ohlcv' in include:
            snapshot['ohlcv'] = self.fetch_ohlcv(formatted_symbol, timeframe=timeframe, limit=ohlcv_limit)

        return snapshot
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的现货交易对
        
        返回:
            交易对列表
        """
        if not self.markets_loaded:
            self._load_markets()
        
        return list(self.spot_markets.keys())
    
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
        if self.spot_markets:
            # 先尝试原始符号
            if symbol in self.spot_markets:
                return True
            
            # 尝试格式标准化后再检查
            formatted_symbol = ExchangeSymbolFormats.format_symbol(
                symbol, self.exchange, 'spot'
            )
            if formatted_symbol in self.spot_markets:
                return True
            
            # 尝试反向转换（如果用户输入了 / 但市场中是 -）
            if '/' in formatted_symbol and '-' not in symbol:
                alt_symbol = formatted_symbol.replace('/', '-')
                if alt_symbol in self.spot_markets:
                    return True
            
            # 如果都找不到，返回 False（表示无效）
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
        formatted = ExchangeSymbolFormats.format_symbol(symbol, self.exchange, 'spot')
        self.logger.debug(f"格式化交易对: {symbol} -> {formatted}")
        return formatted
    
    def fetch_24h_stats(self, symbol: str) -> Dict[str, Any]:
        """
        获取24小时统计信息
        
        参数:
            symbol: 交易对符号
            
        返回:
            24小时统计信息
        """
        ticker = self.fetch_ticker(symbol)
        
        if not ticker:
            return {}
        
        # 提取24小时统计信息
        stats = {
            'symbol': symbol,
            'timestamp': ticker.get('timestamp'),
            'price_change': ticker.get('change'),
            'price_change_percent': ticker.get('percentage'),
            'high_24h': ticker.get('high'),
            'low_24h': ticker.get('low'),
            'volume_24h': ticker.get('baseVolume'),
            'quote_volume_24h': ticker.get('quoteVolume'),
            'last_price': ticker.get('last'),
            'bid_price': ticker.get('bid'),
            'ask_price': ticker.get('ask'),
            'spread': None
        }
        
        # 计算价差
        if stats['bid_price'] and stats['ask_price']:
            stats['spread'] = stats['ask_price'] - stats['bid_price']
            if stats['bid_price'] > 0:
                stats['spread_percent'] = (stats['spread'] / stats['bid_price']) * 100
        
        return stats
    
    def fetch_historical_klines(self, 
                               symbol: str,
                               start_date: Union[str, datetime],
                               end_date: Union[str, datetime],
                               timeframe: str = "1h",
                               **kwargs) -> pd.DataFrame:
        """
        获取历史K线数据（返回DataFrame格式）
        
        参数:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间间隔
            **kwargs: 额外参数
            
        返回:
            K线数据DataFrame
        """
        # 获取OHLCV数据
        ohlcv_list = self.fetch_ohlcv_bulk(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        # 转换为DataFrame
        from data_models import DataConverter
        df = DataConverter.ohlcv_list_to_dataframe(ohlcv_list)
        
        return df
    
    def get_market_depth(self, 
                        symbol: str, 
                        depth_levels: int = 10) -> Dict[str, Any]:
        """
        获取市场深度分析
        
        参数:
            symbol: 交易对符号
            depth_levels: 深度等级
            
        返回:
            市场深度分析
        """
        orderbook = self.fetch_orderbook(symbol, limit=depth_levels * 2)
        
        if not orderbook:
            return {}
        
        # 分析订单簿
        bids = orderbook.bids
        asks = orderbook.asks
        
        # 计算累计量
        bid_cumulative = 0
        ask_cumulative = 0
        
        bid_amounts = [amount for _, amount in bids]
        ask_amounts = [amount for _, amount in asks]
        
        if bid_amounts:
            bid_cumulative = sum(bid_amounts)
        
        if ask_amounts:
            ask_cumulative = sum(ask_amounts)
        
        # 计算买卖压力
        total_volume = bid_cumulative + ask_cumulative
        bid_pressure = bid_cumulative / total_volume if total_volume > 0 else 0
        ask_pressure = ask_cumulative / total_volume if total_volume > 0 else 0
        
        # 计算加权平均价格
        if bids:
            bid_weighted_price = sum(price * amount for price, amount in bids) / bid_cumulative
        else:
            bid_weighted_price = 0
        
        if asks:
            ask_weighted_price = sum(price * amount for price, amount in asks) / ask_cumulative
        else:
            ask_weighted_price = 0
        
        analysis = {
            'symbol': symbol,
            'timestamp': orderbook.timestamp,
            'best_bid': bids[0][0] if bids else 0,
            'best_ask': asks[0][0] if asks else 0,
            'spread': orderbook.spread,
            'bid_depth': len(bids),
            'ask_depth': len(asks),
            'bid_cumulative': bid_cumulative,
            'ask_cumulative': ask_cumulative,
            'bid_pressure': bid_pressure,
            'ask_pressure': ask_pressure,
            'bid_weighted_price': bid_weighted_price,
            'ask_weighted_price': ask_weighted_price,
            'imbalance': (bid_cumulative - ask_cumulative) / (bid_cumulative + ask_cumulative) 
                        if (bid_cumulative + ask_cumulative) > 0 else 0,
            'orderbook_summary': orderbook.get_summary(levels=depth_levels)
        }
        
        return analysis
    
    def test_connection(self) -> bool:
        """
        测试交易所连接
        
        返回:
            连接是否成功
        """
        try:
            if not self.ccxt_exchange:
                return False
            
            # 尝试获取服务器时间
            timestamp = self._retry_request(
                self.ccxt_exchange.fetch_time,
                request_type='test'
            )
            
            self.logger.info(f"交易所连接测试成功，服务器时间: {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"交易所连接测试失败: {e}")
            return False
    
    def close(self):
        """关闭获取器，释放资源"""
        if self.ccxt_exchange:
            # CCXT通常不需要显式关闭，但这里可以清理资源
            try:
                if hasattr(self.ccxt_exchange, 'close'):
                    self.ccxt_exchange.close()
            except Exception as e:
                self.logger.error(f"关闭CCXT连接失败: {e}")
            finally:
                self.ccxt_exchange = None
        
        super().close()


# ==================== 异步现货获取器 ====================

class AsyncSpotFetcher(AsyncFetcher):
    """
    异步现货数据获取器
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化异步现货数据获取器
        
        参数:
            exchange: 交易所名称
            api_key: API密钥
            api_secret: API密钥
            config: 配置字典
            cache_manager: 缓存管理器
            loop: 事件循环
        """
        # 更新配置
        if config is None:
            config = {}
        
        config.update({
            'api_key': api_key,
            'api_secret': api_secret
        })
        
        super().__init__(
            exchange=exchange,
            market_type="spot",
            config=config,
            cache_manager=cache_manager,
            loop=loop
        )
        
        # 加载交易所配置
        self.exchange_config = get_exchange_config(exchange)
        
        # 预设配置
        self.market_config = PresetConfigs.get_spot_config()
        
        # 市场信息
        self.spot_markets = {}
        self.markets_loaded = False
    
    async def _init_exchange_async(self):
        """异步初始化交易所实例"""
        try:
            self.logger.info(f"初始化异步交易所连接: {self.exchange}")
            
            # 尝试导入CCXT Pro（异步版本）
            try:
                import ccxt.pro as ccxt
                self.has_ccxt_pro = True
            except ImportError:
                self.has_ccxt_pro = False
                self.logger.warning("CCXT Pro未安装，将使用同步版本")
                return
            
            # 获取交易所配置
            exchange_config = get_exchange_config(self.exchange)
            ccxt_config = exchange_config.get_ccxt_config()
            
            # 合并预设配置
            ccxt_config.update(self.market_config.get('ccxt_options', {}))
            
            # 如果有API密钥，添加
            if self.config.api_key and self.config.api_secret:
                ccxt_config['apiKey'] = self.config.api_key
                ccxt_config['secret'] = self.config.api_secret
            
            # 创建异步交易所实例
            exchange_class = getattr(ccxt, self.exchange, None)
            if not exchange_class:
                raise ValueError(f"不支持的交易所: {self.exchange}")
            
            self.ccxt_exchange = exchange_class(ccxt_config)
            
            # 加载市场信息
            await self._load_markets_async()
            
            self.logger.info(f"异步交易所连接初始化成功: {self.exchange}")
            
        except Exception as e:
            self.logger.error(f"异步交易所连接初始化失败: {e}")
            self.ccxt_exchange = None
    
    async def _load_markets_async(self):
        """异步加载市场信息"""
        if not self.ccxt_exchange:
            return
        
        try:
            self.logger.info(f"加载 {self.exchange} 市场信息...")
            
            # 加载市场
            await self.ccxt_exchange.load_markets()
            
            # 筛选现货交易对
            self.spot_markets = {}
            for symbol, market in self.ccxt_exchange.markets.items():
                if market.get('spot', False) and market.get('active', False):
                    self.spot_markets[symbol] = market
            
            self.markets_loaded = True
            self.logger.info(f"加载 {len(self.spot_markets)} 个现货交易对")
            
            # 缓存市场信息
            if self.cache_manager:
                self.cache_manager.set(
                    key=f"{self.exchange}_spot_markets_async",
                    data=self.spot_markets,
                    ttl=3600,  # 缓存1小时
                    sub_dir='spot'
                )
            
        except Exception as e:
            self.logger.error(f"加载市场信息失败: {e}")
            self.spot_markets = {}
    
    async def fetch_ohlcv_async(self, 
                               symbol: str, 
                               timeframe: str = "1h",
                               since: Optional[Union[int, datetime, str]] = None,
                               limit: Optional[int] = None,
                               **kwargs) -> List[OHLCVData]:
        """
        异步获取现货K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"异步交易所 {self.exchange} 未初始化")
        
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
        # Gate API 限制：最多 10000 个数据点的历史深度
        if limit is None:
            if str(self.exchange).lower() == 'gate':
                # Gate 较为保守：单次最多 1000 条，但要考虑历史深度限制
                limit = 500
            else:
                limit = 1000
        
        self.logger.info(
            f"异步获取现货K线数据: {formatted_symbol}, "
            f"时间间隔: {timeframe}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'ohlcv_async',
            symbol=formatted_symbol,
            timeframe=timeframe,
            since=since_timestamp,
            limit=limit
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取异步K线数据: {formatted_symbol} {timeframe}")
            return cached_data
        
        try:
            # 使用异步重试机制获取数据
            ohlcv_list = await self._retry_request(
                self.ccxt_exchange.fetch_ohlcv,
                symbol=formatted_symbol,
                timeframe=timeframe,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {}),
                request_type='ohlcv'
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
                        market_type="spot"
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换OHLCV数据失败: {e}")
                    continue
            
            self.logger.info(f"异步获取到 {len(data_models)} 条K线数据")
            
            # 缓存结果 (1小时缓存)
            self._cache_set(cache_key, data_models, ttl=3600)
            
            return data_models
            
        except Exception as e:
            self.logger.error(f"异步获取K线数据失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []
    
    async def fetch_orderbook_async(self, 
                                   symbol: str,
                                   limit: Optional[int] = None,
                                   **kwargs) -> Optional[OrderBookData]:
        """
        异步获取现货订单簿数据
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"异步交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        # 设置默认限制
        if limit is None:
            limit = 20
        
        self.logger.info(f"异步获取现货订单簿: {formatted_symbol}, 深度: {limit}")
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'orderbook_async',
            symbol=formatted_symbol,
            limit=limit
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取异步订单簿: {formatted_symbol}")
            return cached_data
        
        try:
            # 使用异步重试机制获取订单簿
            orderbook = await self._retry_request(
                self.ccxt_exchange.fetch_order_book,
                symbol=formatted_symbol,
                limit=limit,
                request_type='orderbook'
            )
            
            # 转换为数据模型
            data_model = OrderBookData.from_ccxt(
                orderbook=orderbook,
                symbol=formatted_symbol,
                exchange=self.exchange
            )
            
            self.logger.info(
                f"异步订单簿获取成功: 买盘 {len(data_model.bids)} 个, "
                f"卖盘 {len(data_model.asks)} 个, "
                f"价差: {data_model.spread:.2f}"
            )
            
            # 缓存结果
            self._cache_set(cache_key, data_model, ttl=30)
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"异步获取订单簿失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return None
    
    async def fetch_trades_async(self, 
                                symbol: str,
                                since: Optional[Union[int, datetime, str]] = None,
                                limit: Optional[int] = None,
                                **kwargs) -> List[TradeData]:
        """
        异步获取现货成交数据
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"异步交易所 {self.exchange} 未初始化")
        
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
            limit = 100
        
        self.logger.info(
            f"异步获取现货成交数据: {formatted_symbol}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'trades_async',
            symbol=formatted_symbol,
            since=since_timestamp,
            limit=limit
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取异步成交数据: {formatted_symbol}")
            return cached_data
        
        try:
            # 使用异步重试机制获取成交数据
            trades = await self._retry_request(
                self.ccxt_exchange.fetch_trades,
                symbol=formatted_symbol,
                since=since_timestamp,
                limit=limit,
                params=kwargs.get('params', {}),
                request_type='trades'
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
                    self.logger.warning(f"转换成交数据失败: {e}")
                    continue
            
            self.logger.info(f"异步获取到 {len(data_models)} 条成交数据")
            
            # 缓存结果
            self._cache_set(cache_key, data_models, ttl=60)
            
            return data_models
            
        except Exception as e:
            self.logger.error(f"异步获取成交数据失败: {e}")
            if kwargs.get('raise_error', False):
                raise
            return []
    
    async def fetch_ticker_async(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        异步获取现货行情数据
        
        参数:
            symbol: 交易对符号
            **kwargs: 额外参数
            
        返回:
            行情数据字典
        """
        if not self.ccxt_exchange:
            raise RuntimeError(f"异步交易所 {self.exchange} 未初始化")
        
        # 格式化交易对
        formatted_symbol = self.format_symbol(symbol)
        
        self.logger.info(f"异步获取现货行情: {formatted_symbol}")
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'ticker_async',
            symbol=formatted_symbol
        )
        
        # 尝试从缓存获取
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            self.logger.info(f"从缓存获取异步行情: {formatted_symbol}")
            return cached_data
        
        try:
            # 使用异步重试机制获取ticker
            ticker = await self._retry_request(
                self.ccxt_exchange.fetch_ticker,
                formatted_symbol,
                request_type='ticker'
            )
            
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
                f"异步行情获取成功: 最新价 {result.get('last')}, "
                f"24h成交量 {result.get('baseVolume')}"
            )
            
            # 缓存结果
            self._cache_set(cache_key, result, ttl=30)
            
            return result
            
        except Exception as e:
            self.logger.error(f"异步获取行情数据失败: {e}")
            return {}
    
    def format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号为交易所标准格式
        
        参数:
            symbol: 原始交易对符号
            
        返回:
            格式化后的交易对符号
        """
        formatted = ExchangeSymbolFormats.format_symbol(symbol, self.exchange, 'spot')
        self.logger.debug(f"格式化交易对: {symbol} -> {formatted}")
        return formatted
    
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
        if self.spot_markets and symbol not in self.spot_markets:
            # 尝试格式标准化后再检查
            formatted_symbol = self.format_symbol(symbol)
            return formatted_symbol in self.spot_markets
        
        return True
    
    async def get_available_symbols_async(self) -> List[str]:
        """
        异步获取可用的现货交易对
        
        返回:
            交易对列表
        """
        if not self.markets_loaded:
            await self._load_markets_async()
        
        return list(self.spot_markets.keys())
    
    async def fetch_market_info_async(self, symbol: str = None) -> Dict[str, Any]:
        """
        异步获取市场信息
        
        参数:
            symbol: 交易对符号（可选）
            
        返回:
            市场信息字典
        """
        if not self.ccxt_exchange:
            return {}
        
        if not self.markets_loaded:
            await self._load_markets_async()
        
        if symbol:
            # 格式化交易对
            formatted_symbol = self.format_symbol(symbol)
            
            if formatted_symbol not in self.spot_markets:
                return {}
            
            market = self.spot_markets[formatted_symbol]
            
            info = {
                'symbol': formatted_symbol,
                'base': market.get('base'),
                'quote': market.get('quote'),
                'baseId': market.get('baseId'),
                'quoteId': market.get('quoteId'),
                'active': market.get('active'),
                'precision': market.get('precision', {}),
                'limits': market.get('limits', {}),
                'taker': market.get('taker'),
                'maker': market.get('maker'),
                'percentage': market.get('percentage'),
                'tierBased': market.get('tierBased'),
                'feeSide': market.get('feeSide')
            }
            
            return info
        else:
            # 返回所有市场
            return self.spot_markets
    
    async def test_connection_async(self) -> bool:
        """
        异步测试交易所连接
        
        返回:
            连接是否成功
        """
        try:
            if not self.ccxt_exchange:
                return False
            
            # 尝试获取服务器时间
            timestamp = await self._retry_request(
                self.ccxt_exchange.fetch_time,
                request_type='test'
            )
            
            self.logger.info(f"异步交易所连接测试成功，服务器时间: {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"异步交易所连接测试失败: {e}")
            return False
    
    async def close_async(self):
        """异步关闭获取器"""
        if self.ccxt_exchange:
            try:
                await self.ccxt_exchange.close()
            except Exception as e:
                self.logger.error(f"关闭异步CCXT连接失败: {e}")
            finally:
                self.ccxt_exchange = None
        
        await super().close_async()


# ==================== 现货数据管理器 ====================

try:
    from crypto_data_system.storage.data_manager import FileDataManager
except (ImportError, ModuleNotFoundError):
    # 脚本直接运行时的备用导入方式
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crypto_data_system.storage.data_manager import FileDataManager


class SpotDataManager(FileDataManager):
    """
    现货数据管理器
    
    管理多个交易对的现货数据获取
    """
    
    def __init__(self, 
                 exchange: str = "binance",
                 fetcher_config: Optional[Dict] = None,
                 root_dir: Optional[str] = None,
                 cache_manager: Optional[CacheManager] = None,
                 save_json_merged: bool = False):
        """
        初始化现货数据管理器
        
        参数:
            exchange: 交易所名称
            fetcher_config: 获取器配置
            cache_manager: 缓存管理器
        """
        self.exchange = exchange
        self.fetcher_config = fetcher_config or {}
        self.cache_manager = cache_manager
        self.save_json_merged = bool(self.fetcher_config.get('save_json_merged', save_json_merged))
        
        self.fetcher = None
        self.symbols = []
        
        # 初始化日志
        self.logger = get_logger(f"spot_manager.{exchange}")

        # 初始化文件存储（子目录按交易所分组）
        super().__init__(root_dir=root_dir, sub_dir=f"spot/{exchange}", file_format="json", cache_manager=cache_manager)
    
    def init_fetcher(self):
        """初始化数据获取器"""
        if not self.fetcher:
            self.fetcher = create_fetcher(
                exchange=self.exchange,
                market_type="spot",
                config=self.fetcher_config,
                cache_manager=self.cache_manager
            )
            
            # 测试连接
            if not self.fetcher.test_connection():
                self.logger.warning(f"交易所 {self.exchange} 连接测试失败")
    
    def add_symbol(self, symbol: str):
        """
        添加交易对
        
        参数:
            symbol: 交易对符号
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.logger.info(f"添加交易对: {symbol}")
    
    def add_symbols(self, symbols: List[str]):
        """
        批量添加交易对
        
        参数:
            symbols: 交易对列表
        """
        for symbol in symbols:
            self.add_symbol(symbol)
    
    def fetch_all_tickers(self) -> Dict[str, Dict]:
        """
        获取所有交易对的行情
        
        返回:
            交易对行情字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        results = {}
        for symbol in self.symbols:
            try:
                ticker = self.fetcher.fetch_ticker(symbol)
                results[symbol] = ticker
            except Exception as e:
                self.logger.error(f"获取交易对 {symbol} 行情失败: {e}")
                results[symbol] = None
        
        # 统一持久化
        self.save_dict(f"{self.exchange}_tickers", results)
        return results
    
    def fetch_all_ohlcv(self, 
                       timeframe: str = "1h",
                       limit: int = 100) -> Dict[str, List[OHLCVData]]:
        """
        获取所有交易对的K线数据
        
        参数:
            timeframe: 时间间隔
            limit: 数据条数限制
            
        返回:
            交易对K线数据字典
        """
        if not self.fetcher:
            self.init_fetcher()
        
        results = {}
        for symbol in self.symbols:
            try:
                ohlcv = self.fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                results[symbol] = ohlcv
            except Exception as e:
                self.logger.error(f"获取交易对 {symbol} K线数据失败: {e}")
                results[symbol] = []
        
        # 统一持久化
        self.save_timestamped(f"{self.exchange}_ohlcv_{timeframe}", results)
        return results

    def fetch_and_save(self,
                       symbol: str,
                       timeframe: str = "1h",
                       start_date: Optional[Union[datetime, str]] = None,
                       end_date: Optional[Union[datetime, str]] = None,
                       limit: Optional[int] = None) -> bool:
        """获取指定交易对的K线数据并保存到本地存储。

        当提供 `start_date` 和 `end_date` 时，使用批量获取以突破单次请求1000条限制；
        否则根据 `limit` 进行单次获取。
        """
        if not self.fetcher:
            self.init_fetcher()

        self.add_symbol(symbol)

        try:
            data_obj = None
            if start_date and end_date and hasattr(self.fetcher, 'fetch_ohlcv_bulk'):
                data_obj = self.fetcher.fetch_ohlcv_bulk(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    max_bars_per_request=1000
                )
            else:
                data_obj = self.fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )

            def _safe_float(v: Any) -> Optional[float]:
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            def _to_timestamp_ms(v: Any) -> Optional[int]:
                if v is None:
                    return None
                try:
                    # datetime / pandas Timestamp
                    if hasattr(v, 'timestamp'):
                        return int(v.timestamp() * 1000)
                    iv = int(v)
                    # seconds -> ms
                    if iv < 10**11:
                        return iv * 1000
                    return iv
                except Exception:
                    return None

            # 转为可序列化结构（兼容 list / dict / OHLCVData / pandas.DataFrame）
            serializable: List[Dict[str, Any]] = []

            # 1) pandas.DataFrame (BaseFetcher.fetch_ohlcv_bulk 默认返回 DataFrame)
            try:
                import pandas as pd  # type: ignore
                if data_obj is not None and isinstance(data_obj, pd.DataFrame):
                    df = data_obj.copy()
                    if len(df) == 0:
                        serializable = []
                    else:
                        # 兼容 timestamp 在列或 index
                        if 'timestamp' not in df.columns:
                            df = df.reset_index().rename(columns={'index': 'timestamp'})

                        # 仅保留必要列 + 可选扩展列
                        optional_cols = [
                            'quote_volume',
                            'trades',
                            'taker_buy_base_volume',
                            'taker_buy_quote_volume',
                            'vwap'
                        ]
                        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        cols += [c for c in optional_cols if c in df.columns]
                        df = df[[c for c in cols if c in df.columns]]

                        # timestamp 归一化为 ms
                        ts_series = df['timestamp']
                        try:
                            ts_dt = pd.to_datetime(ts_series, utc=True, errors='coerce')
                            ts_ms = (ts_dt.astype('int64') // 1_000_000)
                            # NaT 会变成最小 int64；用原值兜底
                            ts_ms = ts_ms.where(ts_dt.notna(), pd.to_numeric(ts_series, errors='coerce'))
                        except Exception:
                            ts_ms = pd.to_numeric(ts_series, errors='coerce')

                        # seconds -> ms
                        try:
                            ts_ms = ts_ms.where(ts_ms > 10**11, ts_ms * 1000)
                        except Exception:
                            pass
                        df['timestamp'] = ts_ms

                        # 数值列转 float
                        for col in ['open', 'high', 'low', 'close', 'volume'] + [c for c in optional_cols if c in df.columns]:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        serializable = df.to_dict('records')
            except Exception:
                # pandas 不可用时跳过 DataFrame 特判
                pass

            # 2) 非 DataFrame：按列表/对象/字典处理
            if not serializable:
                data_list = data_obj if isinstance(data_obj, list) else (list(data_obj) if data_obj is not None and not isinstance(data_obj, (dict, str)) else [])

                for item in data_list:
                    if item is None:
                        continue

                    # OHLCVData-like object
                    if hasattr(item, 'timestamp') and hasattr(item, 'open') and hasattr(item, 'close'):
                        ts_ms = _to_timestamp_ms(getattr(item, 'timestamp', None))
                        row = {
                            'timestamp': ts_ms,
                            'open': _safe_float(getattr(item, 'open', None)),
                            'high': _safe_float(getattr(item, 'high', None)),
                            'low': _safe_float(getattr(item, 'low', None)),
                            'close': _safe_float(getattr(item, 'close', None)),
                            'volume': _safe_float(getattr(item, 'volume', None)),
                        }
                        # 可选扩展字段
                        for k in ['quote_volume', 'trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'vwap']:
                            if hasattr(item, k):
                                row[k] = _safe_float(getattr(item, k, None))
                        if row.get('timestamp') is not None:
                            serializable.append(row)
                        continue

                    # dict format
                    if isinstance(item, dict):
                        ts_ms = _to_timestamp_ms(item.get('timestamp'))
                        row = {
                            'timestamp': ts_ms,
                            'open': _safe_float(item.get('open')),
                            'high': _safe_float(item.get('high')),
                            'low': _safe_float(item.get('low')),
                            'close': _safe_float(item.get('close')),
                            'volume': _safe_float(item.get('volume')),
                        }
                        for k in ['quote_volume', 'trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'vwap']:
                            if k in item:
                                row[k] = _safe_float(item.get(k))
                        if row.get('timestamp') is not None:
                            serializable.append(row)
                        continue

                    # CCXT raw list/tuple [timestamp, open, high, low, close, volume, ...]
                    if isinstance(item, (list, tuple)) and len(item) >= 6:
                        ts_ms = _to_timestamp_ms(item[0])
                        row = {
                            'timestamp': ts_ms,
                            'open': _safe_float(item[1]),
                            'high': _safe_float(item[2]),
                            'low': _safe_float(item[3]),
                            'close': _safe_float(item[4]),
                            'volume': _safe_float(item[5]),
                        }
                        # Binance klines 12 列扩展字段（若存在）
                        if len(item) >= 11:
                            row['quote_volume'] = _safe_float(item[7])
                            row['trades'] = _safe_float(item[8])
                            row['taker_buy_base_volume'] = _safe_float(item[9])
                            row['taker_buy_quote_volume'] = _safe_float(item[10])
                        if row.get('timestamp') is not None:
                            serializable.append(row)
                        continue

            # 清理：过滤掉 timestamp/价格缺失的行
            serializable = [
                r for r in serializable
                if isinstance(r, dict)
                and r.get('timestamp') is not None
                and r.get('open') is not None
                and r.get('high') is not None
                and r.get('low') is not None
                and r.get('close') is not None
                and r.get('volume') is not None
            ]

            # 1) 增量合并保存（按 timestamp 去重）
            symbol_clean = symbol.replace('/', '_')
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

            # existing 先入，新数据覆盖旧数据（若重叠）
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

            # 保存合并后的“唯一数据集”（可选 JSON）
            if self.save_json_merged:
                self.save(merged_key, merged_list)

            # 2) 新增：按 交易对/时间框架 分目录保存为 Parquet
            try:
                import pandas as pd
                if len(merged_list) == 0:
                    return True  # 没有数据也视为流程成功（已完成请求）

                df = pd.DataFrame(merged_list)
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='last')

                # 目录：data_manager_storage/spot/binance/<BTC_USDT>/<1h>/
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
            self.logger.error(f"保存 {symbol} K线数据失败: {e}")
            return False
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        获取市场摘要
        
        返回:
            市场摘要信息
        """
        if not self.fetcher:
            self.init_fetcher()
        
        tickers = self.fetch_all_tickers()
        
        summary = {
            'exchange': self.exchange,
            'total_symbols': len(self.symbols),
            'successful_fetches': sum(1 for v in tickers.values() if v is not None),
            'failed_fetches': sum(1 for v in tickers.values() if v is None),
            'symbols': list(tickers.keys()),
            'data': {}
        }
        
        # 提取关键数据
        for symbol, ticker in tickers.items():
            if ticker:
                summary['data'][symbol] = {
                    'last_price': ticker.get('last'),
                    '24h_change_percent': ticker.get('percentage'),
                    '24h_volume': ticker.get('baseVolume')
                }
        
        return summary
    
    def get_fetcher_status(self) -> Dict[str, Any]:
        """
        获取获取器状态
        
        返回:
            获取器状态信息
        """
        if not self.fetcher:
            return {'fetcher_initialized': False}
        
        return self.fetcher.get_status()
    
    def close(self):
        """关闭管理器"""
        if self.fetcher:
            self.fetcher.close()
            self.fetcher = None


# ==================== 测试函数 ====================

def test_spot_fetcher():
    """测试现货获取器"""
    print("=" * 60)
    print("现货获取器模块测试")
    print("=" * 60)
    
    # 测试基础功能
    print("\n1. 测试CCXTSpotFetcher基础功能:")
    try:
        # 创建获取器（不使用真实API密钥）
        fetcher = CCXTSpotFetcher(exchange="binance")
        
        print(f"✅ 获取器创建成功: {fetcher}")
        print(f"✅ 交易所: {fetcher.exchange}")
        print(f"✅ 市场类型: {fetcher.market_type}")
        
        # 初始化获取器
        if fetcher.initialize():
            print("✅ 获取器初始化成功")
        else:
            print("⚠️  获取器初始化失败")
        
        # 测试连接
        if fetcher.test_connection():
            print("✅ 交易所连接测试成功")
        else:
            print("⚠️  交易所连接测试失败（可能是网络或代理问题）")
        
        # 获取可用交易对
        symbols = fetcher.get_available_symbols()
        print(f"✅ 获取到 {len(symbols)} 个可用交易对")
        if symbols:
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
        
        # 测试订单簿获取
        print("\n3. 测试订单簿获取（模拟）:")
        orderbook = fetcher.fetch_orderbook(test_symbol, limit=5)
        if orderbook:
            print(f"✅ 订单簿获取成功")
            print(f"  最佳买价: {orderbook.bids[0][0] if orderbook.bids else 'N/A'}")
            print(f"  最佳卖价: {orderbook.asks[0][0] if orderbook.asks else 'N/A'}")
            print(f"  价差: {orderbook.spread}")
        else:
            print("⚠️  订单簿获取失败")
        
        # 测试行情获取
        print("\n4. 测试行情数据获取（模拟）:")
        ticker = fetcher.fetch_ticker(test_symbol)
        if ticker:
            print(f"✅ 行情数据获取成功")
            print(f"  最新价: {ticker.get('last')}")
            print(f"  24h涨跌幅: {ticker.get('percentage')}%")
        else:
            print("⚠️  行情数据获取失败")
        
        # 测试批量获取
        print("\n5. 测试批量数据获取（模拟）:")
        bulk_data = fetcher.fetch_ohlcv_bulk(
            symbol=test_symbol,
            start_date="2024-01-01",
            end_date="2024-01-02",
            timeframe="1h",
            max_bars_per_request=10
        )
        print(f"✅ 批量获取完成: {len(bulk_data) if bulk_data is not None else 0} 条数据")
        
        # 测试市场深度分析
        print("\n6. 测试市场深度分析（模拟）:")
        depth_analysis = fetcher.get_market_depth(test_symbol, depth_levels=5)
        if depth_analysis:
            print(f"✅ 市场深度分析成功")
            print(f"  买卖压力: 买盘 {depth_analysis.get('bid_pressure', 0):.2%}, "
                  f"卖盘 {depth_analysis.get('ask_pressure', 0):.2%}")
        else:
            print("⚠️  市场深度分析失败")
        
        # 获取状态
        status = fetcher.get_status()
        print(f"\n✅ 获取器状态:")
        print(f"  交易所: {status['exchange']}")
        print(f"  请求总数: {status['request_stats']['total_requests']}")
        print(f"  成功率: {status['request_stats']['success_rate']:.2f}%")
        
        # 关闭获取器
        fetcher.close()
        print("\n✅ 获取器关闭成功")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试现货数据管理器
    print("\n7. 测试现货数据管理器:")
    try:
        manager = SpotDataManager(exchange="binance")
        manager.add_symbols(["BTC/USDT", "ETH/USDT", "BNB/USDT"])
        
        print(f"✅ 管理器创建成功，交易对: {manager.symbols}")
        
        # 获取市场摘要
        summary = manager.get_market_summary()
        print(f"✅ 市场摘要: {summary['total_symbols']} 个交易对")
        
        manager.close()
        print("✅ 管理器关闭成功")
        
    except Exception as e:
        print(f"❌ 管理器测试失败: {e}")
    
    # 测试工厂函数
    print("\n8. 测试工厂函数:")
    try:
        spot_fetcher = create_fetcher(exchange="binance", market_type="spot")
        print(f"✅ 工厂函数创建成功: {spot_fetcher}")
        print(f"✅ 类型: {type(spot_fetcher).__name__}")
        
    except Exception as e:
        print(f"❌ 工厂函数测试失败: {e}")
    
    print("\n✅ 现货获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_spot_fetcher()