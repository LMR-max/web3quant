"""
数据聚合模块 - 合并不同类型的市场数据

提供统一的数据容器，可以将同一交易对的不同数据类型（K线、Ticker、OrderBook等）
按时间对齐并合并，支持多维度数据分析。
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from data_models import (
    OHLCVData, OrderBookData, TradeData, FundingRateData,
    OpenInterestData, LiquidationData, SwapOHLCVData
)


@dataclass
class MarketSnapshot:
    """市场快照 - 某个时间点的完整市场数据"""
    timestamp: pd.Timestamp
    symbol: str
    exchange: str = "binance"
    market_type: str = "spot"
    
    # K线数据（不同时间周期）
    ohlcv_1m: Optional[OHLCVData] = None
    ohlcv_5m: Optional[OHLCVData] = None
    ohlcv_15m: Optional[OHLCVData] = None
    ohlcv_1h: Optional[OHLCVData] = None
    ohlcv_4h: Optional[OHLCVData] = None
    ohlcv_1d: Optional[OHLCVData] = None
    
    # 实时行情
    ticker: Optional[Dict[str, Any]] = None
    
    # 订单簿
    orderbook: Optional[OrderBookData] = None
    
    # 成交记录
    recent_trades: List[TradeData] = field(default_factory=list)
    
    # 衍生品专属数据
    funding_rate: Optional[FundingRateData] = None
    open_interest: Optional[OpenInterestData] = None
    liquidations: List[LiquidationData] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'market_type': self.market_type,
        }
        
        # 添加K线数据
        for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
            ohlcv = getattr(self, f'ohlcv_{tf}', None)
            if ohlcv:
                result[f'ohlcv_{tf}'] = {
                    'open': ohlcv.open,
                    'high': ohlcv.high,
                    'low': ohlcv.low,
                    'close': ohlcv.close,
                    'volume': ohlcv.volume,
                }
        
        # 添加Ticker数据
        if self.ticker:
            result['ticker'] = self.ticker
        
        # 添加OrderBook摘要
        if self.orderbook:
            result['orderbook'] = self.orderbook.get_summary()
        
        # 添加衍生品数据
        if self.funding_rate:
            result['funding_rate'] = self.funding_rate.funding_rate
        
        if self.open_interest:
            result['open_interest'] = self.open_interest.open_interest
        
        return result
    
    def get_price_info(self) -> Dict[str, float]:
        """获取价格信息摘要"""
        info = {}
        
        # 从K线获取
        if self.ohlcv_1m:
            info['close_1m'] = self.ohlcv_1m.close
            info['volume_1m'] = self.ohlcv_1m.volume
        
        if self.ohlcv_1h:
            info['close_1h'] = self.ohlcv_1h.close
            info['volume_1h'] = self.ohlcv_1h.volume
        
        # 从Ticker获取
        if self.ticker:
            info['last'] = self.ticker.get('last', 0)
            info['bid'] = self.ticker.get('bid', 0)
            info['ask'] = self.ticker.get('ask', 0)
            info['change_24h'] = self.ticker.get('percentage', 0)
        
        # 从OrderBook获取
        if self.orderbook:
            info['spread'] = self.orderbook.spread
            info['spread_percent'] = (self.orderbook.spread / self.ticker.get('last', 1)) * 100 if self.ticker else 0
            info['best_bid'] = self.orderbook.bids[0][0] if self.orderbook.bids else 0
            info['best_ask'] = self.orderbook.asks[0][0] if self.orderbook.asks else 0
        
        return info


class MarketDataAggregator:
    """市场数据聚合器 - 管理和合并多种数据类型"""
    
    def __init__(self, symbol: str, exchange: str = "binance", market_type: str = "spot"):
        """
        初始化数据聚合器
        
        参数:
            symbol: 交易对符号
            exchange: 交易所名称
            market_type: 市场类型
        """
        self.symbol = symbol
        self.exchange = exchange
        self.market_type = market_type
        
        # 存储不同类型的数据
        self.ohlcv_data: Dict[str, List[OHLCVData]] = defaultdict(list)  # timeframe -> data list
        self.ticker_history: List[Dict] = []
        self.orderbook_history: List[OrderBookData] = []
        self.trade_history: List[TradeData] = []
        
        # 衍生品数据
        self.funding_rate_history: List[FundingRateData] = []
        self.open_interest_history: List[OpenInterestData] = []
        self.liquidation_history: List[LiquidationData] = []
    
    def add_ohlcv(self, ohlcv_data: Union[OHLCVData, List[OHLCVData]], timeframe: str = "1h"):
        """添加K线数据"""
        if isinstance(ohlcv_data, list):
            self.ohlcv_data[timeframe].extend(ohlcv_data)
        else:
            self.ohlcv_data[timeframe].append(ohlcv_data)
    
    def add_ticker(self, ticker_data: Dict):
        """添加Ticker数据"""
        if 'timestamp' not in ticker_data:
            ticker_data['timestamp'] = pd.Timestamp.now()
        self.ticker_history.append(ticker_data)
    
    def add_orderbook(self, orderbook: OrderBookData):
        """添加OrderBook数据"""
        self.orderbook_history.append(orderbook)
    
    def add_trades(self, trades: Union[TradeData, List[TradeData]]):
        """添加成交记录"""
        if isinstance(trades, list):
            self.trade_history.extend(trades)
        else:
            self.trade_history.append(trades)
    
    def add_funding_rate(self, funding_rate: FundingRateData):
        """添加资金费率数据"""
        self.funding_rate_history.append(funding_rate)
    
    def add_open_interest(self, open_interest: OpenInterestData):
        """添加持仓量数据"""
        self.open_interest_history.append(open_interest)
    
    def add_liquidation(self, liquidation: LiquidationData):
        """添加强平数据"""
        self.liquidation_history.append(liquidation)
    
    def get_snapshot(self, timestamp: pd.Timestamp, 
                    time_tolerance: timedelta = timedelta(minutes=1)) -> MarketSnapshot:
        """
        获取指定时间的市场快照
        
        参数:
            timestamp: 目标时间
            time_tolerance: 时间容差（用于匹配最接近的数据）
        
        返回:
            MarketSnapshot对象
        """
        snapshot = MarketSnapshot(
            timestamp=timestamp,
            symbol=self.symbol,
            exchange=self.exchange,
            market_type=self.market_type
        )
        
        # 查找最接近的K线数据
        for timeframe, data_list in self.ohlcv_data.items():
            closest = self._find_closest_data(data_list, timestamp, time_tolerance)
            if closest:
                setattr(snapshot, f'ohlcv_{timeframe}', closest)
        
        # 查找最接近的Ticker
        snapshot.ticker = self._find_closest_data(self.ticker_history, timestamp, time_tolerance)
        
        # 查找最接近的OrderBook
        snapshot.orderbook = self._find_closest_data(self.orderbook_history, timestamp, time_tolerance)
        
        # 查找时间范围内的成交记录
        snapshot.recent_trades = self._find_data_in_range(
            self.trade_history, 
            timestamp - time_tolerance, 
            timestamp + time_tolerance
        )
        
        # 衍生品数据
        snapshot.funding_rate = self._find_closest_data(self.funding_rate_history, timestamp, time_tolerance)
        snapshot.open_interest = self._find_closest_data(self.open_interest_history, timestamp, time_tolerance)
        snapshot.liquidations = self._find_data_in_range(
            self.liquidation_history,
            timestamp - time_tolerance,
            timestamp + time_tolerance
        )
        
        return snapshot
    
    def merge_to_dataframe(self, 
                          timeframe: str = "1h",
                          include_ticker: bool = True,
                          include_orderbook: bool = True,
                          include_trades: bool = False) -> pd.DataFrame:
        """
        将多种数据类型合并为单个DataFrame
        
        参数:
            timeframe: 主时间周期（用作基础索引）
            include_ticker: 是否包含Ticker数据
            include_orderbook: 是否包含OrderBook数据
            include_trades: 是否包含成交数据统计
        
        返回:
            合并后的DataFrame
        """
        # 以K线数据为基础
        if timeframe not in self.ohlcv_data or not self.ohlcv_data[timeframe]:
            return pd.DataFrame()
        
        # 转换K线数据为DataFrame
        ohlcv_list = self.ohlcv_data[timeframe]
        df_ohlcv = pd.DataFrame([{
            'timestamp': d.timestamp,
            'open': d.open,
            'high': d.high,
            'low': d.low,
            'close': d.close,
            'volume': d.volume,
            'quote_volume': getattr(d, 'quote_volume', 0),
            'trades': getattr(d, 'trades', 0),
        } for d in ohlcv_list])
        
        df_ohlcv.set_index('timestamp', inplace=True)
        df_ohlcv.sort_index(inplace=True)
        
        # 合并Ticker数据
        if include_ticker and self.ticker_history:
            df_ticker = pd.DataFrame(self.ticker_history)
            if 'timestamp' in df_ticker.columns:
                df_ticker['timestamp'] = pd.to_datetime(df_ticker['timestamp'])
                df_ticker.set_index('timestamp', inplace=True)
                
                # 选择关键列
                ticker_cols = ['last', 'bid', 'ask', 'percentage', 'baseVolume', 'quoteVolume']
                available_cols = [col for col in ticker_cols if col in df_ticker.columns]
                
                if available_cols:
                    df_ticker = df_ticker[available_cols].add_prefix('ticker_')
                    # 使用最近的Ticker数据填充K线时间点
                    df_ohlcv = df_ohlcv.join(df_ticker, how='left')
            df_ohlcv.ffill(inplace=True)  # 使用 ffill 替代 fillna(method='ffill')
        if include_orderbook and self.orderbook_history:
            orderbook_data = []
            for ob in self.orderbook_history:
                orderbook_data.append({
                    'timestamp': ob.timestamp,
                    'spread': ob.spread,
                    'bid_total': ob.bid_total,
                    'ask_total': ob.ask_total,
                    'depth': ob.depth,
                    'best_bid': ob.bids[0][0] if ob.bids else 0,
                    'best_ask': ob.asks[0][0] if ob.asks else 0,
                })
            
            df_ob = pd.DataFrame(orderbook_data)
            df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'])
            df_ob.set_index('timestamp', inplace=True)
            df_ob = df_ob.add_prefix('ob_')
            
            # 使用最近的OrderBook数据填充K线时间点
            df_ohlcv = df_ohlcv.join(df_ob, how='left')
            df_ohlcv.ffill(inplace=True)  # 使用 ffill 替代 fillna(method='ffill')
        
        # 统计成交数据
        if include_trades and self.trade_history:
            # 按K线时间段聚合成交数据
            trade_data = []
            for trade in self.trade_history:
                trade_data.append({
                    'timestamp': trade.timestamp,
                    'price': trade.price,
                    'amount': trade.amount,
                    'side': trade.side,
                })
            
            df_trades = pd.DataFrame(trade_data)
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
            
            # 按K线周期重采样
            df_trades['buy_volume'] = df_trades.apply(
                lambda x: x['amount'] if x['side'] == 'buy' else 0, axis=1
            )
            df_trades['sell_volume'] = df_trades.apply(
                lambda x: x['amount'] if x['side'] == 'sell' else 0, axis=1
            )
            
            df_trades.set_index('timestamp', inplace=True)
            
            # 根据timeframe进行重采样
            resample_rule = self._get_resample_rule(timeframe)
            df_trades_agg = df_trades.resample(resample_rule).agg({
                'amount': 'sum',
                'buy_volume': 'sum',
                'sell_volume': 'sum',
            }).add_prefix('trade_')
            
            df_ohlcv = df_ohlcv.join(df_trades_agg, how='left')
        
        return df_ohlcv
    
    def get_aligned_data(self, 
                        start_time: Optional[pd.Timestamp] = None,
                        end_time: Optional[pd.Timestamp] = None,
                        timeframe: str = "1h") -> pd.DataFrame:
        """
        获取时间对齐后的综合数据
        
        参数:
            start_time: 开始时间
            end_time: 结束时间
            timeframe: 时间周期
        
        返回:
            时间对齐的DataFrame
        """
        df = self.merge_to_dataframe(
            timeframe=timeframe,
            include_ticker=True,
            include_orderbook=True,
            include_trades=True
        )
        
        if start_time:
            df = df[df.index >= start_time]
        
        if end_time:
            df = df[df.index <= end_time]
        
        return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'market_type': self.market_type,
            'data_counts': {
                'ohlcv': {tf: len(data) for tf, data in self.ohlcv_data.items()},
                'ticker': len(self.ticker_history),
                'orderbook': len(self.orderbook_history),
                'trades': len(self.trade_history),
            }
        }
        
        # 时间范围
        if self.ohlcv_data:
            all_ohlcv = []
            for data_list in self.ohlcv_data.values():
                all_ohlcv.extend(data_list)
            
            if all_ohlcv:
                timestamps = [d.timestamp for d in all_ohlcv]
                stats['time_range'] = {
                    'start': min(timestamps),
                    'end': max(timestamps),
                    'duration': max(timestamps) - min(timestamps)
                }
        
        return stats
    
    def _find_closest_data(self, data_list: List, target_time: pd.Timestamp, 
                          tolerance: timedelta) -> Optional[Any]:
        """查找最接近目标时间的数据"""
        if not data_list:
            return None
        
        closest = None
        min_diff = tolerance
        
        for item in data_list:
            # 获取时间戳
            if isinstance(item, dict):
                ts = item.get('timestamp')
            else:
                ts = getattr(item, 'timestamp', None)
            
            if ts is None:
                continue
            
            # 确保是Timestamp类型
            if not isinstance(ts, pd.Timestamp):
                try:
                    ts = pd.Timestamp(ts)
                except:
                    continue
            
            # 检查是否为NaT
            if pd.isna(ts):
                continue
            
            try:
                diff = abs(ts - target_time)
                if diff < min_diff:
                    min_diff = diff
                    closest = item
            except:
                continue
        
        return closest
    
    def _find_data_in_range(self, data_list: List, start_time: pd.Timestamp, 
                           end_time: pd.Timestamp) -> List[Any]:
        """查找时间范围内的所有数据"""
        result = []
        
        for item in data_list:
            # 获取时间戳
            if isinstance(item, dict):
                ts = item.get('timestamp')
            else:
                ts = getattr(item, 'timestamp', None)
            
            if ts is None:
                continue
            
            # 确保是Timestamp类型
            if not isinstance(ts, pd.Timestamp):
                ts = pd.Timestamp(ts)
            
            if start_time <= ts <= end_time:
                result.append(item)
        
        return result
    
    def _get_resample_rule(self, timeframe: str) -> str:
        """获取pandas resample规则"""
        mapping = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W',
        }
        return mapping.get(timeframe, '1H')
    
    def clear(self):
        """清空所有数据"""
        self.ohlcv_data.clear()
        self.ticker_history.clear()
        self.orderbook_history.clear()
        self.trade_history.clear()
        self.funding_rate_history.clear()
        self.open_interest_history.clear()
        self.liquidation_history.clear()


class MultiSymbolAggregator:
    """多交易对数据聚合器"""
    
    def __init__(self, exchange: str = "binance", market_type: str = "spot"):
        """
        初始化多交易对聚合器
        
        参数:
            exchange: 交易所名称
            market_type: 市场类型
        """
        self.exchange = exchange
        self.market_type = market_type
        self.aggregators: Dict[str, MarketDataAggregator] = {}
    
    def get_aggregator(self, symbol: str) -> MarketDataAggregator:
        """获取或创建指定交易对的聚合器"""
        if symbol not in self.aggregators:
            self.aggregators[symbol] = MarketDataAggregator(
                symbol=symbol,
                exchange=self.exchange,
                market_type=self.market_type
            )
        return self.aggregators[symbol]
    
    def add_data(self, symbol: str, data_type: str, data: Any, **kwargs):
        """
        添加数据到指定交易对
        
        参数:
            symbol: 交易对符号
            data_type: 数据类型 ('ohlcv', 'ticker', 'orderbook', 'trades')
            data: 数据对象
            **kwargs: 额外参数（如timeframe）
        """
        aggregator = self.get_aggregator(symbol)
        
        if data_type == 'ohlcv':
            timeframe = kwargs.get('timeframe', '1h')
            aggregator.add_ohlcv(data, timeframe)
        elif data_type == 'ticker':
            aggregator.add_ticker(data)
        elif data_type == 'orderbook':
            aggregator.add_orderbook(data)
        elif data_type == 'trades':
            aggregator.add_trades(data)
        elif data_type == 'funding_rate':
            aggregator.add_funding_rate(data)
        elif data_type == 'open_interest':
            aggregator.add_open_interest(data)
        elif data_type == 'liquidation':
            aggregator.add_liquidation(data)
    
    def get_multi_symbol_dataframe(self, 
                                  symbols: List[str],
                                  timeframe: str = "1h",
                                  start_time: Optional[pd.Timestamp] = None,
                                  end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        获取多个交易对的合并DataFrame
        
        参数:
            symbols: 交易对列表
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
        
        返回:
            多交易对合并的DataFrame（多级索引）
        """
        dfs = []
        
        for symbol in symbols:
            if symbol in self.aggregators:
                df = self.aggregators[symbol].get_aligned_data(
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
                df['symbol'] = symbol
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # 合并所有DataFrame
        combined = pd.concat(dfs)
        combined.set_index('symbol', append=True, inplace=True)
        
        return combined
    
    def get_correlation_matrix(self, 
                              symbols: List[str],
                              column: str = 'close',
                              timeframe: str = "1h") -> pd.DataFrame:
        """
        计算多个交易对之间的相关性矩阵
        
        参数:
            symbols: 交易对列表
            column: 用于计算相关性的列
            timeframe: 时间周期
        
        返回:
            相关性矩阵DataFrame
        """
        data_dict = {}
        
        for symbol in symbols:
            if symbol in self.aggregators:
                df = self.aggregators[symbol].merge_to_dataframe(
                    timeframe=timeframe,
                    include_ticker=False,
                    include_orderbook=False
                )
                if column in df.columns:
                    data_dict[symbol] = df[column]
        
        if not data_dict:
            return pd.DataFrame()
        
        # 构建DataFrame并计算相关性
        price_df = pd.DataFrame(data_dict)
        correlation = price_df.corr()
        
        return correlation
