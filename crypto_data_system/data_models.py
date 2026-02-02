"""
数据模型模块 - 定义统一的数据结构和格式
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum

# ==================== 枚举定义 ====================

class MarketType(Enum):
    """市场类型枚举"""
    SPOT = "spot"
    SWAP = "swap"  # 永续合约
    MARGIN = "margin"
    FUTURE = "future"
    OPTION = "option"
    

class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"
    

class OrderType(Enum):
    """订单类型枚举"""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    

class Timeframe(Enum):
    """时间间隔枚举"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    

class PositionSide(Enum):
    """持仓方向枚举"""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"
    

class OptionType(Enum):
    """期权类型枚举"""
    CALL = "call"
    PUT = "put"
    

class GreeksType(Enum):
    """希腊值类型枚举"""
    DELTA = "delta"
    GAMMA = "gamma"
    THETA = "theta"
    VEGA = "vega"
    RHO = "rho"

# ==================== 基础数据类 ====================

@dataclass
class BaseData:
    """基础数据类"""
    timestamp: pd.Timestamp
    symbol: str
    source: str = "unknown"
    exchange: str = "unknown"
    market_type: str = "spot"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_series(self) -> pd.Series:
        """转换为Pandas Series"""
        return pd.Series(self.to_dict())
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为Pandas DataFrame（单行）"""
        return pd.DataFrame([self.to_dict()])


@dataclass
class TimeSeriesData(BaseData):
    """时间序列数据基类"""
    timeframe: str = "1h"
    
    def __post_init__(self):
        """后初始化处理"""
        if isinstance(self.timestamp, (int, float)):
            self.timestamp = pd.Timestamp(self.timestamp, unit='ms')
        elif isinstance(self.timestamp, str):
            self.timestamp = pd.Timestamp(self.timestamp)


# ==================== 现货市场数据 ====================

@dataclass
class OHLCVData(TimeSeriesData):
    """K线数据"""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_base_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0
    vwap: Optional[float] = None
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    
    @classmethod
    def from_ccxt(cls, ohlcv: List, symbol: str, timeframe: str, 
                  exchange: str = "binance", market_type: str = "spot") -> 'OHLCVData':
        """从CCXT OHLCV格式转换"""
        if len(ohlcv) < 6:
            raise ValueError(f"OHLCV数据需要至少6个元素，实际得到{len(ohlcv)}个")

        # 标准 CCXT: [timestamp, open, high, low, close, volume]
        # Binance /api/v3/klines: 12 columns
        # [openTime, open, high, low, close, volume, closeTime, quoteVolume, numberOfTrades, takerBuyBaseVolume, takerBuyQuoteVolume, ignore]
        quote_volume = 0.0
        trades = 0
        taker_buy_base_volume = 0.0
        taker_buy_quote_volume = 0.0
        vwap = None
        if len(ohlcv) >= 11:
            try:
                quote_volume = float(ohlcv[7]) if ohlcv[7] is not None else 0.0
            except Exception:
                quote_volume = 0.0
            try:
                trades = int(float(ohlcv[8])) if ohlcv[8] is not None else 0
            except Exception:
                trades = 0
            try:
                taker_buy_base_volume = float(ohlcv[9]) if ohlcv[9] is not None else 0.0
            except Exception:
                taker_buy_base_volume = 0.0
            try:
                taker_buy_quote_volume = float(ohlcv[10]) if ohlcv[10] is not None else 0.0
            except Exception:
                taker_buy_quote_volume = 0.0

            try:
                vol = float(ohlcv[5])
                if vol > 0 and quote_volume > 0:
                    vwap = quote_volume / vol
            except Exception:
                vwap = None

        ts_val = ohlcv[0]
        if isinstance(ts_val, str):
            try:
                # Binance klines 可能返回字符串时间戳
                ts_val = int(float(ts_val))
            except Exception:
                pass

        timestamp = None
        if isinstance(ts_val, (int, float, np.integer, np.floating)):
            timestamp = pd.Timestamp(int(ts_val), unit='ms')
        else:
            timestamp = pd.Timestamp(ts_val)

        return cls(
            timestamp=timestamp,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            open=float(ohlcv[1]),
            high=float(ohlcv[2]),
            low=float(ohlcv[3]),
            close=float(ohlcv[4]),
            volume=float(ohlcv[5]),
            quote_volume=quote_volume,
            trades=trades,
            taker_buy_base_volume=taker_buy_base_volume,
            taker_buy_quote_volume=taker_buy_quote_volume,
            vwap=vwap,
        )
    
    def to_dataframe(self, include_indicators: bool = True) -> pd.DataFrame:
        """转换为DataFrame"""
        df = pd.DataFrame([{
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'trades': self.trades,
            'taker_buy_base_volume': self.taker_buy_base_volume,
            'taker_buy_quote_volume': self.taker_buy_quote_volume,
            'timeframe': self.timeframe,
            'exchange': self.exchange,
            'market_type': self.market_type
        }])
        
        if include_indicators and self.vwap is not None:
            df['vwap'] = self.vwap
        if include_indicators and self.rsi is not None:
            df['rsi'] = self.rsi
        if include_indicators and self.sma_20 is not None:
            df['sma_20'] = self.sma_20
        if include_indicators and self.sma_50 is not None:
            df['sma_50'] = self.sma_50
        if include_indicators and self.sma_200 is not None:
            df['sma_200'] = self.sma_200

        if include_indicators and self.ema_12 is not None:
            df['ema_12'] = self.ema_12
        if include_indicators and self.ema_26 is not None:
            df['ema_26'] = self.ema_26
        if include_indicators and self.macd is not None:
            df['macd'] = self.macd
        if include_indicators and self.macd_signal is not None:
            df['macd_signal'] = self.macd_signal
        if include_indicators and self.macd_histogram is not None:
            df['macd_histogram'] = self.macd_histogram

        if include_indicators and self.bollinger_upper is not None:
            df['bollinger_upper'] = self.bollinger_upper
        if include_indicators and self.bollinger_middle is not None:
            df['bollinger_middle'] = self.bollinger_middle
        if include_indicators and self.bollinger_lower is not None:
            df['bollinger_lower'] = self.bollinger_lower
            
        df.set_index('timestamp', inplace=True)
        return df


@dataclass
class OrderBookData(BaseData):
    """订单簿数据"""
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (价格, 数量)
    asks: List[Tuple[float, float]] = field(default_factory=list)  # (价格, 数量)
    bid_total: float = 0.0
    ask_total: float = 0.0
    spread: float = 0.0
    depth: float = 0.0
    bid_volume_1_percent: float = 0.0  # 买盘价格上下1%的挂单量
    ask_volume_1_percent: float = 0.0  # 卖盘价格上下1%的挂单量
    
    @classmethod
    def from_ccxt(cls, orderbook: Dict, symbol: str, 
                  exchange: str = "binance") -> 'OrderBookData':
        """从CCXT订单簿格式转换"""
        bids = [(float(bid[0]), float(bid[1])) for bid in orderbook.get('bids', [])]
        asks = [(float(ask[0]), float(ask[1])) for ask in orderbook.get('asks', [])]
        
        # 计算订单簿指标
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid
        
        bid_total = sum(amount for _, amount in bids)
        ask_total = sum(amount for _, amount in asks)
        
        # 计算深度（价格上下1%范围内的挂单量）
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        bid_depth = sum(amount for price, amount in bids if price >= mid_price * 0.99)
        ask_depth = sum(amount for price, amount in asks if price <= mid_price * 1.01)
        
        return cls(
            timestamp=pd.Timestamp(orderbook.get('timestamp', pd.Timestamp.now())),
            symbol=symbol,
            exchange=exchange,
            market_type="spot",
            bids=bids,
            asks=asks,
            bid_total=bid_total,
            ask_total=ask_total,
            spread=spread,
            depth=bid_depth + ask_depth,
            bid_volume_1_percent=bid_depth,
            ask_volume_1_percent=ask_depth
        )
    
    def get_summary(self, levels: int = 5) -> Dict:
        """获取订单簿摘要"""
        return {
            'best_bid': self.bids[0][0] if self.bids else 0,
            'best_ask': self.asks[0][0] if self.asks else 0,
            'spread': self.spread,
            'spread_percent': (self.spread / ((self.bids[0][0] + self.asks[0][0]) / 2)) * 100 
                            if self.bids and self.asks else 0,
            'bid_total': self.bid_total,
            'ask_total': self.ask_total,
            'imbalance': (self.bid_total - self.ask_total) / (self.bid_total + self.ask_total) 
                        if (self.bid_total + self.ask_total) > 0 else 0,
            'bids_top5': self.bids[:min(levels, len(self.bids))],
            'asks_top5': self.asks[:min(levels, len(self.asks))]
        }


@dataclass
class TradeData(BaseData):
    """成交数据"""
    trade_id: str = ""
    price: float = 0.0
    amount: float = 0.0
    side: str = ""  # buy/sell
    taker: bool = True
    order_type: str = "market"
    
    @classmethod
    def from_ccxt(cls, trade: Dict, symbol: str, 
                  exchange: str = "binance") -> 'TradeData':
        """从CCXT成交数据格式转换"""
        return cls(
            timestamp=pd.Timestamp(trade['timestamp'], unit='ms'),
            symbol=symbol,
            exchange=exchange,
            market_type="spot",
            trade_id=str(trade['id']),
            price=float(trade['price']),
            amount=float(trade['amount']),
            side=trade['side'],
            taker=trade.get('taker', True)
        )


# ==================== 衍生品市场数据 ====================

@dataclass
class SwapOHLCVData(OHLCVData):
    """永续合约K线数据"""
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    basis: Optional[float] = None  # 现货与合约价差
    basis_percent: Optional[float] = None  # 价差百分比
    
    def calculate_basis(self, spot_price: float) -> None:
        """计算基差"""
        if spot_price > 0:
            self.basis = self.close - spot_price
            self.basis_percent = (self.basis / spot_price) * 100


@dataclass
class FundingRateData(BaseData):
    """资金费率数据"""
    funding_rate: float = 0.0
    funding_time: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    predicted_rate: Optional[float] = None
    interval_hours: int = 8
    cumulative_rate: Optional[float] = None  # 累计资金费率
    
    def __post_init__(self):
        """后初始化处理"""
        if isinstance(self.funding_time, (int, float)):
            self.funding_time = pd.Timestamp(self.funding_time, unit='ms')
        elif isinstance(self.funding_time, str):
            self.funding_time = pd.Timestamp(self.funding_time)


@dataclass
class OpenInterestData(BaseData):
    """未平仓合约数据"""
    open_interest: float = 0.0
    open_interest_value: float = 0.0
    volume_24h: float = 0.0
    turnover_24h: float = 0.0
    open_interest_change: Optional[float] = None  # 24小时变化
    open_interest_change_percent: Optional[float] = None  # 24小时变化百分比
    
    def calculate_changes(self, previous_oi: 'OpenInterestData') -> None:
        """计算变化率"""
        if previous_oi and previous_oi.open_interest > 0:
            self.open_interest_change = self.open_interest - previous_oi.open_interest
            self.open_interest_change_percent = (
                self.open_interest_change / previous_oi.open_interest * 100
            )


@dataclass
class LiquidationData(BaseData):
    """强平数据"""
    side: str = ""  # buy/sell
    quantity: float = 0.0
    price: float = 0.0
    value: float = 0.0
    liquidation_type: str = "partial"  # partial/full
    leverage: Optional[float] = None  # 杠杆倍数
    
    @property
    def is_buy_liquidation(self) -> bool:
        """是否为买入强平（空头平仓）"""
        return self.side.lower() == 'buy'
    
    @property
    def is_sell_liquidation(self) -> bool:
        """是否为卖出强平（多头平仓）"""
        return self.side.lower() == 'sell'


@dataclass
class PositionData(BaseData):
    """持仓数据"""
    side: str = ""  # long/short
    size: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0
    leverage: float = 1.0
    liquidation_price: Optional[float] = None
    margin_mode: str = "cross"  # cross/isolated
    position_mode: str = "hedge"  # hedge/one-way


# ==================== 期货数据 ====================

@dataclass
class FutureContractData(BaseData):
    """期货合约数据"""
    contract_symbol: str = ""
    expiry_date: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    contract_size: float = 1.0
    settlement_asset: str = ""
    contract_type: str = "inverse"  # inverse/linear
    tick_size: float = 0.01
    lot_size: float = 1.0
    is_active: bool = True
    last_trading_date: Optional[pd.Timestamp] = None
    delivery_date: Optional[pd.Timestamp] = None
    basis: Optional[float] = None  # 期现价差
    annualized_basis: Optional[float] = None  # 年化基差
    
    def calculate_basis(self, spot_price: float, future_price: float) -> None:
        """计算基差"""
        if spot_price > 0 and future_price > 0:
            self.basis = future_price - spot_price
            
            # 计算年化基差
            days_to_expiry = (self.expiry_date - pd.Timestamp.now()).days
            if days_to_expiry > 0:
                self.annualized_basis = (self.basis / spot_price) * (365 / days_to_expiry) * 100


@dataclass
class TermStructureData:
    """期货期限结构数据"""
    timestamp: pd.Timestamp
    symbol: str
    contracts: List[FutureContractData] = field(default_factory=list)
    spot_price: float = 0.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = []
        for contract in self.contracts:
            data.append({
                'timestamp': self.timestamp,
                'symbol': self.symbol,
                'contract_symbol': contract.contract_symbol,
                'expiry_date': contract.expiry_date,
                'contract_size': contract.contract_size,
                'basis': contract.basis,
                'annualized_basis': contract.annualized_basis,
                'days_to_expiry': (contract.expiry_date - pd.Timestamp.now()).days
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


# ==================== 期权数据 ====================

@dataclass
class OptionContractData(BaseData):
    """期权合约数据"""
    option_symbol: str = ""
    strike_price: float = 0.0
    expiry_date: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    option_type: str = ""  # call/put
    underlying_symbol: str = ""
    contract_size: float = 1.0
    is_european: bool = True  # 欧式/美式期权
    is_active: bool = True
    last_trading_date: Optional[pd.Timestamp] = None
    settlement_date: Optional[pd.Timestamp] = None


@dataclass
class GreeksData(BaseData):
    """期权希腊值数据"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    iv: float = 0.0  # 隐含波动率
    iv_rank: Optional[float] = None  # 隐含波动率百分位
    iv_percentile: Optional[float] = None  # 隐含波动率百分比
    option_symbol: Optional[str] = None
    strike_price: Optional[float] = None
    expiry_date: Optional[pd.Timestamp] = None
    option_type: Optional[str] = None  # call/put
    
    def get_summary(self) -> Dict:
        """获取希腊值摘要"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'iv': self.iv,
            'iv_rank': self.iv_rank,
            'option_type': self.option_type
        }


@dataclass
class VolatilitySurfaceData:
    """波动率曲面数据"""
    timestamp: pd.Timestamp
    symbol: str
    strikes: List[float] = field(default_factory=list)
    expiries: List[pd.Timestamp] = field(default_factory=list)
    iv_matrix: np.ndarray = field(default_factory=lambda: np.array([]))  # 隐含波动率矩阵
    delta_matrix: Optional[np.ndarray] = None  # Delta矩阵
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = []
        for i, expiry in enumerate(self.expiries):
            for j, strike in enumerate(self.strikes):
                data.append({
                    'timestamp': self.timestamp,
                    'symbol': self.symbol,
                    'expiry_date': expiry,
                    'strike': strike,
                    'iv': self.iv_matrix[i, j] if i < len(self.iv_matrix) and j < len(self.iv_matrix[i]) else None,
                    'days_to_expiry': (expiry - pd.Timestamp.now()).days
                })
        
        df = pd.DataFrame(data)
        return df


# ==================== 链上数据 ====================

@dataclass
class OnChainMetric(BaseData):
    """链上指标"""
    metric_name: str = ""
    value: float = 0.0
    network: str = "ethereum"
    address: Optional[str] = None
    extra_info: Dict = field(default_factory=dict)
    
    @property
    def is_ethereum(self) -> bool:
        """是否为以太坊网络"""
        return self.network.lower() in ['ethereum', 'eth']
    
    @property
    def is_bitcoin(self) -> bool:
        """是否为比特币网络"""
        return self.network.lower() in ['bitcoin', 'btc']


@dataclass
class TokenFlowData(BaseData):
    """代币资金流数据"""
    token_symbol: str = ""
    token_address: str = ""
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    amount: float = 0.0
    value_usd: float = 0.0
    transaction_hash: Optional[str] = None
    flow_type: str = "transfer"  # transfer, deposit, withdrawal
    is_cex_flow: bool = False  # 是否为交易所资金流
    cex_name: Optional[str] = None  # 交易所名称


@dataclass
class ExchangeFlowData(BaseData):
    """交易所资金流数据"""
    exchange_name: str = ""
    asset: str = ""
    net_flow: float = 0.0  # 净流入流出
    inflow: float = 0.0  # 流入量
    outflow: float = 0.0  # 流出量
    flow_type: str = "spot"  # spot, derivative, etc.
    is_smart_money: bool = False  # 是否为聪明钱
    
    @property
    def net_flow_percent(self) -> float:
        """净流入流出百分比"""
        total_volume = self.inflow + self.outflow
        if total_volume > 0:
            return (self.net_flow / total_volume) * 100
        return 0.0


# ==================== 社交媒体数据 ====================

@dataclass
class SocialSentimentData(BaseData):
    """社交媒体情绪数据"""
    platform: str = ""  # twitter, reddit, telegram, etc.
    keyword: str = ""
    sentiment_score: float = 0.0  # -1 到 1
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_mentions: int = 0
    engagement_rate: Optional[float] = None  # 互动率
    top_influencers: List[Dict] = field(default_factory=list)  # 主要影响者
    
    @property
    def sentiment_percentage(self) -> float:
        """情绪百分比"""
        if self.total_mentions > 0:
            return ((self.positive_count - self.negative_count) / self.total_mentions) * 100
        return 0.0
    
    @property
    def positive_ratio(self) -> float:
        """正面比例"""
        if self.total_mentions > 0:
            return self.positive_count / self.total_mentions
        return 0.0
    
    @property
    def negative_ratio(self) -> float:
        """负面比例"""
        if self.total_mentions > 0:
            return self.negative_count / self.total_mentions
        return 0.0


@dataclass
class SocialSentiment(BaseData):
    """兼容旧版的情绪摘要对象（用于接口返回与聚合）。"""
    platform: str = ""
    overall_sentiment: str = "neutral"  # 'positive'|'negative'|'neutral'
    sentiment_score: float = 0.0
    confidence: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_mentions: int = 0
    analysis_period: str = ""  # 如 '7天'
    
    def to_dict(self) -> Dict:
        """兼容 BaseData 的 to_dict"""
        base = super().to_dict()
        extra = {
            'platform': self.platform,
            'overall_sentiment': self.overall_sentiment,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'neutral_count': self.neutral_count,
            'total_mentions': self.total_mentions,
            'analysis_period': self.analysis_period
        }
        base.update(extra)
        return base


@dataclass
class SocialPostData(BaseData):
    """社交媒体帖子数据"""
    post_id: str = ""
    text: str = ""
    author_id: str = ""
    author_name: Optional[str] = None
    author_followers: Optional[int] = None
    share_count: int = 0  # 转发/分享数
    like_count: int = 0   # 点赞数
    reply_count: int = 0  # 回复/评论数
    quote_count: int = 0  # 引用数
    sentiment_score: Optional[float] = None
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    extra_info: Dict = field(default_factory=dict)  # 额外信息(subreddit等)
    
    @property
    def engagement_rate(self) -> float:
        """互动率"""
        total_engagement = self.share_count + self.like_count + self.reply_count
        if self.author_followers and self.author_followers > 0:
            return (total_engagement / self.author_followers) * 100
        return 0.0

# 兼容旧名称
TweetData = SocialPostData


# ==================== 数据转换工具 ====================

class DataConverter:
    """数据转换工具类"""
    
    @staticmethod
    def ohlcv_list_to_dataframe(ohlcv_list: List[OHLCVData]) -> pd.DataFrame:
        """将OHLCV数据列表转换为DataFrame"""
        if not ohlcv_list:
            return pd.DataFrame()
        
        data = []
        for ohlcv in ohlcv_list:
            data.append({
                'timestamp': ohlcv.timestamp,
                'symbol': ohlcv.symbol,
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume,
                'quote_volume': ohlcv.quote_volume,
                'trades': ohlcv.trades,
                'taker_buy_base_volume': getattr(ohlcv, 'taker_buy_base_volume', 0.0),
                'taker_buy_quote_volume': getattr(ohlcv, 'taker_buy_quote_volume', 0.0),
                'vwap': getattr(ohlcv, 'vwap', None),
                'rsi': getattr(ohlcv, 'rsi', None),
                'sma_20': getattr(ohlcv, 'sma_20', None),
                'sma_50': getattr(ohlcv, 'sma_50', None),
                'sma_200': getattr(ohlcv, 'sma_200', None),
                'ema_12': getattr(ohlcv, 'ema_12', None),
                'ema_26': getattr(ohlcv, 'ema_26', None),
                'macd': getattr(ohlcv, 'macd', None),
                'macd_signal': getattr(ohlcv, 'macd_signal', None),
                'macd_histogram': getattr(ohlcv, 'macd_histogram', None),
                'bollinger_upper': getattr(ohlcv, 'bollinger_upper', None),
                'bollinger_middle': getattr(ohlcv, 'bollinger_middle', None),
                'bollinger_lower': getattr(ohlcv, 'bollinger_lower', None),
                'timeframe': ohlcv.timeframe,
                'exchange': ohlcv.exchange,
                'market_type': ohlcv.market_type
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @staticmethod
    def trades_list_to_dataframe(trades_list: List[TradeData]) -> pd.DataFrame:
        """将成交数据列表转换为DataFrame"""
        if not trades_list:
            return pd.DataFrame()
        
        data = []
        for trade in trades_list:
            data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'trade_id': trade.trade_id,
                'price': trade.price,
                'amount': trade.amount,
                'side': trade.side,
                'taker': trade.taker,
                'order_type': trade.order_type,
                'exchange': trade.exchange
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @staticmethod
    def funding_rates_list_to_dataframe(funding_rates: List[FundingRateData]) -> pd.DataFrame:
        """将资金费率列表转换为DataFrame"""
        if not funding_rates:
            return pd.DataFrame()
        
        data = []
        for fr in funding_rates:
            data.append({
                'timestamp': fr.timestamp,
                'funding_time': fr.funding_time,
                'symbol': fr.symbol,
                'funding_rate': fr.funding_rate,
                'predicted_rate': fr.predicted_rate,
                'cumulative_rate': fr.cumulative_rate,
                'exchange': fr.exchange
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @staticmethod
    def format_symbol_for_market(symbol: str, market_type: str, exchange: str = "binance") -> str:
        """根据市场类型格式化交易对符号"""
        if market_type == MarketType.SPOT.value:
            return symbol
        elif market_type == MarketType.SWAP.value:
            if exchange == "binance":
                # 币安永续合约格式: BTC/USDT:USDT
                base, quote = symbol.split('/')
                return f"{base}/{quote}:{quote}"
            elif exchange == "okx":
                # OKX永续合约格式: BTC-USDT-SWAP
                return symbol.replace('/', '-') + '-SWAP'
        elif market_type == MarketType.FUTURE.value:
            if exchange == "binance":
                # 币安期货格式: BTC/USD:BTC
                base, quote = symbol.split('/')
                return f"{base}/{quote}:{base}"
        elif market_type == MarketType.OPTION.value:
            # 期权格式特殊，需要更多信息
            return symbol
        
        return symbol
    
    @staticmethod
    def parse_symbol(symbol: str, exchange: str = "binance") -> Dict[str, str]:
        """解析交易对符号"""
        result = {
            'original': symbol,
            'base': '',
            'quote': '',
            'settle': '',
            'market_type': 'spot'
        }
        
        try:
            if exchange == "binance":
                if ':' in symbol:
                    # 合约格式: BTC/USDT:USDT
                    symbol_part, settle = symbol.split(':')
                    base, quote = symbol_part.split('/')
                    result['base'] = base
                    result['quote'] = quote
                    result['settle'] = settle
                    
                    if settle == quote:
                        result['market_type'] = 'swap'  # USDT永续
                    elif settle == base:
                        result['market_type'] = 'future'  # 币本位合约
                else:
                    # 现货格式: BTC/USDT
                    base, quote = symbol.split('/')
                    result['base'] = base
                    result['quote'] = quote
                    result['market_type'] = 'spot'
            
            elif exchange == "okx":
                if '-SWAP' in symbol:
                    # OKX永续合约: BTC-USDT-SWAP
                    symbol_clean = symbol.replace('-SWAP', '')
                    base, quote = symbol_clean.split('-')
                    result['base'] = base
                    result['quote'] = quote
                    result['market_type'] = 'swap'
                elif '-' in symbol:
                    # OKX现货: BTC-USDT
                    base, quote = symbol.split('-')
                    result['base'] = base
                    result['quote'] = quote
                    result['market_type'] = 'spot'
        
        except:
            # 解析失败，返回原始值
            pass
        
        return result


# ==================== 数据质量检查 ====================

class DataQualityChecker:
    """数据质量检查器"""
    
    @staticmethod
    def check_ohlcv_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """检查OHLCV数据质量"""
        if df.empty:
            return {
                'status': 'empty',
                'issues': ['DataFrame为空'],
                'score': 0.0
            }
        
        issues = []
        score = 100.0
        
        # 1. 检查缺失值
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            issues.append(f"存在 {missing_values} 个缺失值")
            score -= (missing_values / len(df)) * 100
        
        # 2. 检查时间连续性
        if len(df) > 1:
            time_diff = df.index.to_series().diff().dropna()
            if not time_diff.nunique() == 1:
                issues.append("时间间隔不一致")
                score -= 10
        
        # 3. 检查价格合理性
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                issues.append("存在非正价格")
                score -= 20
            
            # 检查异常价格变动
            returns = df['close'].pct_change().dropna()
            extreme_returns = returns[abs(returns) > 0.5]  # 价格变动超过50%
            if len(extreme_returns) > 0:
                issues.append(f"存在 {len(extreme_returns)} 次极端价格变动")
                score -= len(extreme_returns) * 5
        
        # 4. 检查交易量
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                issues.append("存在负的交易量")
                score -= 15
        
        score = max(0.0, score)  # 确保分数不为负
        
        return {
            'status': 'good' if score >= 80 else 'warning' if score >= 60 else 'bad',
            'issues': issues,
            'score': score,
            'rows': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'days': (df.index.max() - df.index.min()).days
            }
        }
    
    @staticmethod
    def check_orderbook_quality(orderbook: OrderBookData) -> Dict[str, Any]:
        """检查订单簿数据质量"""
        issues = []
        score = 100.0
        
        # 检查买卖盘
        if not orderbook.bids:
            issues.append("买盘为空")
            score -= 30
        
        if not orderbook.asks:
            issues.append("卖盘为空")
            score -= 30
        
        if orderbook.bids and orderbook.asks:
            # 检查价差合理性
            best_bid = orderbook.bids[0][0]
            best_ask = orderbook.asks[0][0]
            spread_percent = (orderbook.spread / ((best_bid + best_ask) / 2)) * 100
            
            if spread_percent > 1.0:  # 价差超过1%
                issues.append(f"价差过大: {spread_percent:.2f}%")
                score -= 20
        
        return {
            'status': 'good' if score >= 80 else 'warning' if score >= 60 else 'bad',
            'issues': issues,
            'score': score,
            'bid_levels': len(orderbook.bids),
            'ask_levels': len(orderbook.asks),
            'spread': orderbook.spread
        }


# ==================== 数据工厂 ====================

class DataFactory:
    """数据工厂，用于创建数据对象"""
    
    @staticmethod
    def create_ohlcv_from_ccxt(ohlcv_list: List, symbol: str, timeframe: str, 
                               exchange: str = "binance", market_type: str = "spot") -> List[OHLCVData]:
        """从CCXT OHLCV数据创建OHLCVData对象列表"""
        result = []
        
        for ohlcv in ohlcv_list:
            try:
                data = OHLCVData.from_ccxt(ohlcv, symbol, timeframe, exchange, market_type)
                result.append(data)
            except Exception as e:
                print(f"创建OHLCVData失败: {e}")
                continue
        
        return result
    
    @staticmethod
    def create_orderbook_from_ccxt(orderbook: Dict, symbol: str, 
                                   exchange: str = "binance") -> OrderBookData:
        """从CCXT订单簿数据创建OrderBookData对象"""
        try:
            return OrderBookData.from_ccxt(orderbook, symbol, exchange)
        except Exception as e:
            print(f"创建OrderBookData失败: {e}")
            return None
    
    @staticmethod
    def create_trades_from_ccxt(trades_list: List[Dict], symbol: str, 
                                exchange: str = "binance") -> List[TradeData]:
        """从CCXT成交数据创建TradeData对象列表"""
        result = []
        
        for trade in trades_list:
            try:
                data = TradeData.from_ccxt(trade, symbol, exchange)
                result.append(data)
            except Exception as e:
                print(f"创建TradeData失败: {e}")
                continue
        
        return result


# ==================== 测试函数 ====================

def test_data_models():
    """测试数据模型"""
    print("=" * 60)
    print("数据模型模块测试")
    print("=" * 60)
    
    # 测试OHLCVData
    print("\n1. 测试OHLCVData:")
    ohlcv = OHLCVData(
        timestamp=pd.Timestamp.now(),
        symbol="BTC/USDT",
        exchange="binance",
        market_type="spot",
        timeframe="1h",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=1000.0
    )
    print(f"OHLCVData: {ohlcv.symbol} 价格: {ohlcv.close}")
    print(f"DataFrame形状: {ohlcv.to_dataframe().shape}")
    
    # 测试OrderBookData
    print("\n2. 测试OrderBookData:")
    orderbook = OrderBookData(
        timestamp=pd.Timestamp.now(),
        symbol="BTC/USDT",
        exchange="binance",
        bids=[(50400, 1.5), (50300, 2.0)],
        asks=[(50600, 1.0), (50700, 1.5)],
        spread=200.0
    )
    print(f"OrderBookData: 买一价 {orderbook.bids[0][0]}, 卖一价 {orderbook.asks[0][0]}")
    print(f"订单簿摘要: {orderbook.get_summary()}")
    
    # 测试数据转换
    print("\n3. 测试DataConverter:")
    symbol = DataConverter.format_symbol_for_market("BTC/USDT", "swap", "binance")
    print(f"交易对格式化 (swap): {symbol}")
    
    parsed = DataConverter.parse_symbol("BTC/USDT:USDT", "binance")
    print(f"交易对解析: {parsed}")
    
    # 测试数据质量检查
    print("\n4. 测试DataQualityChecker:")
    ohlcv_list = [ohlcv, ohlcv]
    df = DataConverter.ohlcv_list_to_dataframe(ohlcv_list)
    quality = DataQualityChecker.check_ohlcv_quality(df)
    print(f"数据质量检查: {quality}")
    
    print("\n✅ 数据模型模块测试完成")


if __name__ == "__main__":
    test_data_models()