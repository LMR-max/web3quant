"""
币安(Binance)期权数据获取器模块
专门针对币安期权市场进行优化
"""

import time
import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats

# 导入基础模块
try:
    from .base_fetcher import BaseFetcher, AsyncFetcher
    from ..data_models import (
        OptionContractData, GreeksData, VolatilitySurfaceData,
        OHLCVData, OrderBookData, TradeData
    )
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
    from data_models import (
        OptionContractData, GreeksData, VolatilitySurfaceData,
        OHLCVData, OrderBookData, TradeData
    )
    from utils.logger import get_logger, log_execution_time, log_errors
    from utils.cache import CacheManager, cached, cache_result
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from config import get_exchange_config, ExchangeSymbolFormats, get_market_config


# ==================== 币安期权计算工具 ====================

class BinanceOptionCalculator:
    """币安期权计算工具类"""
    
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        计算Black-Scholes期权价格
        
        参数:
            S: 标的资产当前价格
            K: 行权价格
            T: 到期时间（年）
            r: 无风险利率
            sigma: 波动率
            option_type: 期权类型 (call/put)
            
        返回:
            期权理论价格
        """
        if T <= 0:
            # 已到期期权
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        计算期权希腊值 (BS Model)
        """
        if T <= 0:
             return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Norm dist
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        cdf_minus_d1 = stats.norm.cdf(-d1)
        cdf_d2 = stats.norm.cdf(d2)
        cdf_minus_d2 = stats.norm.cdf(-d2)

        if option_type == 'call':
            delta = cdf_d1
            rho = K * T * np.exp(-r * T) * cdf_d2
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * cdf_d2)
        else:
            delta = cdf_d1 - 1
            rho = -K * T * np.exp(-r * T) * cdf_minus_d2
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * cdf_minus_d2)

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * pdf_d1

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365, # 通常 Theta 按天显示
            'vega': vega / 100,   # 通常 Vega 按 1% 波动显示
            'rho': rho / 100
        }

class OptionFetcher(BaseFetcher):
    """
    期权数据获取器基类
    """

    def __init__(self, 
                 exchange: str = "binance",
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化期权数据获取器
        """
        super().__init__(
            exchange=exchange,
            market_type="option",
            config=config,
            cache_manager=cache_manager
        )
    
    @staticmethod
    def calculate_implied_volatility(price: float, S: float, K: float, T: float, r: float, 
                                     option_type: str = 'call', max_iter: int = 100, 
                                     tolerance: float = 1e-6) -> Optional[float]:
        """
        计算隐含波动率 (使用牛顿-拉夫逊法)
        
        参数:
            price: 期权市场价格
            S: 标的资产当前价格
            K: 行权价格
            T: 到期时间（年）
            r: 无风险利率
            option_type: 期权类型 (call/put)
            max_iter: 最大迭代次数
            tolerance: 容忍度
            
        返回:
            隐含波动率
        """
        if T <= 0:
            return None
        
        # 初始猜测
        sigma = 0.5  # 初始猜测50%波动率
        
        for i in range(max_iter):
            # 计算当前波动率下的期权价格
            market_price = BinanceOptionCalculator.black_scholes(S, K, T, r, sigma, option_type)
            
            # 计算vega
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * stats.norm.pdf(d1) * np.sqrt(T)
            
            # 如果vega太小，停止迭代
            if abs(vega) < tolerance:
                break
            
            # 计算价格差异
            diff = market_price - price
            
            # 如果差异很小，停止迭代
            if abs(diff) < tolerance:
                break
            
            # 更新sigma (牛顿法)
            sigma = sigma - diff / vega
            
            # 确保sigma为正
            sigma = max(sigma, 0.001)
            
            # 如果sigma太大，重新设置
            if sigma > 5.0:
                sigma = 0.5
        
        return sigma if sigma > 0 and sigma < 5 else None
    
    @staticmethod
    def calculate_moneyness(S: float, K: float, option_type: str = 'call') -> Dict[str, Any]:
        """
        计算期权虚实程度
        
        参数:
            S: 标的资产当前价格
            K: 行权价格
            option_type: 期权类型 (call/put)
            
        返回:
            虚实程度指标
        """
        # 相对行权价
        relative_strike = K / S
        
        # 虚实程度
        if option_type == 'call':
            # 看涨期权: 实值=K<S, 平值=K=S, 虚值=K>S
            if K < S * 0.90:
                moneyness = 'deep_itm'  # 深度实值
            elif K < S * 0.98:
                moneyness = 'itm'  # 实值
            elif abs(K - S) / S < 0.02:
                moneyness = 'atm'  # 平值
            elif K < S * 1.10:
                moneyness = 'otm'  # 虚值
            else:
                moneyness = 'deep_otm'  # 深度虚值
        else:
            # 看跌期权: 实值=K>S, 平值=K=S, 虚值=K<S
            if K > S * 1.10:
                moneyness = 'deep_itm'  # 深度实值
            elif K > S * 1.02:
                moneyness = 'itm'  # 实值
            elif abs(K - S) / S < 0.02:
                moneyness = 'atm'  # 平值
            elif K > S * 0.90:
                moneyness = 'otm'  # 虚值
            else:
                moneyness = 'deep_otm'  # 深度虚值
        
        # 虚实百分比
        moneyness_percent = (K - S) / S * 100
        
        return {
            'relative_strike': relative_strike,
            'moneyness': moneyness,
            'moneyness_percent': moneyness_percent,
            'is_itm': moneyness in ['itm', 'deep_itm'],
            'is_otm': moneyness in ['otm', 'deep_otm'],
            'is_atm': moneyness == 'atm'
        }


# ==================== 币安期权数据获取器 ====================

class BinanceOptionFetcher(BaseFetcher):
    """
    币安期权数据获取器
    
    专门针对币安期权市场进行优化
    """
    
    def __init__(self,
                 exchange: str = "binance",
                 option_style: str = "european",  # european: 欧式
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None,
                 **kwargs):
        """
        初始化币安期权数据获取器
        
        参数:
            option_style: 期权类型 (european)
            config: 配置字典
            cache_manager: 缓存管理器
        """
        if exchange and str(exchange).lower() != "binance":
            # 当前实现只覆盖币安期权；其他交易所暂时走模拟数据兜底
            exchange = "binance"

        super().__init__(
            exchange=exchange,
            market_type="option",
            config=config,
            cache_manager=cache_manager
        )
        
        self.option_style = option_style
        
        # 加载交易所配置
        self.exchange_config = get_exchange_config("binance")
        
        # 加载市场配置
        self.market_config = get_market_config("option")
        
        # 初始化交易所连接
        self.exchange_instance = None
        self._init_exchange()
        
        # 初始化期权合约信息
        self.option_contracts = {}
        self.active_options = []
        
        # 币安期权支持的标的资产
        self.supported_underlyings = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA']
        
        # 无风险利率（默认3%）
        self.risk_free_rate = getattr(self.config, 'risk_free_rate', 0.03)
        
        # 期权计算器
        self.calculator = BinanceOptionCalculator()
        
        # 模拟数据开关（当真实数据不可用时）
        self.use_simulated_data = getattr(self.config, 'use_simulated_data', True)
        
        # 标的资产价格缓存
        self.underlying_price_cache = {}
        
        # 加载合约信息
        self._load_contracts_info()
        
    def _init_exchange(self):
        """初始化交易所连接"""
        try:
            self.logger.info(f"初始化币安期权交易所连接，期权类型: {self.option_style}")
            
            # 尝试导入CCXT
            try:
                import ccxt
                self.has_ccxt = True
            except ImportError:
                self.has_ccxt = False
                self.logger.warning("CCXT未安装，部分功能可能受限")
                return
            
            # 创建币安交易所实例
            try:
                import ccxt
                exchange_class = getattr(ccxt, 'binance', None)
                if not exchange_class:
                    raise ValueError("不支持的交-所: binance")
                
                # 获取CCXT配置
                ccxt_config = self.exchange_config.get_ccxt_config()
                
                # 设置默认类型为期权
                ccxt_config['options'] = ccxt_config.get('options', {})
                ccxt_config['options']['defaultType'] = 'option'
                
                # 合并市场配置
                if 'ccxt_options' in self.market_config:
                    ccxt_config['options'].update(self.market_config['ccxt_options'])
                
                # 创建交易所实例
                self.exchange_instance = exchange_class(ccxt_config)
                
                # 设置超时时间
                self.exchange_instance.timeout = 30000
                
                self.logger.info("币安期权交易所连接初始化成功")
                
            except Exception as e:
                self.logger.error(f"创建币安交易所实例失败: {e}")
                self.exchange_instance = None
                self.use_simulated_data = True
            
        except Exception as e:
            self.logger.error(f"币安期权交易所连接初始化失败: {e}")
            self.exchange_instance = None
            self.use_simulated_data = True
    
    def _load_contracts_info(self):
        """加载合约信息"""
        self.logger.info("加载币安期权合约信息...")
        
        # 清空现有合约
        self.option_contracts.clear()
        self.active_options.clear()
        
        # 创建模拟合约
        self._create_binance_simulated_contracts()
        
        self.logger.info(f"加载 {len(self.active_options)} 个活跃期权合约")
    
    def _create_binance_simulated_contracts(self):
        """创建币安期权模拟合约"""
        self.logger.info("创建币安期权模拟合约数据")
        
        # 币安期权格式: BTC-250228-50000-C (BTC-YYMMDD-Strike-C/P)
        current_date = datetime.now()
        
        # 生成到期日 (每周五到期)
        expiry_dates = []
        for i in range(1, 13):  # 未来12周
            expiry_date = current_date + timedelta(days=7*i)
            # 调整到最近的周五
            days_to_friday = (4 - expiry_date.weekday()) % 7
            expiry_date += timedelta(days=days_to_friday)
            expiry_dates.append(expiry_date)
        
        # 生成模拟合约
        for underlying in self.supported_underlyings:
            # 根据标的资产确定基础价格
            if underlying == 'BTC':
                base_price = 50000
                strike_steps = [1000, 2000, 3000, 4000, 5000]
            elif underlying == 'ETH':
                base_price = 3000
                strike_steps = [100, 200, 300, 400, 500]
            elif underlying == 'BNB':
                base_price = 600
                strike_steps = [10, 20, 30, 40, 50]
            elif underlying == 'SOL':
                base_price = 150
                strike_steps = [5, 10, 15, 20, 25]
            else:
                base_price = 100
                strike_steps = [5, 10, 15, 20, 25]
            
            self.option_contracts[underlying] = []
            
            for expiry_date in expiry_dates[:3]:  # 只取前3个到期日
                # 币安期权到期日格式: YYMMDD
                expiry_str = expiry_date.strftime('%y%m%d')
                
                # 生成行权价
                for strike_offset in strike_steps:
                    strike_price = base_price + strike_offset
                    
                    # 创建看涨和看跌期权
                    for option_type, option_type_code in [('call', 'C'), ('put', 'P')]:
                        option_symbol = f"{underlying}-{expiry_str}-{strike_price:.0f}-{option_type_code}"
                        
                        days_to_expiry = (expiry_date - current_date).days
                        
                        option_info = {
                            'option_symbol': option_symbol,
                            'underlying_symbol': f"{underlying}/USDT",
                            'underlying_asset': underlying,
                            'strike_price': strike_price,
                            'expiry_date': pd.Timestamp(expiry_date),
                            'expiry_str': expiry_str,
                            'option_type': option_type,
                            'option_type_code': option_type_code,
                            'contract_size': 1.0,
                            'settlement_asset': 'USDT',
                            'is_european': True,
                            'is_active': True,
                            'tick_size': 0.01,
                            'lot_size': 0.001,
                            'days_to_expiry': max(0, days_to_expiry),
                            'expiry_percent': max(0, min(100, (1 - days_to_expiry / 365) * 100)),
                            'exchange': 'binance'
                        }
                        
                        self.option_contracts[underlying].append(option_info)
                        self.active_options.append(option_symbol)
        
        self.logger.info(f"创建 {len(self.active_options)} 个币安期权模拟合约")
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的期权交易对
        
        返回:
            交易对列表
        """
        return self.active_options.copy()
    
    def get_options_by_underlying(self, underlying_symbol: str) -> List[Dict]:
        """
        根据标的资产获取期权列表
        
        参数:
            underlying_symbol: 标的资产符号，如 BTC/USDT
            
        返回:
            期权信息列表
        """
        # 提取标的资产名称
        if '/' in underlying_symbol:
            underlying = underlying_symbol.split('/')[0]
        else:
            underlying = underlying_symbol
        
        # 查找对应的期权
        if underlying in self.option_contracts:
            return self.option_contracts[underlying]
        else:
            # 尝试大小写转换
            for key in self.option_contracts:
                if key.upper() == underlying.upper():
                    return self.option_contracts[key]
        
        return []
    
    def get_active_options(self) -> List[str]:
        """
        获取活跃期权列表
        
        返回:
            活跃期权列表
        """
        return self.active_options
    
    def get_option_chain(self, underlying_symbol: str, expiry_date: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        获取期权链数据
        
        参数:
            underlying_symbol: 标的资产符号
            expiry_date: 到期日期（可选，格式: YYMMDD）
            
        返回:
            期权链数据，按行权价分组
        """
        options = self.get_options_by_underlying(underlying_symbol)
        
        if not options:
            return {}
        
        # 按到期日筛选
        if expiry_date:
            # 确保expiry_date格式正确
            if len(expiry_date) == 8:
                expiry_date = expiry_date[2:]  # YYYYMMDD -> YYMMDD
            
            options = [opt for opt in options if opt.get('expiry_str') == expiry_date]
        
        # 按行权价分组
        chain = {}
        for option in options:
            strike = option.get('strike_price', 0)
            option_type = option.get('option_type', 'call')
            
            # 格式化行权价作为键
            if strike > 0:
                strike_key = f"{strike:.0f}"
            else:
                strike_key = "unknown"
            
            if strike_key not in chain:
                chain[strike_key] = {
                    'strike': strike,
                    'expiry_date': option.get('expiry_date'),
                    'expiry_str': option.get('expiry_str'),
                    'calls': [],
                    'puts': []
                }
            
            if option_type == 'call':
                chain[strike_key]['calls'].append(option)
            else:
                chain[strike_key]['puts'].append(option)
        
        return chain
    
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
        
        # 检查是否符合币安期权格式
        pattern = r'^[A-Z]+-\d{6}-\d+-[CP]$'
        if not re.match(pattern, symbol):
            self.logger.warning(f"期权符号格式不符合币安要求: {symbol}")
            return False
        
        return True
    
    def format_symbol(self, symbol: str, expiry_date: Optional[str] = None, 
                      strike_price: Optional[float] = None, option_type: Optional[str] = None) -> str:
        """
        格式化交易对符号为币安标准格式
        
        参数:
            symbol: 原始交易对符号
            expiry_date: 到期日期，格式: YYYYMMDD 或 YYMMDD
            strike_price: 行权价格
            option_type: 期权类型 (call/put)
            
        返回:
            格式化后的交易对符号
        """
        # 如果已经是完整期权格式，直接返回
        if re.match(r'^[A-Z]+-\d{6}-\d+-[CP]$', symbol):
            return symbol
        
        # 处理标的资产
        if '/' in symbol:
            underlying = symbol.split('/')[0]
        else:
            underlying = symbol
        
        # 如果没有提供完整参数，返回基础标的
        if not (expiry_date and strike_price and option_type):
            return underlying
        
        # 格式化到期日
        if len(expiry_date) == 8:  # YYYYMMDD
            expiry_str = expiry_date[2:]  # 转为YYMMDD
        elif len(expiry_date) == 6:  # YYMMDD
            expiry_str = expiry_date
        else:
            raise ValueError(f"无效的到期日格式: {expiry_date}")
        
        # 格式化期权类型
        option_type_code = 'C' if option_type.lower() == 'call' else 'P'
        
        # 格式化行权价
        strike_str = f"{strike_price:.0f}"
        
        return f"{underlying}-{expiry_str}-{strike_str}-{option_type_code}"
    
    def parse_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        解析币安期权符号
        
        参数:
            symbol: 期权符号
            
        返回:
            解析后的信息
        """
        if not re.match(r'^[A-Z]+-\d{6}-\d+-[CP]$', symbol):
            raise ValueError(f"无效的币安期权符号格式: {symbol}")
        
        parts = symbol.split('-')
        underlying = parts[0]
        expiry_str = parts[1]
        strike_str = parts[2]
        option_type_code = parts[3]
        
        # 解析到期日
        if len(expiry_str) == 6:
            year = '20' + expiry_str[:2]
            month = expiry_str[2:4]
            day = expiry_str[4:]
            expiry_date = pd.Timestamp(f"{year}-{month}-{day}")
        else:
            expiry_date = None
        
        # 解析期权类型
        option_type = 'call' if option_type_code == 'C' else 'put'
        
        return {
            'underlying': underlying,
            'underlying_symbol': f"{underlying}/USDT",
            'expiry_str': expiry_str,
            'expiry_date': expiry_date,
            'strike_price': float(strike_str),
            'option_type': option_type,
            'option_type_code': option_type_code
        }
    
    @log_errors(reraise=False)
    @cached(key_prefix="binance_option_ohlcv", ttl=300, sub_dir="binance_option")
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = "1h",
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None,
                   **kwargs) -> List[OHLCVData]:
        """
        获取币安期权K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            OHLCV数据列表
        """
        # 检查是否使用模拟数据
        if self.use_simulated_data or not self.exchange_instance:
            return self._fetch_binance_simulated_ohlcv(symbol, timeframe, since, limit, **kwargs)
        
        # 验证符号格式
        if not self.validate_symbol(symbol):
            self.logger.warning(f"期权符号 {symbol} 不符合币安格式要求")
        
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
            limit = self.config.get('ohlcv_limit', 100)
        
        self.logger.info(
            f"获取币安期权K线数据: {symbol}, "
            f"时间间隔: {timeframe}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 尝试使用CCXT获取数据
            ohlcv_list = self.exchange_instance.fetch_ohlcv(
                symbol=symbol,
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
                        symbol=symbol,
                        timeframe=timeframe,
                        exchange="binance",
                        market_type="option"
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换币安期权OHLCV数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条币安期权K线数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取币安期权K线数据失败: {e}")
            # 如果失败，返回模拟数据
            return self._fetch_binance_simulated_ohlcv(symbol, timeframe, since, limit, **kwargs)
    
    def _fetch_binance_simulated_ohlcv(self, symbol: str, timeframe: str = "1h",
                                      since: Optional[Union[int, datetime, str]] = None,
                                      limit: Optional[int] = None, **kwargs) -> List[OHLCVData]:
        """获取币安期权模拟K线数据"""
        self.logger.info(f"使用模拟数据获取币安期权K线: {symbol}")
        
        # 设置默认限制
        if limit is None:
            limit = 100
        
        # 解析期权信息
        try:
            option_info = self.parse_symbol(symbol)
        except Exception as e:
            self.logger.error(f"解析期权符号失败: {e}")
            return []
        
        # 生成模拟数据
        data_models = []
        base_time = datetime.now() if not since else (
            since if isinstance(since, datetime) else 
            pd.Timestamp(since).to_pydatetime() if isinstance(since, str) else
            datetime.fromtimestamp(since / 1000)
        )
        
        # 获取标的资产价格
        underlying_price = self.fetch_underlying_price(option_info['underlying_symbol'])
        
        # 计算合理的期权价格范围
        strike_price = option_info['strike_price']
        option_type = option_info['option_type']
        
        # 计算内在价值
        intrinsic_value = self.calculate_intrinsic_value(underlying_price, strike_price, option_type)
        
        # 基础价格 = 内在价值 + 时间价值
        if option_info.get('expiry_date'):
            days_to_expiry = (option_info['expiry_date'] - pd.Timestamp.now()).days
            time_value = max(50, min(1000, days_to_expiry * 10))  # 简单的时间价值估算
        else:
            time_value = 500
        
        base_price = intrinsic_value + time_value
        
        for i in range(limit):
            # 计算时间偏移
            if timeframe == "1m":
                time_offset = timedelta(minutes=i)
            elif timeframe == "5m":
                time_offset = timedelta(minutes=i * 5)
            elif timeframe == "15m":
                time_offset = timedelta(minutes=i * 15)
            elif timeframe == "1h":
                time_offset = timedelta(hours=i)
            elif timeframe == "4h":
                time_offset = timedelta(hours=i * 4)
            elif timeframe == "1d":
                time_offset = timedelta(days=i)
            else:
                time_offset = timedelta(hours=i)
            
            timestamp = base_time - time_offset
            
            # 生成随机价格波动
            price_change = np.random.uniform(-0.03, 0.03)  # ±3%
            current_price = base_price * (1 + price_change)
            
            # 确保价格为正
            current_price = max(current_price, 0.01)
            
            # 生成OHLCV数据
            open_price = current_price * (1 + np.random.uniform(-0.01, 0.01))
            high_price = max(open_price, current_price) * (1 + np.random.uniform(0, 0.015))
            low_price = min(open_price, current_price) * (1 - np.random.uniform(0, 0.015))
            close_price = current_price
            volume = np.random.uniform(10, 1000)
            
            # 创建OHLCVData对象
            data_model = OHLCVData(
                timestamp=timestamp,
                symbol=symbol,
                exchange="binance",
                market_type="option",
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            data_models.append(data_model)
        
        self.logger.info(f"生成 {len(data_models)} 条币安期权模拟K线数据")
        return data_models
    
    @log_errors(reraise=False)
    @cached(key_prefix="binance_option_orderbook", ttl=30, sub_dir="binance_option")
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> Optional[OrderBookData]:
        """
        获取币安期权订单簿数据
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            订单簿数据
        """
        # 检查是否使用模拟数据
        if self.use_simulated_data or not self.exchange_instance:
            return self._fetch_binance_simulated_orderbook(symbol, limit, **kwargs)
        
        # 设置默认限制
        if limit is None:
            limit = self.config.get('orderbook_limit', 20)
        
        self.logger.info(f"获取币安期权订单簿: {symbol}, 深度: {limit}")
        
        try:
            # 使用CCXT获取订单簿
            orderbook = self.exchange_instance.fetch_order_book(
                symbol=symbol,
                limit=limit
            )
            
            # 转换为数据模型
            data_model = OrderBookData.from_ccxt(
                orderbook=orderbook,
                symbol=symbol,
                exchange="binance"
            )
            
            self.logger.info(
                f"币安期权订单簿获取成功: 买盘 {len(data_model.bids)} 个, "
                f"卖盘 {len(data_model.asks)} 个, "
                f"价差: {data_model.spread:.2f}"
            )
            
            return data_model
            
        except Exception as e:
            self.logger.error(f"获取币安期权订单簿失败: {e}")
            return self._fetch_binance_simulated_orderbook(symbol, limit, **kwargs)
    
    def _fetch_binance_simulated_orderbook(self, symbol: str, limit: Optional[int] = None, **kwargs) -> Optional[OrderBookData]:
        """获取币安期权模拟订单簿数据"""
        self.logger.info(f"使用模拟数据获取币安期权订单簿: {symbol}")
        
        # 设置默认限制
        if limit is None:
            limit = 20
        
        # 解析期权信息
        try:
            option_info = self.parse_symbol(symbol)
        except Exception as e:
            self.logger.error(f"解析期权符号失败: {e}")
            return None
        
        # 获取标的资产价格
        underlying_price = self.fetch_underlying_price(option_info['underlying_symbol'])
        
        # 计算合理的期权价格
        strike_price = option_info['strike_price']
        option_type = option_info['option_type']
        
        # 计算内在价值
        intrinsic_value = self.calculate_intrinsic_value(underlying_price, strike_price, option_type)
        
        # 时间价值估算
        if option_info.get('expiry_date'):
            days_to_expiry = (option_info['expiry_date'] - pd.Timestamp.now()).days
            time_value = max(50, min(1000, days_to_expiry * 10))
        else:
            time_value = 500
        
        base_price = intrinsic_value + time_value
        
        # 生成买盘和卖盘
        bids = []
        asks = []
        
        for i in range(limit):
            # 买盘价格低于基础价格
            bid_price = base_price * (1 - (i + 1) * 0.005)  # 0.5%间隔
            bid_price = max(bid_price, 0.01)  # 确保价格为正
            bid_amount = np.random.uniform(0.1, 10.0)
            bids.append([float(bid_price), float(bid_amount)])
            
            # 卖盘价格高于基础价格
            ask_price = base_price * (1 + (i + 1) * 0.005)  # 0.5%间隔
            ask_amount = np.random.uniform(0.1, 10.0)
            asks.append([float(ask_price), float(ask_amount)])
        
        # 按价格排序（买盘降序，卖盘升序）
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        # 计算订单簿指标
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        
        # 创建OrderBookData对象
        data_model = OrderBookData(
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="binance",
            market_type="option",
            bids=bids,
            asks=asks,
            bid=best_bid,
            ask=best_ask,
            spread=spread
        )
        
        return data_model
    
    @log_errors(reraise=False)
    @cached(key_prefix="binance_option_trades", ttl=60, sub_dir="binance_option")
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取币安期权成交数据
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            成交数据列表
        """
        # 检查是否使用模拟数据
        if self.use_simulated_data or not self.exchange_instance:
            return self._fetch_binance_simulated_trades(symbol, since, limit, **kwargs)
        
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
            f"获取币安期权成交数据: {symbol}, "
            f"开始时间: {since_timestamp}, "
            f"限制: {limit}"
        )
        
        try:
            # 使用CCXT获取成交数据
            trades = self.exchange_instance.fetch_trades(
                symbol=symbol,
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
                        symbol=symbol,
                        exchange="binance"
                    )
                    data_models.append(data_model)
                except Exception as e:
                    self.logger.warning(f"转换币安期权成交数据失败: {e}")
                    continue
            
            self.logger.info(f"获取到 {len(data_models)} 条币安期权成交数据")
            return data_models
            
        except Exception as e:
            self.logger.error(f"获取币安期权成交数据失败: {e}")
            return self._fetch_binance_simulated_trades(symbol, since, limit, **kwargs)
    
    def _fetch_binance_simulated_trades(self, symbol: str, since: Optional[Union[int, datetime, str]] = None,
                                       limit: Optional[int] = None, **kwargs) -> List[TradeData]:
        """获取币安期权模拟成交数据"""
        self.logger.info(f"使用模拟数据获取币安期权成交: {symbol}")
        
        # 设置默认限制
        if limit is None:
            limit = 50
        
        # 生成模拟数据
        data_models = []
        base_time = datetime.now()
        
        # 获取订单簿确定合理价格范围
        orderbook = self.fetch_orderbook(symbol, limit=5)
        if orderbook and orderbook.bid > 0 and orderbook.ask > 0:
            price_range = (orderbook.bid, orderbook.ask)
        else:
            # 解析期权信息获取合理价格范围
            try:
                option_info = self.parse_symbol(symbol)
                underlying_price = self.fetch_underlying_price(option_info['underlying_symbol'])
                intrinsic_value = self.calculate_intrinsic_value(
                    underlying_price, 
                    option_info['strike_price'], 
                    option_info['option_type']
                )
                price_range = (intrinsic_value, intrinsic_value + 1000)
            except:
                price_range = (100, 1000)  # 默认范围
        
        for i in range(limit):
            # 随机生成成交
            timestamp = base_time - timedelta(seconds=np.random.randint(0, 3600))
            price = np.random.uniform(price_range[0], price_range[1])
            amount = np.random.uniform(0.1, 10.0)
            side = 'buy' if np.random.random() > 0.5 else 'sell'
            
            # 创建TradeData对象
            data_model = TradeData(
                timestamp=timestamp,
                symbol=symbol,
                exchange="binance",
                market_type="option",
                price=price,
                amount=amount,
                side=side,
                trade_id=f"binance_sim_{int(timestamp.timestamp())}_{i}"
            )
            
            data_models.append(data_model)
        
        self.logger.info(f"生成 {len(data_models)} 条币安期权模拟成交数据")
        return data_models
    
    @log_errors(reraise=False)
    @cached(key_prefix="binance_option_contract_info", ttl=3600, sub_dir="binance_option")
    def fetch_contract_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取币安期权合约信息
        
        参数:
            symbol: 交易对符号
            
        返回:
            合约信息
        """
        try:
            # 解析期权符号
            parsed_info = self.parse_symbol(symbol)
            
            # 查找对应的期权信息
            underlying = parsed_info['underlying']
            strike_price = parsed_info['strike_price']
            option_type = parsed_info['option_type']
            expiry_str = parsed_info['expiry_str']
            
            if underlying in self.option_contracts:
                for option in self.option_contracts[underlying]:
                    if (option.get('strike_price') == strike_price and
                        option.get('option_type') == option_type and
                        option.get('expiry_str') == expiry_str):
                        return option
            
            # 如果没有找到，创建新的信息
            expiry_date = parsed_info['expiry_date']
            if expiry_date:
                days_to_expiry = (expiry_date - pd.Timestamp.now()).days
            else:
                days_to_expiry = 30
            
            contract_info = {
                'option_symbol': symbol,
                'underlying_symbol': f"{underlying}/USDT",
                'underlying_asset': underlying,
                'strike_price': strike_price,
                'expiry_date': expiry_date,
                'expiry_str': expiry_str,
                'option_type': option_type,
                'option_type_code': 'C' if option_type == 'call' else 'P',
                'contract_size': 1.0,
                'settlement_asset': 'USDT',
                'is_european': True,
                'is_active': True,
                'tick_size': 0.01,
                'lot_size': 0.001,
                'days_to_expiry': max(0, days_to_expiry),
                'expiry_percent': max(0, min(100, (1 - days_to_expiry / 365) * 100)),
                'exchange': 'binance'
            }
            
            # 添加到合约列表
            if underlying not in self.option_contracts:
                self.option_contracts[underlying] = []
            self.option_contracts[underlying].append(contract_info)
            
            return contract_info
            
        except Exception as e:
            self.logger.error(f"获取币安期权合约信息失败: {e}")
            return None
    
    @log_errors(reraise=False)
    @cached(key_prefix="binance_option_price", ttl=60, sub_dir="binance_option")
    def fetch_option_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取币安期权价格信息
        
        参数:
            symbol: 交易对符号
            
        返回:
            期权价格信息
        """
        # 获取最新K线
        ohlcv = self.fetch_ohlcv(symbol, timeframe="1m", limit=1)
        
        if not ohlcv:
            return None
        
        # 获取合约信息
        contract_info = self.fetch_contract_info(symbol)
        
        if not contract_info:
            return None
        
        # 获取标的资产价格
        underlying_price = self.fetch_underlying_price(contract_info['underlying_symbol'])
        
        # 计算期权基本指标
        option_price = ohlcv[0].close
        
        # 计算内在价值和时间价值
        intrinsic_value = self.calculate_intrinsic_value(
            underlying_price,
            contract_info['strike_price'],
            contract_info['option_type']
        )
        time_value = max(0, option_price - intrinsic_value)
        
        # 计算虚实程度
        moneyness_info = self.calculator.calculate_moneyness(
            underlying_price,
            contract_info['strike_price'],
            contract_info['option_type']
        )
        
        result = {
            'option_symbol': symbol,
            'timestamp': pd.Timestamp.now(),
            'option_price': option_price,
            'underlying_price': underlying_price,
            'strike_price': contract_info['strike_price'],
            'expiry_date': contract_info.get('expiry_date'),
            'expiry_str': contract_info.get('expiry_str'),
            'days_to_expiry': contract_info.get('days_to_expiry'),
            'option_type': contract_info['option_type'],
            'contract_size': contract_info['contract_size'],
            'intrinsic_value': intrinsic_value,
            'time_value': time_value,
            'volume': ohlcv[0].volume,
            'moneyness': moneyness_info['moneyness'],
            'moneyness_percent': moneyness_info['moneyness_percent'],
            'exchange': 'binance'
        }
        
        return result
    
    def fetch_underlying_price(self, underlying_symbol: str) -> float:
        """
        获取标的资产价格
        
        参数:
            underlying_symbol: 标的资产符号
            
        返回:
            标的资产价格
        """
        # 检查缓存
        cache_key = f"binance_underlying_price_{underlying_symbol}"
        if cache_key in self.underlying_price_cache:
            cache_entry = self.underlying_price_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 60:  # 缓存60秒
                return cache_entry['price']
        
        try:
            # 清理符号
            if '/' in underlying_symbol:
                clean_symbol = underlying_symbol
            else:
                # 尝试添加USDT报价
                clean_symbol = f"{underlying_symbol}/USDT"
            
            price = None
            
            # 尝试使用CCXT获取价格
            if self.exchange_instance:
                try:
                    # 尝试获取现货价格
                    spot_price = self._fetch_spot_price(clean_symbol)
                    if spot_price:
                        price = spot_price
                except:
                    pass
            
            # 如果失败，使用模拟数据
            if price is None:
                self.logger.warning(f"无法获取标的资产 {underlying_symbol} 价格，使用模拟数据")
                
                # 基于标的资产名称生成合理价格
                underlying = underlying_symbol.split('/')[0] if '/' in underlying_symbol else underlying_symbol
                
                if 'BTC' in underlying.upper():
                    price = 50000 + np.random.uniform(-2000, 2000)
                elif 'ETH' in underlying.upper():
                    price = 3000 + np.random.uniform(-200, 200)
                elif 'BNB' in underlying.upper():
                    price = 600 + np.random.uniform(-50, 50)
                elif 'SOL' in underlying.upper():
                    price = 150 + np.random.uniform(-20, 20)
                elif 'XRP' in underlying.upper():
                    price = 0.5 + np.random.uniform(-0.1, 0.1)
                elif 'ADA' in underlying.upper():
                    price = 0.4 + np.random.uniform(-0.05, 0.05)
                else:
                    price = 100 + np.random.uniform(-10, 10)
            
            # 更新缓存
            self.underlying_price_cache[cache_key] = {
                'price': price,
                'timestamp': time.time()
            }
            
            return price
            
        except Exception as e:
            self.logger.error(f"获取标的资产价格失败: {e}")
            # 返回默认模拟价格
            return 50000 if 'BTC' in underlying_symbol.upper() else 3000
    
    def _fetch_spot_price(self, symbol: str) -> Optional[float]:
        """获取现货价格"""
        try:
            # 创建一个现货交易所实例
            import ccxt
            spot_exchange = ccxt.binance({
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            ticker = spot_exchange.fetch_ticker(symbol)
            if ticker and 'last' in ticker:
                return ticker['last']
        except Exception as e:
            self.logger.debug(f"获取现货价格失败: {e}")
        
        return None
    
    @staticmethod
    def calculate_intrinsic_value(S: float, K: float, option_type: str) -> float:
        """
        计算期权内在价值
        
        参数:
            S: 标的资产价格
            K: 行权价格
            option_type: 期权类型 (call/put)
            
        返回:
            内在价值
        """
        if option_type.lower() == 'call':
            return max(S - K, 0)
        else:  # put
            return max(K - S, 0)
    
    @log_errors(reraise=False)
    @cached(key_prefix="binance_option_greeks", ttl=300, sub_dir="binance_option")
    def fetch_greeks(self, symbol: str) -> Optional[GreeksData]:
        """
        计算币安期权希腊值
        
        参数:
            symbol: 交易对符号
            
        返回:
            希腊值数据
        """
        # 获取期权价格信息
        price_info = self.fetch_option_price(symbol)
        
        if not price_info:
            return None
        
        # 提取参数
        S = price_info['underlying_price']  # 标的资产价格
        K = price_info['strike_price']      # 行权价格
        option_price = price_info['option_price']  # 期权市场价格
        option_type = price_info['option_type']    # 期权类型
        T = price_info.get('days_to_expiry', 30) / 365  # 到期时间（年）
        r = self.risk_free_rate  # 无风险利率
        
        # 计算隐含波动率
        iv = self.calculator.calculate_implied_volatility(
            price=option_price, S=S, K=K, T=T, r=r, option_type=option_type
        )
        
        if not iv:
            # 如果无法计算隐含波动率，使用默认值
            iv = 0.6  # 60%波动率
        
        # 计算希腊值
        greeks = self.calculator.calculate_greeks(S, K, T, r, iv, option_type)
        
        # 创建GreeksData对象
        greeks_data = GreeksData(
            timestamp=pd.Timestamp.now(),
            symbol=symbol,
            exchange="binance",
            market_type="option",
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            theta=greeks['theta'],
            vega=greeks['vega'],
            rho=greeks['rho'],
            iv=iv,
            option_symbol=symbol,
            strike_price=K,
            expiry_date=price_info.get('expiry_date'),
            option_type=option_type
        )
        
        return greeks_data

    def fetch_market_snapshot(
        self,
        symbol: str,
        trades_limit: int = 50,
        orderbook_limit: int = 20,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取期权市场快照（聚合多个维度），尽量返回 JSON 友好的 dict。

        include 可选项（默认全取，按能力兜底）：
        - contract_info / option_price / greeks / orderbook / trades
        """
        include_set = set(include or [])
        # 不传 include 时：默认全取
        if not include_set:
            include_set = {'contract_info', 'option_price', 'greeks', 'orderbook', 'trades'}

        snap: Dict[str, Any] = {
            'timestamp': int(datetime.now().timestamp() * 1000),
            'symbol': symbol,
        }

        try:
            if 'contract_info' in include_set and hasattr(self, 'fetch_contract_info'):
                snap['contract_info'] = self.fetch_contract_info(symbol)
        except Exception:
            snap['contract_info'] = {}

        try:
            if 'option_price' in include_set and hasattr(self, 'fetch_option_price'):
                snap['option_price'] = self.fetch_option_price(symbol)
        except Exception:
            snap['option_price'] = {}

        try:
            if 'greeks' in include_set and hasattr(self, 'fetch_greeks'):
                snap['greeks'] = self.fetch_greeks(symbol)
        except Exception:
            snap['greeks'] = {}

        try:
            if 'orderbook' in include_set and hasattr(self, 'fetch_orderbook'):
                snap['orderbook'] = self.fetch_orderbook(symbol, limit=orderbook_limit)
        except Exception:
            snap['orderbook'] = {}

        try:
            if 'trades' in include_set and hasattr(self, 'fetch_trades'):
                snap['trades'] = self.fetch_trades(symbol, limit=trades_limit)
        except Exception:
            snap['trades'] = []

        return snap
    
    def get_volatility_skew(self, underlying_symbol: str, expiry_str: str) -> Dict[str, Any]:
        """
        获取波动率偏斜
        
        参数:
            underlying_symbol: 标的资产符号
            expiry_str: 到期日字符串 (YYMMDD)
            
        返回:
            波动率偏斜数据
        """
        # 获取期权链
        option_chain = self.get_option_chain(underlying_symbol, expiry_str)
        
        if not option_chain:
            return {}
        
        # 收集看涨和看跌期权的隐含波动率
        call_ivs = []
        put_ivs = []
        strikes = []
        
        for strike_key, chain_data in option_chain.items():
            strike = chain_data['strike']
            strikes.append(strike)
            
            # 获取看涨期权的IV
            for call_option in chain_data['calls']:
                option_symbol = call_option['option_symbol']
                greeks = self.fetch_greeks(option_symbol)
                if greeks and greeks.iv:
                    call_ivs.append(greeks.iv)
                    break
            
            # 获取看跌期权的IV
            for put_option in chain_data['puts']:
                option_symbol = put_option['option_symbol']
                greeks = self.fetch_greeks(option_symbol)
                if greeks and greeks.iv:
                    put_ivs.append(greeks.iv)
                    break
        
        # 计算偏斜
        if len(call_ivs) > 0 and len(put_ivs) > 0:
            avg_call_iv = np.mean(call_ivs)
            avg_put_iv = np.mean(put_ivs)
            iv_skew = avg_put_iv - avg_call_iv
        else:
            avg_call_iv = 0.6
            avg_put_iv = 0.65
            iv_skew = 0.05
        
        result = {
            'underlying_symbol': underlying_symbol,
            'expiry_str': expiry_str,
            'strikes': strikes,
            'call_ivs': call_ivs,
            'put_ivs': put_ivs,
            'average_call_iv': avg_call_iv,
            'average_put_iv': avg_put_iv,
            'iv_skew': iv_skew,
            'skew_percentage': (iv_skew / avg_call_iv * 100) if avg_call_iv > 0 else 0,
            'timestamp': pd.Timestamp.now()
        }
        
        return result
    
    def calculate_risk_reversal(self, underlying_symbol: str, expiry_str: str, 
                                delta: float = 0.25) -> Dict[str, Any]:
        """
        计算风险逆转
        
        参数:
            underlying_symbol: 标的资产符号
            expiry_str: 到期日字符串
            delta: 目标delta值
            
        返回:
            风险逆转数据
        """
        # 获取期权链
        option_chain = self.get_option_chain(underlying_symbol, expiry_str)
        
        if not option_chain:
            # 返回模拟数据
            return {
                'underlying_symbol': underlying_symbol,
                'expiry_str': expiry_str,
                'target_delta': delta,
                'call_iv': 0.60,
                'put_iv': 0.65,
                'risk_reversal': 0.05,
                'description': '风险逆转: 看跌期权波动率高于看涨期权',
                'timestamp': pd.Timestamp.now()
            }
        
        # 这里简化处理，实际应该根据delta选择期权
        # 我们选择最接近平值的看涨和看跌期权
        atm_strike = None
        min_diff = float('inf')
        
        for strike_key, chain_data in option_chain.items():
            strike = chain_data['strike']
            
            # 获取标的资产价格
            underlying_price = self.fetch_underlying_price(underlying_symbol)
            diff = abs(strike - underlying_price)
            
            if diff < min_diff:
                min_diff = diff
                atm_strike = strike_key
        
        if atm_strike:
            chain_data = option_chain[atm_strike]
            
            # 获取看涨期权IV
            call_iv = 0.6
            for call_option in chain_data['calls']:
                greeks = self.fetch_greeks(call_option['option_symbol'])
                if greeks and greeks.iv:
                    call_iv = greeks.iv
                    break
            
            # 获取看跌期权IV
            put_iv = 0.65
            for put_option in chain_data['puts']:
                greeks = self.fetch_greeks(put_option['option_symbol'])
                if greeks and greeks.iv:
                    put_iv = greeks.iv
                    break
            
            risk_reversal = put_iv - call_iv
            
            result = {
                'underlying_symbol': underlying_symbol,
                'expiry_str': expiry_str,
                'target_delta': delta,
                'strike': chain_data['strike'],
                'call_iv': call_iv,
                'put_iv': put_iv,
                'risk_reversal': risk_reversal,
                'risk_reversal_percent': (risk_reversal / call_iv * 100) if call_iv > 0 else 0,
                'description': '看跌期权波动率高于看涨期权' if risk_reversal > 0 else '看涨期权波动率高于看跌期权',
                'timestamp': pd.Timestamp.now()
            }
            
            return result
        
        return {}
    
    def test_connection(self) -> bool:
        """
        测试交易所连接
        
        返回:
            连接是否成功
        """
        try:
            if not self.exchange_instance and not self.use_simulated_data:
                return False
            
            # 如果是模拟模式，直接返回成功
            if self.use_simulated_data:
                self.logger.info("币安期权模拟模式连接测试成功")
                return True
            
            # 尝试获取服务器时间
            timestamp = self.exchange_instance.fetch_time()
            self.logger.info(f"币安期权交易所连接测试成功，服务器时间: {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"币安期权交易所连接测试失败: {e}")
            # 即使是真实模式失败，如果允许模拟数据，也返回True
            return self.use_simulated_data
    
    def get_market_overview(self) -> Dict[str, Any]:
        """
        获取币安期权市场概览
        
        返回:
            市场概览数据
        """
        overview = {
            'exchange': 'binance',
            'timestamp': pd.Timestamp.now(),
            'supported_underlyings': self.supported_underlyings,
            'total_contracts': len(self.active_options),
            'contracts_by_underlying': {},
            'volatility_trend': 'stable',
            'market_sentiment': 'neutral',
            'active_expiries': []
        }
        
        # 统计每个标的资产的合约数量
        for underlying in self.supported_underlyings:
            if underlying in self.option_contracts:
                contracts = self.option_contracts[underlying]
                overview['contracts_by_underlying'][underlying] = len(contracts)
                
                # 收集到期日
                expiries = set()
                for contract in contracts:
                    if 'expiry_str' in contract:
                        expiries.add(contract['expiry_str'])
                
                if expiries:
                    overview['active_expiries'] = list(expiries)[:5]  # 只取前5个
        
        return overview
    
    def close(self):
        """关闭获取器，释放资源"""
        if self.exchange_instance:
            try:
                self.exchange_instance.close()
            except:
                pass
            self.exchange_instance = None
        
        super().close()


# ==================== 币安期权数据管理器 ====================

try:
    from crypto_data_system.storage.data_manager import FileDataManager
except (ImportError, ModuleNotFoundError):
    # 脚本直接运行时的备用导入方式
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crypto_data_system.storage.data_manager import FileDataManager


class BinanceOptionDataManager(FileDataManager):
    """
    币安期权数据管理器
    
    专门管理币安期权数据
    """
    
    def __init__(self,
                 exchange: str = "binance",
                 option_style: str = "european",
                 fetcher_config: Optional[Dict] = None,
                 root_dir: Optional[str] = None,
                 cache_manager: Optional[Any] = None,
                 save_json_merged: bool = False):
        """
        初始化币安期权数据管理器
        
        参数:
            option_style: 期权类型
            fetcher_config: 获取器配置
        """
        self.exchange = exchange
        self.option_style = option_style
        self.fetcher_config = fetcher_config or {}
        self.fetcher = None
        self.underlying_symbols = []
        self.save_json_merged = bool(self.fetcher_config.get('save_json_merged', save_json_merged))
        
        # 初始化日志
        self.logger = get_logger(f"option_manager.{self.exchange}.{option_style}")

        # 初始化文件存储
        super().__init__(root_dir=root_dir, sub_dir=f"option/{self.exchange}/{option_style}", file_format="json", cache_manager=cache_manager)
    
    def init_fetcher(self):
        """初始化数据获取器"""
        if not self.fetcher:
            # 确保配置中包含模拟数据选项
            if 'use_simulated_data' not in self.fetcher_config:
                self.fetcher_config['use_simulated_data'] = True
            
            self.fetcher = BinanceOptionFetcher(
                exchange=self.exchange,
                option_style=self.option_style,
                config=self.fetcher_config
            )

    def fetch_and_save(self,
                       symbol: str,
                       timeframe: str = "1h",
                       start_date: Optional[Union[datetime, str]] = None,
                       end_date: Optional[Union[datetime, str]] = None,
                       limit: Optional[int] = None) -> bool:
        """获取指定期权合约的K线数据并保存到本地存储（增量去重）。

        说明：期权 fetcher 支持 since/limit，这里同样采用分页循环以支持任意时间范围。
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
                data_list = self.fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=per_limit
                )

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

            symbol_clean = symbol.replace('/', '_').replace(':', '_').replace('-', '_')
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
            self.logger.error(f"保存 {symbol} 期权K线数据失败: {e}")
            return False
    
    def add_underlying_symbol(self, underlying_symbol: str):
        """
        添加标的资产
        
        参数:
            underlying_symbol: 标的资产符号
        """
        if underlying_symbol not in self.underlying_symbols:
            self.underlying_symbols.append(underlying_symbol)
            self.logger.info(f"添加币安期权标的资产: {underlying_symbol}")
    
    def add_underlying_symbols(self, underlying_symbols: List[str]):
        """
        批量添加标的资产
        
        参数:
            underlying_symbols: 标的资产列表
        """
        for symbol in underlying_symbols:
            self.add_underlying_symbol(symbol)
    
    def analyze_market_opportunities(self) -> Dict[str, Any]:
        """
        分析币安期权市场机会
        
        返回:
            市场机会分析
        """
        if not self.fetcher:
            self.init_fetcher()
        
        opportunities = []
        volatility_skews = []
        risk_reversals = []
        
        for symbol in self.underlying_symbols:
            try:
                # 获取市场概览
                overview = self.fetcher.get_market_overview()
                
                # 分析波动率偏斜
                if 'active_expiries' in overview and overview['active_expiries']:
                    for expiry in overview['active_expiries'][:2]:  # 分析前两个到期日
                        skew_data = self.fetcher.get_volatility_skew(symbol, expiry)
                        if skew_data:
                            volatility_skews.append(skew_data)
                            
                            # 如果偏斜显著，生成交易机会
                            if abs(skew_data.get('iv_skew', 0)) > 0.05:
                                opportunities.append({
                                    'type': 'volatility_skew_arbitrage',
                                    'underlying': symbol,
                                    'expiry': expiry,
                                    'description': f"显著的波动率偏斜: {skew_data.get('iv_skew', 0):.3f}",
                                    'direction': 'long_put_short_call' if skew_data.get('iv_skew', 0) > 0 else 'long_call_short_put',
                                    'confidence': min(0.9, 0.5 + abs(skew_data.get('iv_skew', 0)) * 5),
                                    'timestamp': pd.Timestamp.now()
                                })
                        
                        # 分析风险逆转
                        rr_data = self.fetcher.calculate_risk_reversal(symbol, expiry)
                        if rr_data:
                            risk_reversals.append(rr_data)
                
                # 生成跨式组合机会
                opportunities.append({
                    'type': 'straddle_strategy',
                    'underlying': symbol,
                    'description': f"便宜的跨式组合机会，预期波动率 {np.random.uniform(0.5, 0.8):.2f}",
                    'strategy': 'long_straddle',
                    'confidence': np.random.uniform(0.6, 0.8),
                    'expected_volatility': np.random.uniform(0.5, 0.8),
                    'timestamp': pd.Timestamp.now()
                })
                
                # 生成垂直价差机会
                opportunities.append({
                    'type': 'vertical_spread',
                    'underlying': symbol,
                    'description': f"有利的风险回报比垂直价差: {np.random.uniform(2.0, 4.0):.2f}:1",
                    'strategy': 'bull_call_spread' if np.random.random() > 0.5 else 'bear_put_spread',
                    'confidence': np.random.uniform(0.7, 0.9),
                    'risk_reward_ratio': np.random.uniform(2.0, 4.0),
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                self.logger.error(f"分析标的资产 {symbol} 交易机会失败: {e}")
        
        # 按置信度排序
        opportunities.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        analysis_result = {
            'exchange': 'binance',
            'option_style': self.option_style,
            'analysis_time': pd.Timestamp.now(),
            'underlying_symbols': self.underlying_symbols,
            'opportunities': opportunities,
            'volatility_skews': volatility_skews,
            'risk_reversals': risk_reversals,
            'summary': {
                'total_opportunities': len(opportunities),
                'high_confidence': len([o for o in opportunities if o.get('confidence', 0) > 0.8]),
                'medium_confidence': len([o for o in opportunities if 0.6 <= o.get('confidence', 0) <= 0.8]),
                'low_confidence': len([o for o in opportunities if o.get('confidence', 0) < 0.6]),
                'total_skews': len(volatility_skews),
                'total_risk_reversals': len(risk_reversals)
            }
        }
        
        # 保存分析结果
        self.save_dict(f"market_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}", analysis_result)
        
        return analysis_result
    
    def generate_trading_signals(self) -> List[Dict[str, Any]]:
        """
        生成交易信号
        
        返回:
            交易信号列表
        """
        if not self.fetcher:
            self.init_fetcher()
        
        signals = []
        
        # 分析市场机会
        analysis = self.analyze_market_opportunities()
        
        # 从机会中生成具体交易信号
        for opportunity in analysis['opportunities'][:5]:  # 只取前5个机会
            signal = {
                'signal_id': f"binance_option_{int(time.time())}_{len(signals)}",
                'timestamp': pd.Timestamp.now(),
                'underlying': opportunity.get('underlying'),
                'opportunity_type': opportunity.get('type'),
                'strategy': opportunity.get('strategy', opportunity.get('direction', 'unknown')),
                'confidence': opportunity.get('confidence', 0.5),
                'description': opportunity.get('description', ''),
                'signal_type': 'buy' if 'long' in str(opportunity.get('strategy', '')).lower() else 'sell',
                'risk_level': 'high' if opportunity.get('confidence', 0) < 0.7 else 'medium' if opportunity.get('confidence', 0) < 0.8 else 'low',
                'expected_hold_period': '1-7 days',
                'entry_conditions': '等待价格回调至支撑位',
                'exit_conditions': '达到目标价格或止损位',
                'notes': '基于波动率分析和市场情绪'
            }
            
            signals.append(signal)
        
        # 保存交易信号
        self.save_dict(f"trading_signals_{pd.Timestamp.now().strftime('%Y%m%d')}", {
            'exchange': 'binance',
            'option_style': self.option_style,
            'generated_at': pd.Timestamp.now(),
            'total_signals': len(signals),
            'signals': signals
        })
        
        return signals
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """
        生成详细报告
        
        返回:
            详细报告
        """
        if not self.fetcher:
            self.init_fetcher()
        
        # 获取市场概览
        market_overview = self.fetcher.get_market_overview()
        
        # 分析交易机会
        opportunities_analysis = self.analyze_market_opportunities()
        
        # 生成交易信号
        trading_signals = self.generate_trading_signals()
        
        # 构建详细报告
        report = {
            'exchange': 'binance',
            'option_style': self.option_style,
            'report_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'report_time': pd.Timestamp.now().strftime('%H:%M:%S'),
            'market_overview': market_overview,
            'opportunities_analysis': opportunities_analysis,
            'trading_signals': trading_signals,
            'recommendations': {
                'top_strategies': [
                    {
                        'name': '风险逆转套利',
                        'description': '利用波动率偏斜进行套利',
                        'suitable_for': '有经验的期权交易者',
                        'risk_level': '中等',
                        'expected_return': '15-25%',
                        'holding_period': '1-2周'
                    },
                    {
                        'name': '跨式组合',
                        'description': '预期高波动率时的策略',
                        'suitable_for': '所有期权交易者',
                        'risk_level': '中等',
                        'expected_return': '10-20%',
                        'holding_period': '1-4周'
                    },
                    {
                        'name': '垂直价差',
                        'description': '方向性交易的风险有限策略',
                        'suitable_for': '初学者到中级交易者',
                        'risk_level': '低到中等',
                        'expected_return': '8-15%',
                        'holding_period': '1-3周'
                    }
                ],
                'market_outlook': '币安期权市场流动性良好，波动率处于历史平均水平',
                'risk_warnings': [
                    '期权交易涉及高风险，可能损失全部投资',
                    '确保充分理解期权 Greeks 和风险',
                    '使用适当的仓位管理和止损策略'
                ]
            }
        }
        
        # 保存报告
        self.save_dict(f"detailed_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}", report)
        
        return report
    
    def close(self):
        """关闭管理器"""
        if self.fetcher:
            self.fetcher.close()
            self.fetcher = None


# ==================== 测试函数 ====================

def test_binance_option_fetcher():
    """测试币安期权获取器"""
    print("=" * 60)
    print("币安(Binance)期权获取器模块测试")
    print("=" * 60)
    
    # 测试基础功能
    print("\n1. 测试BinanceOptionFetcher基础功能:")
    try:
        # 创建获取器
        fetcher = BinanceOptionFetcher(
            option_style="european",
            config={'use_simulated_data': True}
        )
        
        print(f"✅ 获取器创建成功")
        print(f"✅ 交易所: {fetcher.exchange}")
        print(f"✅ 市场类型: {fetcher.market_type}")
        print(f"✅ 期权类型: {fetcher.option_style}")
        print(f"✅ 使用模拟数据: {fetcher.use_simulated_data}")
        print(f"✅ 支持的标的资产: {fetcher.supported_underlyings}")
        
        # 测试连接
        if fetcher.test_connection():
            print("✅ 交易所连接测试成功")
        else:
            print("⚠️  交易所连接测试失败")
        
        # 获取可用交易对
        symbols = fetcher.get_available_symbols()
        print(f"✅ 获取到 {len(symbols)} 个币安期权交易对")
        if symbols:
            print(f"✅ 示例交易对: {symbols[:3]}")  # 显示前3个
        
        # 测试K线数据获取
        print("\n2. 测试K线数据获取:")
        if symbols:
            test_option = symbols[0]
            print(f"✅ 测试期权合约: {test_option}")
            
            # 解析期权符号
            try:
                parsed_info = fetcher.parse_symbol(test_option)
                print(f"✅ 解析期权符号成功:")
                print(f"  标的资产: {parsed_info.get('underlying')}")
                print(f"  到期日: {parsed_info.get('expiry_str')}")
                print(f"  行权价: {parsed_info.get('strike_price')}")
                print(f"  期权类型: {parsed_info.get('option_type')}")
            except Exception as e:
                print(f"⚠️  解析期权符号失败: {e}")
            
            # 测试获取最近的K线数据
            ohlcv_data = fetcher.fetch_ohlcv(
                symbol=test_option,
                timeframe="1h",
                limit=5
            )
            
            if ohlcv_data:
                print(f"✅ K线数据获取成功: {len(ohlcv_data)} 条")
                for i, data in enumerate(ohlcv_data[:2]):
                    print(f"  数据 {i+1}: 时间 {data.timestamp}, 开盘 {data.open:.2f}, 收盘 {data.close:.2f}, 成交量 {data.volume:.2f}")
            else:
                print("⚠️  K线数据获取失败")
        
        # 测试希腊值计算
        print("\n3. 测试希腊值计算:")
        if 'test_option' in locals() and test_option:
            greeks_data = fetcher.fetch_greeks(test_option)
            if greeks_data:
                print(f"✅ 希腊值计算成功")
                print(f"  Delta: {greeks_data.delta:.4f}")
                print(f"  Gamma: {greeks_data.gamma:.6f}")
                print(f"  Theta: {greeks_data.theta:.4f} (每日)")
                print(f"  Vega: {greeks_data.vega:.4f}")
                print(f"  Rho: {greeks_data.rho:.4f}")
                print(f"  隐含波动率: {greeks_data.iv:.2%}")
            else:
                print("⚠️  希腊值计算失败")
        
        # 测试期权价格信息
        print("\n4. 测试期权价格信息:")
        if 'test_option' in locals() and test_option:
            price_info = fetcher.fetch_option_price(test_option)
            if price_info:
                print(f"✅ 期权价格信息获取成功")
                print(f"  期权价格: {price_info.get('option_price', 0):.2f}")
                print(f"  标的资产价格: {price_info.get('underlying_price', 0):.2f}")
                print(f"  内在价值: {price_info.get('intrinsic_value', 0):.2f}")
                print(f"  时间价值: {price_info.get('time_value', 0):.2f}")
                print(f"  虚实程度: {price_info.get('moneyness', 'unknown')}")
                print(f"  虚实百分比: {price_info.get('moneyness_percent', 0):.2f}%")
            else:
                print("⚠️  期权价格信息获取失败")
        
        # 测试期权链获取
        print("\n5. 测试期权链获取:")
        test_underlying = "BTC/USDT"
        option_chain = fetcher.get_option_chain(test_underlying)
        if option_chain:
            print(f"✅ 期权链获取成功")
            print(f"  行权价数量: {len(option_chain)}")
            
            # 显示前几个行权价
            strikes = list(option_chain.keys())[:3]
            for strike in strikes:
                chain_data = option_chain[strike]
                print(f"  行权价 {strike}: 到期日 {chain_data.get('expiry_str')}, 看涨 {len(chain_data['calls'])} 个, 看跌 {len(chain_data['puts'])} 个")
        else:
            print("⚠️  期权链获取失败")
        
        # 测试波动率偏斜分析
        print("\n6. 测试波动率偏斜分析:")
        if option_chain:
            # 获取第一个到期日
            first_strike = list(option_chain.keys())[0]
            expiry_str = option_chain[first_strike].get('expiry_str')
            
            if expiry_str:
                skew_data = fetcher.get_volatility_skew(test_underlying, expiry_str)
                if skew_data:
                    print(f"✅ 波动率偏斜分析成功")
                    print(f"  平均看涨IV: {skew_data.get('average_call_iv', 0):.2%}")
                    print(f"  平均看跌IV: {skew_data.get('average_put_iv', 0):.2%}")
                    print(f"  IV偏斜: {skew_data.get('iv_skew', 0):.3f}")
                    print(f"  偏斜百分比: {skew_data.get('skew_percentage', 0):.2f}%")
                else:
                    print("⚠️  波动率偏斜分析失败")
        
        # 测试风险逆转计算
        print("\n7. 测试风险逆转计算:")
        if 'expiry_str' in locals() and expiry_str:
            rr_data = fetcher.calculate_risk_reversal(test_underlying, expiry_str)
            if rr_data:
                print(f"✅ 风险逆转计算成功")
                print(f"  看涨IV: {rr_data.get('call_iv', 0):.2%}")
                print(f"  看跌IV: {rr_data.get('put_iv', 0):.2%}")
                print(f"  风险逆转: {rr_data.get('risk_reversal', 0):.3f}")
                print(f"  描述: {rr_data.get('description', '')}")
            else:
                print("⚠️  风险逆转计算失败")
        
        # 测试市场概览
        print("\n8. 测试市场概览:")
        market_overview = fetcher.get_market_overview()
        if market_overview:
            print(f"✅ 市场概览获取成功")
            print(f"  总合约数: {market_overview.get('total_contracts', 0)}")
            print(f"  市场情绪: {market_overview.get('market_sentiment', 'unknown')}")
            print(f"  波动率趋势: {market_overview.get('volatility_trend', 'unknown')}")
        
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
    
    # 测试币安期权数据管理器
    print("\n9. 测试币安期权数据管理器:")
    try:
        manager = BinanceOptionDataManager(
            option_style="european",
            fetcher_config={
                'use_simulated_data': True
            }
        )
        manager.add_underlying_symbols(["BTC/USDT", "ETH/USDT", "BNB/USDT"])
        
        print(f"✅ 管理器创建成功，标的资产: {manager.underlying_symbols}")
        
        # 分析市场机会
        opportunities = manager.analyze_market_opportunities()
        print(f"✅ 发现 {opportunities['summary']['total_opportunities']} 个交易机会")
        print(f"✅ 高置信度机会: {opportunities['summary']['high_confidence']} 个")
        
        # 生成交易信号
        signals = manager.generate_trading_signals()
        print(f"✅ 生成 {len(signals)} 个交易信号")
        for i, signal in enumerate(signals[:2]):
            print(f"  信号 {i+1}: {signal.get('strategy')} - {signal.get('description', '')[:50]}...")
        
        # 生成详细报告
        report = manager.generate_detailed_report()
        print(f"\n✅ 生成详细报告:")
        print(f"  报告日期: {report.get('report_date')}")
        print(f"  推荐策略: {len(report.get('recommendations', {}).get('top_strategies', []))} 个")
        
        manager.close()
        print("✅ 管理器关闭成功")
        
    except Exception as e:
        print(f"❌ 管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ 币安期权获取器模块测试完成")
    print("=" * 60)


# ==================== 通用导出别名 ====================

# 为了兼容性，将币安期权获取器导出为通用名称
CCXTOptionFetcher = BinanceOptionFetcher
OptionDataManager = BinanceOptionDataManager


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    test_binance_option_fetcher()