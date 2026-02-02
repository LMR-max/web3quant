"""配置文件模块 - 加密货币数据获取系统配置

注意：本模块不会在代码中硬编码任何 API Key。
请将密钥放到本地未提交文件 `crypto_data_system/local_secrets.json` 或环境变量中。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# ==================== Secrets / 环境变量配置 ====================

SECRETS_FILE_ENV = "WEB3QUANT_SECRETS_FILE"
DEFAULT_SECRETS_FILE = Path(__file__).with_name("local_secrets.json")

# 这里集中维护项目会用到的环境变量 key 名称
API_SECRET_KEYS: List[str] = [
    # 链上/数据分析
    "DUNE_API_KEY",
    "THEGRAPH_API_KEY",
    "THEGRAPH_API_TOKEN",
    "ETHERSCAN_API_KEY",
    "BSCSCAN_API_KEY",
    "POLYGONSCAN_API_KEY",
    "ARBISCAN_API_KEY",
    "ALCHEMY_API_KEY",
    "INFURA_PROJECT_ID",
    "ANKR_API_KEY",
    "BLOCKNATIVE_API_KEY",
    # 市场数据
    "COINGECKO_API_KEY",
    "COINMARKETCAP_API_KEY",
    # 社交/舆情
    "X_API_KEY",
    "X_API_KEY_SECRET",
]


def _read_json_secrets_file(filepath: Path) -> Dict[str, str]:
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Secrets 文件格式错误（应为 JSON 对象）: {filepath}")

    secrets: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str) and value:
            secrets[key] = value
    return secrets


def ensure_api_secrets_loaded(
    secrets_file: Optional[str] = None,
    *,
    override_existing_env: bool = False,
) -> Dict[str, str]:
    """确保 API secrets 已加载到环境变量中。

    加载优先级：
    1) 已存在的环境变量
    2) `secrets_file` 参数指定的 JSON 文件
    3) 环境变量 `WEB3QUANT_SECRETS_FILE` 指定的 JSON 文件
    4) 默认文件 `crypto_data_system/local_secrets.json`

    返回：从 secrets 文件加载到的键值（不包含原本就在环境变量中的）。
    """

    candidate = (
        Path(secrets_file)
        if secrets_file
        else Path(os.environ.get(SECRETS_FILE_ENV, ""))
        if os.environ.get(SECRETS_FILE_ENV)
        else DEFAULT_SECRETS_FILE
    )

    loaded_from_file: Dict[str, str] = {}
    if candidate.exists() and candidate.is_file():
        try:
            loaded_from_file = _read_json_secrets_file(candidate)
        except Exception as e:
            raise RuntimeError(f"读取 secrets 文件失败: {candidate} ({e})") from e

    for key in API_SECRET_KEYS:
        if key in loaded_from_file:
            if override_existing_env or not os.environ.get(key):
                os.environ[key] = loaded_from_file[key]

    return loaded_from_file

# ==================== 主要配置类 ====================

@dataclass
class ExchangeConfig:
    """交易所配置"""
    exchange_id: str = "binance"  # 交易所ID: binance, okx, bybit
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    proxy_url: Optional[str] = "http://127.0.0.1:7890"  # 代理地址，如: http://127.0.0.1:7890
    rate_limit: int = 1000  # 请求频率限制
    timeout: int = 30000  # 超时时间(ms)
    
    # CCXT特定配置
    ccxt_options: Dict[str, Any] = field(default_factory=lambda: {
        'enableRateLimit': True,
        'adjustForTimeDifference': True,
        'warnOnFetchOHLCVLimitArgument': False
    })
    
    def get_ccxt_config(self) -> Dict[str, Any]:
        """获取CCXT配置字典"""
        config = {
            'enableRateLimit': True,
            'timeout': self.timeout,
            'rateLimit': self.rate_limit,
        }
        
        # 添加代理
        if self.proxy_url:
            config['proxies'] = {'http': self.proxy_url, 'https': self.proxy_url}
        
        # 添加API密钥
        if self.api_key and self.api_secret:
            config['apiKey'] = self.api_key
            config['secret'] = self.api_secret
        
        # 合并CCXT选项
        config.update(self.ccxt_options)
        
        return config


@dataclass
class DataFetchConfig:
    """数据获取配置"""
    symbol: str = "BTC/USDT"  # 交易对
    start_date: str = "2023-12-01"  # 开始日期
    end_date: str = "2025-12-01"  # 结束日期
    timeframe: str = "1h"  # 时间间隔
    limit: int = None  # 每次请求数量限制
    data_types: List[str] = field(default_factory=lambda: ['ohlcv'])  # 数据类型
    
    # 可选的数据类型
    ALL_DATA_TYPES = ['ohlcv', 'orderbook', 'trades', 'funding_rate', 
                     'open_interest', 'liquidations', 'greeks']
    
    # 支持的时间间隔
    SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    
    def validate(self) -> bool:
        """验证配置"""
        if self.timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise ValueError(f"不支持的时间间隔: {self.timeframe}")
        
        for data_type in self.data_types:
            if data_type not in self.ALL_DATA_TYPES:
                raise ValueError(f"不支持的数据类型: {data_type}")
        
        return True


@dataclass
class StorageConfig:
    """存储配置"""
    base_dir: str = "./crypto_data"  # 基础目录
    format: str = "parquet"  # 存储格式: parquet, csv, feather
    compression: str = "snappy"  # 压缩格式: snappy, gzip, brotli
    partition_by: List[str] = field(default_factory=lambda: ['symbol', 'timeframe'])  # 分区字段
    
    # 缓存配置
    cache_enabled: bool = True
    cache_dir: str = "./data/cache"
    cache_expire: Dict[str, int] = field(default_factory=lambda: {
        'tick': 60,      # 1分钟
        'minute': 300,   # 5分钟
        'hour': 3600,    # 1小时
        'day': 86400     # 1天
    })
    
    def get_file_extension(self) -> str:
        """获取文件扩展名"""
        return {
            'parquet': '.parquet',
            'csv': '.csv',
            'feather': '.feather'
        }.get(self.format, '.parquet')


@dataclass
class APIConfig:
    """API配置"""
    
    # Dune Analytics配置
    dune_api_url: str = "https://api.dune.com/api/v2"
    dune_api_key: Optional[str] = None

    # The Graph 配置
    thegraph_api_key: Optional[str] = None
    thegraph_api_token: Optional[str] = None
    
    # X (Twitter) API配置
    x_api_url: str = "https://api.x.com"
    x_oauth2_url: str = "https://api.x.com/oauth2/token"
    x_api_key: Optional[str] = None
    x_api_key_secret: Optional[str] = None
    
    # 其他API配置
    coinmarketcap_api_key: Optional[str] = None
    coingecko_api_key: Optional[str] = None
    
    def __post_init__(self):
        """初始化后设置，从环境变量读取API密钥"""
        # 尝试从本地 secrets 文件加载（如果存在），并填充到环境变量中
        ensure_api_secrets_loaded()
        
        if not self.dune_api_key:
            self.dune_api_key = os.environ.get('DUNE_API_KEY')
        
        if not self.x_api_key:
            self.x_api_key = os.environ.get('X_API_KEY')
        
        if not self.x_api_key_secret:
            self.x_api_key_secret = os.environ.get('X_API_KEY_SECRET')

        if not self.thegraph_api_key:
            self.thegraph_api_key = os.environ.get('THEGRAPH_API_KEY') or os.environ.get('THE_GRAPH_API_KEY')

        if not self.thegraph_api_token:
            self.thegraph_api_token = os.environ.get('THEGRAPH_API_TOKEN') or os.environ.get('THE_GRAPH_API_TOKEN')


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"  # 日志级别: DEBUG, INFO, WARNING, ERROR
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file: str = "crypto_data.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5  # 保留的备份文件数


# ==================== 交易所特定配置 ====================

class ExchangeURLs:
    """交易所URL配置"""
    BINANCE = {
        'spot': 'https://api.binance.com',
        'futures': 'https://fapi.binance.com',
        'coin_futures': 'https://dapi.binance.com',
        'options': 'https://eapi.binance.com',
        'margin': 'https://api.binance.com'
    }
    
    OKX = {
        'spot': 'https://www.okx.com',
        'futures': 'https://www.okx.com',
        'options': 'https://www.okx.com',
        'margin': 'https://www.okx.com'
    }
    
    BYBIT = {
        'spot': 'https://api.bybit.com',
        'futures': 'https://api.bybit.com',
        'options': 'https://api.bybit.com',
        'margin': 'https://api.bybit.com'
    }
    
    @classmethod
    def get_urls(cls, exchange_id: str) -> Dict[str, str]:
        """获取交易所URLs"""
        exchange_id = exchange_id.lower()
        
        if exchange_id == 'binance':
            return cls.BINANCE
        elif exchange_id == 'okx':
            return cls.OKX
        elif exchange_id == 'bybit':
            return cls.BYBIT
        else:
            raise ValueError(f"不支持的交易所: {exchange_id}")


class ExchangeSymbolFormats:
    """交易所交易对格式配置"""
    
    @staticmethod
    def format_binance_symbol(symbol: str, market_type: str = 'spot') -> str:
        """格式化币安交易对"""
        if '/' not in symbol:
            return symbol
        
        base, quote = symbol.split('/')
        
        if market_type == 'swap':
            # 永续合约格式: BTC/USDT:USDT
            return f"{base}/{quote}:{quote}"
        elif market_type == 'coin_futures':
            # 币本位合约格式: BTC/USD:BTC
            return f"{base}/{quote}:{base}"
        elif market_type == 'margin':
            # 杠杆交易对格式不变
            return symbol
        else:
            # 现货格式不变
            return symbol
    
    @staticmethod
    def format_okx_symbol(symbol: str, market_type: str = 'spot') -> str:
        """格式化OKX交易对
        
        注意：OKX 现货通过 CCXT 已标准化为 / 格式，不需要转换为 -
        只有 swap/futures 等衍生品才使用 - 格式
        """
        if '/' not in symbol:
            return symbol
        
        base, quote = symbol.split('/')
        
        if market_type == 'swap':
            # OKX永续合约格式: BTC-USDT-SWAP
            return f"{base}-{quote}-SWAP"
        elif market_type == 'futures':
            # OKX期货合约格式: BTC-USDT-YYMMDD
            return f"{base}-{quote}-"
        else:
            # 现货和杠杆格式保持 CCXT 的 / 格式（不转换为 -）
            return symbol
    
    @staticmethod
    def format_symbol(symbol: str, exchange_id: str, market_type: str = 'spot') -> str:
        """格式化交易对符号"""
        if exchange_id == 'binance':
            return ExchangeSymbolFormats.format_binance_symbol(symbol, market_type)
        elif exchange_id == 'okx':
            return ExchangeSymbolFormats.format_okx_symbol(symbol, market_type)
        else:
            # 其他交易所保持原样
            return symbol


# ==================== 预设配置 ====================

class PresetConfigs:
    """预设配置"""
    
    @staticmethod
    def get_spot_config() -> Dict[str, Any]:
        """现货配置"""
        return {
            'default_type': 'spot',
            'fetch_markets_types': ['spot'],
            'ccxt_options': {
                'defaultType': 'spot',
                'fetchMarkets': ['spot']
            }
        }
    
    @staticmethod
    def get_swap_config() -> Dict[str, Any]:
        """永续合约配置"""
        return {
            'default_type': 'swap',
            'fetch_markets_types': ['linear'],
            'ccxt_options': {
                'defaultType': 'swap',
                'fetchMarkets': ['linear']
            }
        }
    
    @staticmethod
    def get_margin_config() -> Dict[str, Any]:
        """杠杆配置"""
        return {
            'default_type': 'margin',
            'fetch_markets_types': ['spot'],
            'fetch_margins': True,
            'ccxt_options': {
                'defaultType': 'margin',
                'fetchMarkets': ['spot']
            }
        }
    
    @staticmethod
    def get_future_config() -> Dict[str, Any]:
        """期货配置"""
        return {
            'default_type': 'future',
            'fetch_markets_types': ['inverse'],
            'ccxt_options': {
                'defaultType': 'future',
                'fetchMarkets': ['inverse']
            }
        }
    
    @staticmethod
    def get_option_config() -> Dict[str, Any]:
        """期权配置"""
        return {
            'default_type': 'option',
            'fetch_markets_types': ['option'],
            'load_all_options': True,
            'ccxt_options': {
                'defaultType': 'option',
                'fetchMarkets': ['option']
            }
        }


# ==================== 配置管理器 ====================

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.configs = {}
        self.load_default_configs()
    
    def load_default_configs(self):
        """加载默认配置"""
        # 默认交易所配置
        self.configs['exchange_binance'] = ExchangeConfig(
            exchange_id='binance',
            proxy_url='http://127.0.0.1:7890'  # 默认代理
        )
        
        # 默认数据获取配置
        self.configs['data_fetch'] = DataFetchConfig(
            symbol='BTC/USDT',
            start_date='2023-12-01',
            end_date='2025-12-01',
            timeframe='1h',
            limit=1000,
            data_types=['ohlcv']
        )
        
        # 默认存储配置
        self.configs['storage'] = StorageConfig(
            base_dir='./crypto_data',
            format='parquet',
            compression='snappy'
        )
        
        # 默认API配置
        self.configs['api'] = APIConfig()
        
        # 默认日志配置
        self.configs['logging'] = LoggingConfig()
    
    def get_config(self, config_name: str) -> Any:
        """获取配置"""
        if config_name in self.configs:
            return self.configs[config_name]
        else:
            raise KeyError(f"配置 {config_name} 不存在")
    
    def update_config(self, config_name: str, **kwargs):
        """更新配置"""
        if config_name in self.configs:
            config = self.configs[config_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    raise AttributeError(f"配置 {config_name} 没有属性 {key}")
        else:
            raise KeyError(f"配置 {config_name} 不存在")
    
    def save_config_to_file(self, filepath: str = 'config.json'):
        """保存配置到文件"""
        import json
        
        config_dict = {}
        for name, config in self.configs.items():
            if hasattr(config, '__dict__'):
                config_dict[name] = config.__dict__
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"✅ 配置已保存到: {filepath}")
    
    def load_config_from_file(self, filepath: str = 'config.json'):
        """从文件加载配置"""
        import json
        
        if not os.path.exists(filepath):
            print(f"⚠️ 配置文件不存在: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        for name, config_data in config_dict.items():
            if name in self.configs:
                config = self.configs[name]
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        print(f"✅ 配置已从文件加载: {filepath}")
        return True


# ==================== 全局配置实例 ====================

# 创建全局配置管理器实例
config_manager = ConfigManager()

# 便捷访问函数
def get_exchange_config(exchange_id: str = 'binance') -> ExchangeConfig:
    """获取交易所配置"""
    config_key = f"exchange_{exchange_id}"
    
    if config_key not in config_manager.configs:
        # 创建新的交易所配置
        config_manager.configs[config_key] = ExchangeConfig(exchange_id=exchange_id)
    
    return config_manager.configs[config_key]

def get_data_fetch_config() -> DataFetchConfig:
    """获取数据获取配置"""
    return config_manager.get_config('data_fetch')

def get_storage_config() -> StorageConfig:
    """获取存储配置"""
    return config_manager.get_config('storage')

def get_api_config() -> APIConfig:
    """获取API配置"""
    return config_manager.get_config('api')

def get_logging_config() -> LoggingConfig:
    """获取日志配置"""
    return config_manager.get_config('logging')

def get_market_config(market_type: str) -> Dict[str, Any]:
    """获取市场类型配置"""
    market_type = market_type.lower()
    
    if market_type == 'spot':
        return PresetConfigs.get_spot_config()
    elif market_type == 'swap':
        return PresetConfigs.get_swap_config()
    elif market_type == 'margin':
        return PresetConfigs.get_margin_config()
    elif market_type == 'future':
        return PresetConfigs.get_future_config()
    elif market_type == 'option':
        return PresetConfigs.get_option_config()
    else:
        raise ValueError(f"不支持的市场类型: {market_type}")


# ==================== 工具函数 ====================

def get_all_symbols(market_type: str = 'spot') -> List[str]:
    """获取所有交易对（示例）"""
    if market_type == 'spot':
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
        ]
    elif market_type == 'swap':
        return [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT',
            'ADA/USDT:USDT', 'SOL/USDT:USDT'
        ]
    elif market_type == 'margin':
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    else:
        return ['BTC/USDT']

def get_timeframe_seconds(timeframe: str) -> int:
    """将时间间隔转换为秒数"""
    timeframe_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800
    }
    
    return timeframe_map.get(timeframe, 3600)

def split_date_range(start_date: str, end_date: str, timeframe: str, max_bars: int = 1000):
    """分割日期范围以适应API限制"""
    from datetime import datetime, timedelta
    import pandas as pd
    
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    # 计算每个时间间隔的秒数
    seconds_per_bar = get_timeframe_seconds(timeframe)
    
    # 计算最大时间范围（秒）
    max_seconds = max_bars * seconds_per_bar
    
    chunks = []
    current = start_dt
    
    while current < end_dt:
        chunk_end = min(current + pd.Timedelta(seconds=max_seconds), end_dt)
        chunks.append((current, chunk_end))
        current = chunk_end
    
    return chunks


# ==================== 测试配置 ====================

if __name__ == "__main__":
    # 测试配置
    ensure_api_secrets_loaded()
    
    # 打印配置信息
    print("=" * 60)
    print("配置模块测试")
    print("=" * 60)
    
    # 测试获取配置
    exchange_config = get_exchange_config('binance')
    print(f"交易所配置: {exchange_config.exchange_id}")
    print(f"代理地址: {exchange_config.proxy_url}")
    
    data_config = get_data_fetch_config()
    print(f"数据配置 - 交易对: {data_config.symbol}")
    print(f"数据配置 - 时间范围: {data_config.start_date} 到 {data_config.end_date}")
    print(f"数据配置 - 时间间隔: {data_config.timeframe}")
    
    storage_config = get_storage_config()
    print(f"存储配置 - 格式: {storage_config.format}")
    print(f"存储配置 - 压缩: {storage_config.compression}")
    
    # 测试日期分割
    chunks = split_date_range("2023-01-01", "2023-12-31", "1h")
    print(f"日期分割: {len(chunks)} 个分块")
    
    print("✅ 配置模块测试完成")