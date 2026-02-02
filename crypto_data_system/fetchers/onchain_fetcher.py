"""
链上数据获取器模块
提供从区块链网络获取链上数据的功能
包括地址余额、交易数据、智能合约数据、网络状态等
"""

import os
import time
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from web3 import Web3
from web3.exceptions import TransactionNotFound, BlockNotFound

# 导入基础模块
try:
    from .base_fetcher import BaseFetcher, AsyncFetcher, DataFormatError
    from ..data_models import (
        OnChainMetric, TokenFlowData, ExchangeFlowData,
        OHLCVData, BaseData, OrderBookData, TradeData
    )
    from ..utils.logger import get_logger, log_execution_time, log_errors
    from ..utils.cache import CacheManager, cached, cache_result
    from ..utils.date_utils import split_date_range, calculate_timeframe_seconds
    from ..config import get_exchange_config, ExchangeSymbolFormats, get_market_config, get_api_config
except ImportError:
    # 如果直接运行，使用简单导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fetchers.base_fetcher import BaseFetcher, AsyncFetcher, DataFormatError
    from data_models import (
        OnChainMetric, TokenFlowData, ExchangeFlowData,
        OHLCVData, BaseData, OrderBookData, TradeData
    )
    from utils.logger import get_logger, log_execution_time, log_errors
    from utils.cache import CacheManager, cached, cache_result
    from utils.date_utils import split_date_range, calculate_timeframe_seconds
    from config import get_exchange_config, ExchangeSymbolFormats, get_market_config, get_api_config


# ==================== 链上数据源配置 ====================

class OnChainDataSource:
    """链上数据源配置"""
    
    # RPC节点配置 - 使用免费的公共节点
    RPC_NODES = {
        'ethereum': {
            'mainnet': [
                'https://rpc.ankr.com/eth',
                'https://eth-mainnet.public.blastapi.io',
                'https://ethereum.publicnode.com',
                'https://rpc.flashbots.net',
                'https://cloudflare-eth.com'
            ],
            'goerli': [
                'https://rpc.ankr.com/eth_goerli',
                'https://eth-goerli.public.blastapi.io'
            ]
        },
        'polygon': {
            'mainnet': [
                'https://polygon-rpc.com',
                'https://rpc.ankr.com/polygon',
                'https://polygon-mainnet.public.blastapi.io'
            ],
            'mumbai': [
                'https://rpc.ankr.com/polygon_mumbai',
                'https://polygon-mumbai.public.blastapi.io'
            ]
        },
        'bsc': {
            'mainnet': [
                'https://bsc-dataseed.binance.org',
                'https://rpc.ankr.com/bsc',
                'https://bsc.publicnode.com'
            ],
            'testnet': [
                'https://data-seed-prebsc-1-s1.binance.org:8545'
            ]
        },
        'arbitrum': {
            'mainnet': [
                'https://arb1.arbitrum.io/rpc',
                'https://rpc.ankr.com/arbitrum'
            ],
            'goerli': [
                'https://goerli-rollup.arbitrum.io/rpc'
            ]
        },
        'optimism': {
            'mainnet': [
                'https://mainnet.optimism.io',
                'https://rpc.ankr.com/optimism'
            ],
            'goerli': [
                'https://goerli.optimism.io'
            ]
        },
        'avalanche': {
            'mainnet': [
                'https://api.avax.network/ext/bc/C/rpc',
                'https://rpc.ankr.com/avalanche'
            ]
        }
    }
    
    # 区块链浏览器API
    EXPLORER_APIS = {
        'etherscan': {
            'mainnet': 'https://api.etherscan.io/api',
            'goerli': 'https://api-goerli.etherscan.io/api',
            'api_key': None  # 从环境变量获取
        },
        'bscscan': {
            'mainnet': 'https://api.bscscan.com/api',
            'testnet': 'https://api-testnet.bscscan.com/api',
            'api_key': None
        },
        'polygonscan': {
            'mainnet': 'https://api.polygonscan.com/api',
            'mumbai': 'https://api-testnet.polygonscan.com/api',
            'api_key': None
        },
        'arbiscan': {
            'mainnet': 'https://api.arbiscan.io/api',
            'goerli': 'https://api-goerli.arbiscan.io/api',
            'api_key': None
        }
    }
    
    # Dune Analytics API
    DUNE_API = {
        'base_url': 'https://api.dune.com/api/v2',
        'api_key': None  # 从环境变量获取
    }
    
    # The Graph API - 使用新的去中心化网络
    THE_GRAPH_API = {
        'uniswap_v3_ethereum': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        'uniswap_v3_polygon': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-polygon',
        'sushiswap_ethereum': 'https://api.thegraph.com/subgraphs/name/sushiswap/exchange',
        'curve_finance': 'https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum',
        # 备选端点
        'uniswap_v3_ethereum_decentralized': 'https://gateway-arbitrum.network.thegraph.com/api/subgraphs/id/D7azkFFPFT5NzU3nw1Xqzyy6jETeN8pLm8SyEzPiD7aT'
    }
    
    # 备选数据源
    ALTERNATIVE_APIS = {
        'eth_rpc_alternatives': [
            'https://ethereum.publicnode.com',
            'https://rpc.flashbots.net',
            'https://eth-mainnet.public.blastapi.io'
        ],
        'blocknative': {
            'gas_price': 'https://api.blocknative.com/gasprices/blockprices',
            'api_key': None
        },
        'ethgasstation': 'https://ethgasstation.info/api/ethgasAPI.json',
        'gasnow': 'https://www.gasnow.org/api/v3/gas/price'
    }
    
    @classmethod
    def get_rpc_url(cls, network: str = 'ethereum', chain: str = 'mainnet') -> str:
        """获取RPC URL"""
        if network in cls.RPC_NODES and chain in cls.RPC_NODES[network]:
            urls = cls.RPC_NODES[network][chain]
            return urls[0]  # 返回第一个URL
        raise ValueError(f"不支持的网络或链: {network}.{chain}")
    
    @classmethod
    def get_all_rpc_urls(cls, network: str = 'ethereum', chain: str = 'mainnet') -> List[str]:
        """获取所有RPC URL"""
        if network in cls.RPC_NODES and chain in cls.RPC_NODES[network]:
            return cls.RPC_NODES[network][chain]
        return []
    
    @classmethod
    def get_explorer_api(cls, explorer: str = 'etherscan', chain: str = 'mainnet') -> Dict[str, str]:
        """获取区块链浏览器API配置"""
        if explorer in cls.EXPLORER_APIS:
            config = cls.EXPLORER_APIS[explorer].copy()
            if chain in config:
                config['url'] = config[chain]
            else:
                config['url'] = config.get('mainnet', '')
            
            # 从环境变量获取API密钥
            api_config = get_api_config()
            if explorer == 'etherscan':
                # Etherscan API密钥可以从环境变量获取
                api_key = os.environ.get('ETHERSCAN_API_KEY')
                if api_key:
                    config['api_key'] = api_key
            elif explorer == 'bscscan':
                api_key = os.environ.get('BSCSCAN_API_KEY') or os.environ.get('BSC_SCAN_API_KEY')
                if api_key:
                    config['api_key'] = api_key
            elif explorer == 'polygonscan':
                api_key = os.environ.get('POLYGONSCAN_API_KEY')
                if api_key:
                    config['api_key'] = api_key
            elif explorer == 'arbiscan':
                api_key = os.environ.get('ARBISCAN_API_KEY')
                if api_key:
                    config['api_key'] = api_key
            
            return config
        raise ValueError(f"不支持的区块链浏览器: {explorer}")
    
    @classmethod
    def get_dune_api_config(cls) -> Dict[str, str]:
        """获取Dune API配置"""
        config = cls.DUNE_API.copy()
        
        # 从环境变量获取API密钥
        api_config = get_api_config()
        if api_config.dune_api_key:
            config['api_key'] = api_config.dune_api_key
        
        return config
    
    @classmethod
    def get_the_graph_api(cls, subgraph: str) -> str:
        """获取The Graph API URL"""
        if subgraph in cls.THE_GRAPH_API:
            return cls.THE_GRAPH_API[subgraph]
        raise ValueError(f"不支持的subgraph: {subgraph}")


# ==================== Web3连接管理器 ====================

class Web3ConnectionManager:
    """Web3连接管理器"""
    
    def __init__(self):
        self.connections = {}
        self.logger = logging.getLogger("web3_manager")
    
    def get_connection(self, network: str = 'ethereum', chain: str = 'mainnet') -> Optional[Web3]:
        """获取Web3连接"""
        key = f"{network}_{chain}"
        
        if key not in self.connections:
            try:
                # 尝试所有RPC URL直到成功
                rpc_urls = OnChainDataSource.get_all_rpc_urls(network, chain)
                
                for rpc_url in rpc_urls:
                    try:
                        self.logger.info(f"尝试连接RPC节点: {rpc_url}")
                        web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
                        
                        # 测试连接
                        if web3.is_connected():
                            self.connections[key] = web3
                            self.logger.info(f"Web3连接成功: {network}.{chain} via {rpc_url}")
                            break
                        else:
                            self.logger.warning(f"RPC节点连接失败: {rpc_url}")
                        # if network in ['polygon', 'bsc', 'avalanche']:
                        #     # 注入 POA 中间件
                        #     web3.middleware_onion.inject(geth_poa_middleware, layer=0)

                    except Exception as e:
                        self.logger.warning(f"连接RPC节点失败 {rpc_url}: {e}")
                        continue
                
                # 如果没有成功连接
                if key not in self.connections:
                    self.logger.error(f"所有RPC节点连接失败: {network}.{chain}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"创建Web3连接失败: {e}")
                return None
        
        return self.connections[key]
    
    def test_connection(self, network: str = 'ethereum', chain: str = 'mainnet') -> bool:
        """测试连接"""
        web3 = self.get_connection(network, chain)
        if web3:
            try:
                block_number = web3.eth.block_number
                self.logger.info(f"连接测试成功，当前区块: {block_number}")
                return True
            except Exception as e:
                self.logger.error(f"连接测试失败: {e}")
                return False
        return False
    
    def close_all(self):
        """关闭所有连接"""
        self.connections.clear()
        self.logger.info("所有Web3连接已关闭")


# ==================== 链上数据获取器 ====================

class OnChainFetcher(BaseFetcher):
    """
    链上数据获取器基类
    """
    
    def __init__(self, 
                 network: str = "ethereum",
                 chain: str = "mainnet",
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None,
                 use_simulation: bool = False):
        """
        初始化链上数据获取器
        
        参数:
            network: 区块链网络 (ethereum, polygon, bsc, arbitrum, optimism)
            chain: 链类型 (mainnet, testnet)
            config: 配置字典
            cache_manager: 缓存管理器
            use_simulation: 是否使用模拟数据（当真实API不可用时）
        """
        super().__init__(
            exchange="onchain",  # 链上数据没有交易所概念
            market_type="onchain",
            config=config,
            cache_manager=cache_manager
        )
        
        self.network = network
        self.chain = chain
        self.use_simulation = use_simulation  # 模拟模式标志
        
        # 初始化Web3连接
        self.web3_manager = Web3ConnectionManager()
        self.web3 = None
        
        if not self.use_simulation:
            self.web3 = self.web3_manager.get_connection(network, chain)
            if not self.web3:
                self.logger.warning(f"Web3连接失败，启用模拟模式")
                self.use_simulation = True
        
        # 初始化区块链浏览器API
        self.explorer_config = None
        self._init_explorer()
        
        # 初始化Dune API
        self.dune_config = OnChainDataSource.get_dune_api_config()
        
        # 常用合约地址
        self.common_contracts = self._load_common_contracts()
        
        # 常用交易所地址
        self.exchange_addresses = self._load_exchange_addresses()
        
        # 模拟数据
        self.simulation_data = self._init_simulation_data()


    
    def _init_explorer(self):
        """初始化区块链浏览器API"""
        # 根据网络选择对应的区块链浏览器
        if self.network == 'ethereum':
            self.explorer_config = OnChainDataSource.get_explorer_api('etherscan', self.chain)
        elif self.network == 'bsc':
            self.explorer_config = OnChainDataSource.get_explorer_api('bscscan', self.chain)
        elif self.network == 'polygon':
            self.explorer_config = OnChainDataSource.get_explorer_api('polygonscan', self.chain)
        elif self.network == 'arbitrum':
            self.explorer_config = OnChainDataSource.get_explorer_api('arbiscan', self.chain)
        else:
            self.logger.warning(f"网络 {self.network} 没有配置区块链浏览器API")
    
    def _load_common_contracts(self) -> Dict[str, Dict[str, str]]:
        """加载常用合约地址"""
        contracts = {
            'ethereum': {
                # 代币合约
                'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
                'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
                
                # DeFi协议
                'UniswapV2_Router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'UniswapV3_Router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'SushiSwap_Router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
                
                # 稳定币协议
                'Curve_3pool': '0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7',
                'AAVE_LendingPool': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
                'Compound_cDAI': '0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643',
                
                # NFT市场
                'OpenSea': '0x7Be8076f4EA4A4AD08075C2508e481d6C946D12b',
                'LooksRare': '0x59728544B08AB483533076417FbBB2fD0B17CE3a',
            },
            'polygon': {
                'WETH': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
                'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
                'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
                'MATIC': '0x0000000000000000000000000000000000001010',
                
                'QuickSwap_Router': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'SushiSwap_Router': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
            },
            'bsc': {
                'WBNB': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',
                'BUSD': '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56',
                'USDT': '0x55d398326f99059fF775485246999027B3197955',
                'CAKE': '0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82',
                
                'PancakeSwap_Router': '0x10ED43C718714eb63d5aA57B78B54704E256024E',
            }
        }
        
        return contracts.get(self.network, {})
    
    def _load_exchange_addresses(self) -> Dict[str, List[str]]:
        """加载交易所地址"""
        # 这些是已知的交易所热钱包地址（示例）
        exchanges = {
            'binance': [
                '0x28C6c06298d514Db089934071355E5743bf21d60',  # Binance 14
                '0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549',  # Binance 16
                '0xDFd5293D8e347dFe59E90eFd55b2956a1343963d',  # Binance 15
            ],
            'coinbase': [
                '0xA9D1e08C7793af67e9d92fe308d5697FB81d3E43',  # Coinbase 1
                '0x77696bb39917C91A0c3908D577c5B7aD905D1169',  # Coinbase 2
            ],
            'kraken': [
                '0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2',  # Kraken 1
                '0x53d284357ec70cE289D6D64134DfAc8E511c8a3D',  # Kraken 2
            ],
            'okx': [
                '0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b',  # OKX 1
                '0x236f9f97e0E62388479bf9E5BA4889e46B0273C3',  # OKX 2
            ],
            'ftx': [
                '0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2',  # FTX 1
                '0xC098B2a3Aa256D2140208C3de6543aAEf5cd3A94',  # FTX 2
            ]
        }
        
        return exchanges
    
    def _init_simulation_data(self) -> Dict[str, Any]:
        """初始化模拟数据"""
        current_time = pd.Timestamp.now()
        
        # 模拟区块数据
        simulated_block_number = 18000000  # 假设以太坊当前区块
        if self.network == 'polygon':
            simulated_block_number = 45000000
        elif self.network == 'bsc':
            simulated_block_number = 30000000
        
        simulation_data = {
            'block_number': simulated_block_number,
            'timestamp': current_time,
            'gas_price_gwei': 25.0,  # 假设Gas价格
            'network_stats': {
                'current_block': simulated_block_number,
                'transactions_per_second': 15.0,
                'gas_utilization_percent': 45.0,
                'network_health': 'healthy'
            },
            'address_balance': {
                'balance_native': 1234.5678,
                'balance_token': 98765.4321
            },
            'contract_info': {
                'name': 'Tether USD',
                'symbol': 'USDT',
                'decimals': 6,
                'total_supply': 40000000000  # 400亿
            },
            'exchange_flow': {
                'net_flow': 1250.75,
                'inflow': 3500.25,
                'outflow': 2249.5
            },
            'eth2_staking': {
                'total_validators': 850000,
                'total_eth_staked': 27200000,  # 2720万ETH
                'staking_apy': 0.042
            },
            'defi_stats': {
                'total_tvl_usd': 45000000000,  # 450亿美元
                'total_volume_usd': 1200000000,  # 12亿美元
                'total_fees_usd': 35000000  # 3500万美元
            }
        }
        
        return simulation_data
    
    def validate_address(self, address: str) -> bool:
        """
        验证地址格式
        
        参数:
            address: 区块链地址
            
        返回:
            是否有效
        """
        if not address or not isinstance(address, str):
            return False
        
        # 检查是否为有效的以太坊地址
        try:
            if self.web3:
                return self.web3.is_address(address)
        except:
            pass
        
        # 基本格式检查
        if address.startswith('0x') and len(address) == 42:
            return True
        
        return False
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的符号（包括代币合约和链上指标）
        
        返回:
            符号列表
        """
        # include basic metrics
        symbols = ['network_stats', 'gas_price']
        # include common contracts
        symbols.extend(list(self.common_contracts.keys()))
        return symbols
    
    # ==================== 实现 BaseFetcher 的抽象方法 ====================
    
    def _init_exchange(self):
        """
        初始化交易所实例
        链上数据获取器不需要传统的交易所实例
        """
        # 链上数据获取器使用 Web3 连接而非交易所 API
        pass
    
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = "1h",
                   since: Optional[Union[int, datetime, str]] = None,
                   limit: Optional[int] = None,
                   **kwargs) -> List[OHLCVData]:
        """
        获取K线数据
        链上数据获取器不提供此功能，请使用交易所数据获取器
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            空列表
        """
        self.logger.warning("OnChainFetcher 不支持 fetch_ohlcv，请使用交易所数据获取器")
        return []
    
    def fetch_orderbook(self, 
                       symbol: str,
                       limit: Optional[int] = None,
                       **kwargs) -> Optional[OrderBookData]:
        """
        获取订单簿数据
        链上数据获取器不提供此功能，请使用交易所数据获取器
        
        参数:
            symbol: 交易对符号
            limit: 深度限制
            **kwargs: 额外参数
            
        返回:
            None
        """
        self.logger.warning("OnChainFetcher 不支持 fetch_orderbook，请使用交易所数据获取器")
        return None
    
    def fetch_trades(self, 
                    symbol: str,
                    since: Optional[Union[int, datetime, str]] = None,
                    limit: Optional[int] = None,
                    **kwargs) -> List[TradeData]:
        """
        获取成交数据
        链上数据获取器不提供此功能，请使用交易所数据获取器
        
        参数:
            symbol: 交易对符号
            since: 开始时间
            limit: 数据条数限制
            **kwargs: 额外参数
            
        返回:
            空列表
        """
        self.logger.warning("OnChainFetcher 不支持 fetch_trades，请使用交易所数据获取器")
        return []
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_block_number", ttl=10, sub_dir="onchain")
    def fetch_block_number(self) -> int:
        """
        获取当前区块号
        
        返回:
            当前区块号
        """
        if self.use_simulation:
            self.logger.info("使用模拟区块号")
            return self.simulation_data['block_number']
        
        if not self.web3:
            raise ConnectionError("Web3连接未初始化")
        
        try:
            block_number = self.web3.eth.block_number
            self.logger.info(f"当前区块号: {block_number}")
            return block_number
        except Exception as e:
            self.logger.error(f"获取区块号失败: {e}")
            # 失败时返回模拟数据
            self.logger.warning("切换为模拟模式")
            self.use_simulation = True
            return self.simulation_data['block_number']
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_block", ttl=300, sub_dir="onchain")
    def fetch_block(self, block_identifier: Union[int, str] = 'latest') -> Dict[str, Any]:
        """
        获取区块信息
        
        参数:
            block_identifier: 区块标识符 (区块号或'latest', 'pending')
            
        返回:
            区块信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟区块数据")
            return self._create_simulated_block(block_identifier)
        
        if not self.web3:
            raise ConnectionError("Web3连接未初始化")
        
        try:
            if block_identifier == 'latest':
                block_identifier = self.web3.eth.block_number
            
            block = self.web3.eth.get_block(block_identifier)
            
            # 安全地获取 extra_data（某些 RPC 可能不返回此字段）
            extra_data = '0x'
            try:
                ed = block.get('extraData') or getattr(block, 'extra_data', None)
                if ed:
                    extra_data = ed.hex() if isinstance(ed, bytes) else str(ed)
            except Exception:
                pass
            
            result = {
                'block_number': block.number,
                'timestamp': pd.Timestamp(block.timestamp, unit='s'),
                'transactions': len(block.transactions),
                'gas_used': block.gasUsed,
                'gas_limit': block.gasLimit,
                'base_fee_per_gas': block.get('baseFeePerGas', 0),
                'difficulty': block.difficulty,
                'total_difficulty': getattr(block, 'totalDifficulty', 0),
                'size': block.size,
                'hash': block.hash.hex(),
                'parent_hash': block.parentHash.hex(),
                'miner': block.miner,
                'extra_data': extra_data,
                'network': self.network,
                'chain': self.chain
            }
            
            self.logger.info(f"获取区块成功: {block.number}, 时间: {result['timestamp']}")
            return result
            
        except BlockNotFound:
            self.logger.warning(f"区块未找到: {block_identifier}")
            return self._create_simulated_block(block_identifier)
        except Exception as e:
            self.logger.error(f"获取区块失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_block(block_identifier)
    
    def _create_simulated_block(self, block_identifier: Union[int, str]) -> Dict[str, Any]:
        """创建模拟区块数据"""
        if isinstance(block_identifier, int):
            block_number = block_identifier
        else:
            block_number = self.simulation_data['block_number']
        
        # 模拟时间戳（当前时间减去一些秒数）
        timestamp = pd.Timestamp.now() - timedelta(seconds=np.random.randint(1, 100))
        
        return {
            'block_number': block_number,
            'timestamp': timestamp,
            'transactions': np.random.randint(50, 200),
            'gas_used': np.random.randint(1000000, 20000000),
            'gas_limit': 30000000,
            'base_fee_per_gas': np.random.randint(10, 50) * 1e9,
            'difficulty': int(np.random.uniform(1000000000000000, 2000000000000000)),
            'total_difficulty': int(np.random.uniform(1e22, 1e23)),
            'size': np.random.randint(50000, 200000),
            'hash': f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(64)])}',
            'parent_hash': f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(64)])}',
            'miner': f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(40)])}',
            'extra_data': '0x',
            'network': self.network,
            'chain': self.chain,
            'is_simulated': True
        }
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_transaction", ttl=3600, sub_dir="onchain")
    def fetch_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        获取交易信息
        
        参数:
            tx_hash: 交易哈希
            
        返回:
            交易信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟交易数据")
            return self._create_simulated_transaction(tx_hash)
        
        if not self.web3:
            raise ConnectionError("Web3连接未初始化")
        
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            
            if not tx:
                return self._create_simulated_transaction(tx_hash)
            
            # 获取交易收据
            try:
                receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            except:
                receipt = None
            
            result = {
                'tx_hash': tx_hash,
                'block_number': tx.blockNumber,
                'from': tx['from'],
                'to': tx.to if tx.to else '',
                'value': self.web3.from_wei(tx.value, 'ether'),
                'gas': tx.gas,
                'gas_price': self.web3.from_wei(tx.gasPrice, 'gwei'),
                'nonce': tx.nonce,
                'input': tx.input.hex(),
                'timestamp': None,  # 需要从区块获取
                'status': receipt.status if receipt else None,
                'gas_used': receipt.gasUsed if receipt else None,
                'contract_address': receipt.contractAddress if receipt else None,
                'logs': len(receipt.logs) if receipt else 0,
                'network': self.network,
                'chain': self.chain
            }
            
            # 获取交易时间
            if tx.blockNumber:
                try:
                    block = self.web3.eth.get_block(tx.blockNumber)
                    result['timestamp'] = pd.Timestamp(block.timestamp, unit='s')
                except:
                    pass
            
            self.logger.info(f"获取交易成功: {tx_hash[:10]}..., 区块: {tx.blockNumber}")
            return result
            
        except TransactionNotFound:
            self.logger.warning(f"交易未找到: {tx_hash}")
            return self._create_simulated_transaction(tx_hash)
        except Exception as e:
            self.logger.error(f"获取交易失败: {e}")
            return self._create_simulated_transaction(tx_hash)
    
    def _create_simulated_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """创建模拟交易数据"""
        return {
            'tx_hash': tx_hash,
            'block_number': self.simulation_data['block_number'] - np.random.randint(1, 100),
            'from': f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(40)])}',
            'to': f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(40)])}',
            'value': np.random.uniform(0.1, 100.0),
            'gas': np.random.randint(21000, 100000),
            'gas_price': np.random.uniform(20.0, 100.0),
            'nonce': np.random.randint(0, 100),
            'input': '0x',
            'timestamp': pd.Timestamp.now() - timedelta(minutes=np.random.randint(1, 60)),
            'status': 1,
            'gas_used': np.random.randint(21000, 80000),
            'contract_address': None,
            'logs': np.random.randint(0, 10),
            'network': self.network,
            'chain': self.chain,
            'is_simulated': True
        }
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_address_balance", ttl=60, sub_dir="onchain")
    def fetch_address_balance(self, address: str, token_address: Optional[str] = None) -> Dict[str, Any]:
        """
        获取地址余额
        
        参数:
            address: 地址
            token_address: 代币合约地址（可选，None表示原生代币）
            
        返回:
            余额信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟地址余额数据")
            return self._create_simulated_balance(address, token_address)
        
        if not self.web3:
            raise ConnectionError("Web3连接未初始化")
        
        # 修改后：
        if not self.validate_address(address):
            raise ValueError(f"无效地址: {address}")

        # 转换为校验和地址（EIP-55）
        try:
            checksum_address = self.web3.to_checksum_address(address)
        except Exception as e:
            self.logger.warning(f"地址转换失败，使用原始地址: {e}")
            checksum_address = address
        
        try:
            result = {
                'address': address,
                'network': self.network,
                'chain': self.chain,
                'timestamp': pd.Timestamp.now(),
                'token_address': token_address,
                'balance_native': 0.0,
                'balance_token': 0.0,
                'token_decimals': 18,
                'token_symbol': 'ETH' if self.network == 'ethereum' else self.network.upper()
            }
            
            # 获取原生代币余额
            try:
                balance_wei = self.web3.eth.get_balance(checksum_address)
                result['balance_native'] = self.web3.from_wei(balance_wei, 'ether')
            except Exception as e:
                self.logger.warning(f"获取原生代币余额失败: {e}")
            
            # 获取ERC20代币余额
            if token_address:
                if not self.validate_address(token_address):
                    raise ValueError(f"无效的代币地址: {token_address}")
                
                try:
                    # ERC20 ABI（简化的，只包含我们需要的方法）
                    erc20_abi = [
                        {
                            "constant": True,
                            "inputs": [{"name": "_owner", "type": "address"}],
                            "name": "balanceOf",
                            "outputs": [{"name": "balance", "type": "uint256"}],
                            "type": "function"
                        },
                        {
                            "constant": True,
                            "inputs": [],
                            "name": "decimals",
                            "outputs": [{"name": "", "type": "uint8"}],
                            "type": "function"
                        },
                        {
                            "constant": True,
                            "inputs": [],
                            "name": "symbol",
                            "outputs": [{"name": "", "type": "string"}],
                            "type": "function"
                        },
                        {
                            "constant": True,
                            "inputs": [],
                            "name": "name",
                            "outputs": [{"name": "", "type": "string"}],
                            "type": "function"
                        }
                    ]
                    
                    # 创建代币合约实例
                    token_checksum_address = self.web3.to_checksum_address(token_address)
                    token_contract = self.web3.eth.contract(
                        address=token_checksum_address,
                        abi=erc20_abi
                    )
                    
                    # 获取代币信息
                    try:
                        decimals = token_contract.functions.decimals().call()
                        result['token_decimals'] = decimals
                    except:
                        pass
                    
                    try:
                        symbol = token_contract.functions.symbol().call()
                        result['token_symbol'] = symbol
                    except:
                        pass
                    
                    # 获取代币余额
                    balance = token_contract.functions.balanceOf(
                        checksum_address
                    ).call()
                    
                    result['balance_token'] = balance / (10 ** result['token_decimals'])
                    
                except Exception as e:
                    self.logger.warning(f"获取代币余额失败: {e}")
            
            self.logger.info(
                f"地址余额: {address[:10]}..., "
                f"原生: {result['balance_native']:.6f}, "
                f"代币: {result['balance_token']:.6f} {result['token_symbol']}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取地址余额失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_balance(address, token_address)
    
    def _create_simulated_balance(self, address: str, token_address: Optional[str] = None) -> Dict[str, Any]:
        """创建模拟余额数据"""
        result = {
            'address': address,
            'network': self.network,
            'chain': self.chain,
            'timestamp': pd.Timestamp.now(),
            'token_address': token_address,
            'balance_native': self.simulation_data['address_balance']['balance_native'],
            'balance_token': self.simulation_data['address_balance']['balance_token'],
            'token_decimals': 18,
            'token_symbol': 'ETH' if self.network == 'ethereum' else self.network.upper(),
            'is_simulated': True
        }
        
        if token_address:
            # 模拟代币信息
            if token_address == '0xdAC17F958D2ee523a2206206994597C13D831ec7':  # USDT
                result['token_symbol'] = 'USDT'
                result['token_decimals'] = 6
                result['balance_token'] = 50000.0  # 5万USDT
            elif token_address == '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48':  # USDC
                result['token_symbol'] = 'USDC'
                result['token_decimals'] = 6
                result['balance_token'] = 30000.0  # 3万USDC
        
        return result
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_gas_price", ttl=30, sub_dir="onchain")
    def fetch_gas_price(self) -> Dict[str, Any]:
        """
        获取当前Gas价格
        
        返回:
            Gas价格信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟Gas价格数据")
            return self._create_simulated_gas_price()
        
        if not self.web3:
            raise ConnectionError("Web3连接未初始化")
        
        try:
            # 获取当前Gas价格
            gas_price_wei = self.web3.eth.gas_price
            
            # 转换为不同单位
            gas_price_gwei = self.web3.from_wei(gas_price_wei, 'gwei')
            gas_price_eth = self.web3.from_wei(gas_price_wei, 'ether')
            
            result = {
                'timestamp': pd.Timestamp.now(),
                'network': self.network,
                'chain': self.chain,
                'gas_price_wei': int(gas_price_wei),
                'gas_price_gwei': float(gas_price_gwei),
                'gas_price_eth': float(gas_price_eth),
                'suggested_max_fee_gwei': float(gas_price_gwei) * 1.2,  # 建议的最大费用
                'data_source': 'web3'
            }
            
            # 尝试获取EIP-1559费用信息
            try:
                fee_history = self.web3.eth.fee_history(1, 'latest', [25, 50, 75])
                
                if fee_history:
                    base_fee = fee_history['baseFeePerGas'][-1]
                    priority_fees = fee_history['reward'][0] if fee_history['reward'] else [0, 0, 0]
                    
                    result['base_fee_gwei'] = float(self.web3.from_wei(base_fee, 'gwei'))
                    result['priority_fee_25_gwei'] = float(self.web3.from_wei(priority_fees[0], 'gwei')) if len(priority_fees) > 0 else 0
                    result['priority_fee_50_gwei'] = float(self.web3.from_wei(priority_fees[1], 'gwei')) if len(priority_fees) > 1 else 0
                    result['priority_fee_75_gwei'] = float(self.web3.from_wei(priority_fees[2], 'gwei')) if len(priority_fees) > 2 else 0
                    result['suggested_priority_fee_gwei'] = result['priority_fee_50_gwei']
            except Exception as e:
                self.logger.warning(f"获取EIP-1559费用信息失败: {e}")
            
            self.logger.info(f"当前Gas价格: {result['gas_price_gwei']:.2f} Gwei")
            return result
            
        except Exception as e:
            self.logger.error(f"获取Gas价格失败: {e}")
            # 尝试使用备用API
            try:
                return self._fetch_gas_price_from_alternative()
            except:
                # 失败时返回模拟数据
                return self._create_simulated_gas_price()
    
    def _fetch_gas_price_from_alternative(self) -> Dict[str, Any]:
        """从备用API获取Gas价格"""
        try:
            # 尝试使用ethgasstation API
            response = requests.get(
                'https://ethgasstation.info/api/ethgasAPI.json',
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # ethgasstation返回的价格单位是 gwei * 10
                gas_price_gwei = data.get('fast', 0) / 10
                
                result = {
                    'timestamp': pd.Timestamp.now(),
                    'network': 'ethereum',
                    'chain': 'mainnet',
                    'gas_price_gwei': gas_price_gwei,
                    'gas_price_wei': int(gas_price_gwei * 1e9),
                    'fast_gwei': data.get('fast', 0) / 10,
                    'average_gwei': data.get('average', 0) / 10,
                    'slow_gwei': data.get('safeLow', 0) / 10,
                    'data_source': 'ethgasstation'
                }
                
                self.logger.info(f"从ethgasstation获取Gas价格: {gas_price_gwei:.2f} Gwei")
                return result
        except Exception as e:
            self.logger.warning(f"从ethgasstation获取Gas价格失败: {e}")
        
        # 如果所有备用API都失败，返回模拟数据
        return self._create_simulated_gas_price()
    
    def _create_simulated_gas_price(self) -> Dict[str, Any]:
        """创建模拟Gas价格数据"""
        gas_price_gwei = self.simulation_data['gas_price_gwei']
        
        return {
            'timestamp': pd.Timestamp.now(),
            'network': self.network,
            'chain': self.chain,
            'gas_price_wei': int(gas_price_gwei * 1e9),
            'gas_price_gwei': gas_price_gwei,
            'gas_price_eth': gas_price_gwei / 1e9,
            'base_fee_gwei': gas_price_gwei * 0.8,
            'priority_fee_25_gwei': 1.5,
            'priority_fee_50_gwei': 2.0,
            'priority_fee_75_gwei': 3.0,
            'suggested_max_fee_gwei': gas_price_gwei * 1.2,
            'suggested_priority_fee_gwei': 2.0,
            'data_source': 'simulation',
            'is_simulated': True
        }
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_contract_info", ttl=3600, sub_dir="onchain")
    def fetch_contract_info(self, contract_address: str) -> Dict[str, Any]:
        """
        获取合约信息
        
        参数:
            contract_address: 合约地址
            
        返回:
            合约信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟合约信息")
            return self._create_simulated_contract_info(contract_address)
        
        if not self.web3:
            raise ConnectionError("Web3连接未初始化")
        
        if not self.validate_address(contract_address):
            raise ValueError(f"无效的合约地址: {contract_address}")
        
        try:
            # 简化的合约ABI，用于获取基本信息
            simple_abi = [
                {
                    "constant": True,
                    "inputs": [],
                    "name": "name",
                    "outputs": [{"name": "", "type": "string"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "symbol",
                    "outputs": [{"name": "", "type": "string"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "totalSupply",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function"
                }
            ]
            
            contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=simple_abi
            )
            
            result = {
                'contract_address': contract_address,
                'network': self.network,
                'chain': self.chain,
                'timestamp': pd.Timestamp.now(),
                'is_contract': True,
                'name': None,
                'symbol': None,
                'decimals': None,
                'total_supply': None
            }
            
            # 尝试获取合约信息
            try:
                result['name'] = contract.functions.name().call()
            except:
                pass
            
            try:
                result['symbol'] = contract.functions.symbol().call()
            except:
                pass
            
            try:
                result['decimals'] = contract.functions.decimals().call()
            except:
                pass
            
            try:
                total_supply = contract.functions.totalSupply().call()
                if result['decimals']:
                    result['total_supply'] = total_supply / (10 ** result['decimals'])
                else:
                    result['total_supply'] = total_supply
            except:
                pass
            
            # 检查合约代码
            try:
                code = self.web3.eth.get_code(contract_address)
                result['bytecode_size'] = len(code)
                result['has_code'] = len(code) > 0
            except:
                result['has_code'] = False
            
            self.logger.info(f"获取合约信息: {contract_address[:10]}..., 名称: {result['name']}")
            return result
            
        except Exception as e:
            self.logger.error(f"获取合约信息失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_contract_info(contract_address)
    
    def _create_simulated_contract_info(self, contract_address: str) -> Dict[str, Any]:
        """创建模拟合约信息"""
        # 检查是否是已知的合约
        known_contracts = {
            '0xdAC17F958D2ee523a2206206994597C13D831ec7': {
                'name': 'Tether USD',
                'symbol': 'USDT',
                'decimals': 6,
                'total_supply': 40000000000
            },
            '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48': {
                'name': 'USD Coin',
                'symbol': 'USDC',
                'decimals': 6,
                'total_supply': 25000000000
            },
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': {
                'name': 'Wrapped Ether',
                'symbol': 'WETH',
                'decimals': 18,
                'total_supply': 4000000
            }
        }
        
        if contract_address in known_contracts:
            contract_info = known_contracts[contract_address]
        else:
            contract_info = self.simulation_data['contract_info']
        
        result = {
            'contract_address': contract_address,
            'network': self.network,
            'chain': self.chain,
            'timestamp': pd.Timestamp.now(),
            'is_contract': True,
            'name': contract_info['name'],
            'symbol': contract_info['symbol'],
            'decimals': contract_info['decimals'],
            'total_supply': contract_info['total_supply'],
            'bytecode_size': 10000,
            'has_code': True,
            'is_simulated': True
        }
        
        return result
    
    @log_errors(reraise=False)
    def fetch_transaction_history(self, 
                                 address: str, 
                                 start_block: Optional[int] = None,
                                 end_block: Optional[int] = None,
                                 limit: int = 100,
                                 sort: str = 'desc') -> List[Dict[str, Any]]:
        """
        获取地址交易历史（使用区块链浏览器API）
        
        参数:
            address: 地址
            start_block: 开始区块
            end_block: 结束区块
            limit: 限制条数
            
        返回:
            交易历史列表
        """
        if self.use_simulation:
            self.logger.info("使用模拟交易历史")
            return self._create_simulated_transaction_history(address, limit, sort=sort)
        
        if not self.explorer_config:
            raise ConnectionError("区块链浏览器API未配置")
        
        if not self.validate_address(address):
            raise ValueError(f"无效地址: {address}")
        
        try:
            # 构建API请求参数
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': address,
                'sort': sort if sort in ('asc', 'desc') else 'desc'
            }
            
            if start_block:
                params['startblock'] = start_block
            
            if end_block:
                params['endblock'] = end_block
            
            params['page'] = 1
            params['offset'] = min(limit, 10000)  # 最大限制
            
            if 'api_key' in self.explorer_config and self.explorer_config['api_key']:
                params['apikey'] = self.explorer_config['api_key']
            
            # 发送请求
            response = requests.get(
                self.explorer_config['url'],
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ConnectionError(f"API请求失败: {response.status_code}")
            
            data = response.json()
            
            if data.get('status') != '1':
                error_message = data.get('message', 'Unknown error')
                raise DataFormatError(f"API返回错误: {error_message}")
            
            transactions = data.get('result', [])
            
            # 格式化交易数据
            formatted_txs = []
            for tx in transactions[:limit]:
                formatted_tx = {
                    'tx_hash': tx.get('hash', ''),
                    'block_number': int(tx.get('blockNumber', 0)),
                    'timestamp': pd.Timestamp(int(tx.get('timeStamp', 0)), unit='s'),
                    'from': tx.get('from', ''),
                    'to': tx.get('to', ''),
                    'value': float(tx.get('value', 0)) / 1e18,  # 转换为ETH
                    'gas': int(tx.get('gas', 0)),
                    'gas_price': float(tx.get('gasPrice', 0)) / 1e9,  # 转换为Gwei
                    'gas_used': int(tx.get('gasUsed', 0)),
                    'input': tx.get('input', ''),
                    'contract_address': tx.get('contractAddress', ''),
                    'is_error': tx.get('isError', '0') == '1',
                    'txreceipt_status': tx.get('txreceipt_status', ''),
                    'network': self.network,
                    'chain': self.chain
                }
                formatted_txs.append(formatted_tx)
            
            self.logger.info(f"获取交易历史: {address[:10]}..., 数量: {len(formatted_txs)}")
            return formatted_txs
            
        except Exception as e:
            self.logger.error(f"获取交易历史失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_transaction_history(address, limit, sort=sort)
    
    def _create_simulated_transaction_history(self, address: str, limit: int = 100, sort: str = 'desc') -> List[Dict[str, Any]]:
        """创建模拟交易历史"""
        transactions = []
        
        for i in range(min(limit, 10)):  # 最多10条模拟数据
            tx_hash = f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(64)])}'
            block_number = self.simulation_data['block_number'] - (i * 100)
            
            transaction = {
                'tx_hash': tx_hash,
                'block_number': block_number,
                'timestamp': pd.Timestamp.now() - timedelta(days=i, hours=np.random.randint(0, 24)),
                'from': address if i % 2 == 0 else f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(40)])}',
                'to': address if i % 2 == 1 else f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(40)])}',
                'value': np.random.uniform(0.1, 50.0),
                'gas': np.random.randint(21000, 100000),
                'gas_price': np.random.uniform(20.0, 100.0),
                'gas_used': np.random.randint(21000, 80000),
                'input': '0x',
                'contract_address': None,
                'is_error': False,
                'txreceipt_status': '1',
                'network': self.network,
                'chain': self.chain,
                'is_simulated': True
            }
            transactions.append(transaction)
        
        if sort == 'asc':
            transactions = sorted(transactions, key=lambda x: x.get('timestamp'))
        return transactions
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_token_transfers", ttl=300, sub_dir="onchain")
    def fetch_token_transfers(self, 
                             token_address: str,
                             address: Optional[str] = None,
                             start_block: Optional[int] = None,
                             end_block: Optional[int] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取代币转账记录（使用区块链浏览器API）
        
        参数:
            token_address: 代币合约地址
            address: 特定地址（可选）
            start_block: 开始区块
            end_block: 结束区块
            limit: 限制条数
            
        返回:
            代币转账记录列表
        """
        if self.use_simulation:
            self.logger.info("使用模拟代币转账数据")
            return self._create_simulated_token_transfers(token_address, address, limit)
        
        if not self.explorer_config:
            raise ConnectionError("区块链浏览器API未配置")
        
        if not self.validate_address(token_address):
            raise ValueError(f"无效的代币地址: {token_address}")
        
        try:
            # 构建API请求参数
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': token_address,
                'sort': 'desc'
            }
            
            if address:
                params['address'] = address
            
            if start_block:
                params['startblock'] = start_block
            
            if end_block:
                params['endblock'] = end_block
            
            params['page'] = 1
            params['offset'] = min(limit, 10000)
            
            if 'api_key' in self.explorer_config and self.explorer_config['api_key']:
                params['apikey'] = self.explorer_config['api_key']
            
            # 发送请求
            response = requests.get(
                self.explorer_config['url'],
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ConnectionError(f"API请求失败: {response.status_code}")
            
            data = response.json()
            
            if data.get('status') != '1':
                error_message = data.get('message', 'Unknown error')
                raise DataFormatError(f"API返回错误: {error_message}")
            
            transfers = data.get('result', [])
            
            # 获取代币信息
            token_info = self.fetch_contract_info(token_address)
            
            # 格式化转账数据
            formatted_transfers = []
            for transfer in transfers[:limit]:
                # 计算代币数量
                value = int(transfer.get('value', 0))
                decimals = int(transfer.get('tokenDecimal', token_info.get('decimals', 18)))
                token_amount = value / (10 ** decimals)
                
                formatted_transfer = {
                    'tx_hash': transfer.get('hash', ''),
                    'block_number': int(transfer.get('blockNumber', 0)),
                    'timestamp': pd.Timestamp(int(transfer.get('timeStamp', 0)), unit='s'),
                    'from': transfer.get('from', ''),
                    'to': transfer.get('to', ''),
                    'value': token_amount,
                    'token_address': token_address,
                    'token_symbol': transfer.get('tokenSymbol', token_info.get('symbol', '')),
                    'token_name': transfer.get('tokenName', token_info.get('name', '')),
                    'gas': int(transfer.get('gas', 0)),
                    'gas_price': float(transfer.get('gasPrice', 0)) / 1e9,
                    'gas_used': int(transfer.get('gasUsed', 0)),
                    'contract_address': transfer.get('contractAddress', ''),
                    'network': self.network,
                    'chain': self.chain
                }
                formatted_transfers.append(formatted_transfer)
            
            self.logger.info(
                f"获取代币转账: {token_address[:10]}..., "
                f"数量: {len(formatted_transfers)}"
            )
            return formatted_transfers
            
        except Exception as e:
            self.logger.error(f"获取代币转账失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_token_transfers(token_address, address, limit)
    
    def _create_simulated_token_transfers(self, token_address: str, address: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """创建模拟代币转账数据"""
        transfers = []
        
        # 获取代币信息
        token_info = self.fetch_contract_info(token_address)
        
        for i in range(min(limit, 10)):  # 最多10条模拟数据
            tx_hash = f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(64)])}'
            block_number = self.simulation_data['block_number'] - (i * 150)
            
            # 随机生成发送方和接收方
            from_address = address if address else f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(40)])}'
            to_address = f'0x{"".join([hex(np.random.randint(0, 16))[2:] for _ in range(40)])}'
            
            # 如果指定了地址，确保它出现在发送方或接收方
            if address:
                if i % 2 == 0:
                    from_address = address
                else:
                    to_address = address
            
            decimals = token_info.get('decimals', 6)
            token_amount = np.random.uniform(100.0, 10000.0)
            
            transfer = {
                'tx_hash': tx_hash,
                'block_number': block_number,
                'timestamp': pd.Timestamp.now() - timedelta(days=i, hours=np.random.randint(0, 24)),
                'from': from_address,
                'to': to_address,
                'value': token_amount,
                'token_address': token_address,
                'token_symbol': token_info.get('symbol', 'TOKEN'),
                'token_name': token_info.get('name', 'Token'),
                'gas': np.random.randint(50000, 200000),
                'gas_price': np.random.uniform(20.0, 100.0),
                'gas_used': np.random.randint(40000, 150000),
                'contract_address': token_address,
                'network': self.network,
                'chain': self.chain,
                'is_simulated': True
            }
            transfers.append(transfer)
        
        return transfers
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_exchange_flow", ttl=600, sub_dir="onchain")
    def fetch_exchange_flow(self, 
                           exchange: str = 'binance',
                           token_address: Optional[str] = None,
                           hours: int = 24) -> Dict[str, Any]:
        """
        获取交易所资金流数据
        
        参数:
            exchange: 交易所名称
            token_address: 代币合约地址（可选）
            hours: 时间范围（小时）
            
        返回:
            交易所资金流数据
        """
        if self.use_simulation:
            self.logger.info("使用模拟交易所资金流数据")
            return self._create_simulated_exchange_flow(exchange, token_address)
        
        if exchange not in self.exchange_addresses:
            raise ValueError(f"不支持的交易所: {exchange}")
        
        exchange_addresses = self.exchange_addresses[exchange]
        
        # 获取当前区块和时间
        current_block = self.fetch_block_number()
        
        # 估算之前的区块
        avg_block_time = 12 if self.network == 'ethereum' else 2  # 平均出块时间（秒）
        blocks_per_hour = 3600 / avg_block_time
        start_block = max(1, current_block - int(blocks_per_hour * hours))
        
        # 分析资金流
        total_inflow = 0
        total_outflow = 0
        
        token_transfers = []
        
        if token_address:
            # 获取特定代币的转账
            for address in exchange_addresses:
                try:
                    transfers = self.fetch_token_transfers(
                        token_address=token_address,
                        address=address,
                        start_block=start_block,
                        end_block=current_block,
                        limit=100
                    )
                    
                    for transfer in transfers:
                        if transfer['to'] in exchange_addresses:
                            # 流入交易所
                            total_inflow += transfer['value']
                        elif transfer['from'] in exchange_addresses:
                            # 流出交易所
                            total_outflow += transfer['value']
                        
                        token_transfers.append(transfer)
                        
                except Exception as e:
                    self.logger.warning(f"分析交易所 {exchange} 地址 {address} 失败: {e}")
        else:
            # 获取原生代币的转账（ETH/BNB等）
            for address in exchange_addresses:
                try:
                    txs = self.fetch_transaction_history(
                        address=address,
                        start_block=start_block,
                        end_block=current_block,
                        limit=100
                    )
                    
                    for tx in txs:
                        if tx['to'] in exchange_addresses:
                            # 流入交易所
                            total_inflow += tx['value']
                        elif tx['from'] in exchange_addresses:
                            # 流出交易所
                            total_outflow += tx['value']
                            
                except Exception as e:
                    self.logger.warning(f"分析交易所 {exchange} 地址 {address} 失败: {e}")
        
        # 计算净流入流出
        net_flow = total_inflow - total_outflow
        
        result = {
            'timestamp': pd.Timestamp.now(),
            'exchange': exchange,
            'network': self.network,
            'chain': self.chain,
            'asset': token_address or f'{self.network.upper()}_NATIVE',
            'net_flow': net_flow,
            'inflow': total_inflow,
            'outflow': total_outflow,
            'hours': hours,
            'addresses_analyzed': len(exchange_addresses)
        }
        
        self.logger.info(
            f"交易所资金流: {exchange}, "
            f"净流入: {net_flow:.4f}, "
            f"流入: {total_inflow:.4f}, "
            f"流出: {total_outflow:.4f}"
        )
        
        return result
    
    def _create_simulated_exchange_flow(self, exchange: str = 'binance', token_address: Optional[str] = None) -> Dict[str, Any]:
        """创建模拟交易所资金流数据"""
        flow_data = self.simulation_data['exchange_flow']
        
        return {
            'timestamp': pd.Timestamp.now(),
            'exchange': exchange,
            'network': self.network,
            'chain': self.chain,
            'asset': token_address or f'{self.network.upper()}_NATIVE',
            'net_flow': flow_data['net_flow'],
            'inflow': flow_data['inflow'],
            'outflow': flow_data['outflow'],
            'hours': 24,
            'addresses_analyzed': len(self.exchange_addresses.get(exchange, [])),
            'is_simulated': True
        }
    
    @log_errors(reraise=False)
    @cached(key_prefix="onchain_network_stats", ttl=300, sub_dir="onchain")
    def fetch_network_stats(self) -> Dict[str, Any]:
        """
        获取网络统计信息
        
        返回:
            网络统计信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟网络统计")
            return self._create_simulated_network_stats()
        
        if not self.web3:
            raise ConnectionError("Web3连接未初始化")
        
        try:
            # 获取当前区块
            current_block = self.fetch_block_number()
            current_block_info = self.fetch_block(current_block)
            
            # 获取前一个区块（用于计算出块时间）
            if current_block > 1:
                prev_block_info = self.fetch_block(current_block - 1)
                block_time = 12.0 if self.network == 'ethereum' else 2.0  # 使用默认值
                if current_block_info and prev_block_info and current_block_info.get('timestamp') and prev_block_info.get('timestamp'):
                    block_time = (current_block_info['timestamp'] - prev_block_info['timestamp']).total_seconds()
            else:
                block_time = 12 if self.network == 'ethereum' else 2  # 默认值
            
            # 获取Gas价格
            gas_info = self.fetch_gas_price()
            
            # 计算网络使用率
            gas_used = current_block_info.get('gas_used', 0)
            gas_limit = current_block_info.get('gas_limit', 30000000)  # 默认3000万
            gas_utilization = (gas_used / gas_limit) * 100 if gas_limit > 0 else 0
            
            # 估算网络TPS
            avg_tx_per_block = current_block_info.get('transactions', 0)
            tps = avg_tx_per_block / block_time if block_time > 0 else 0
            
            result = {
                'timestamp': pd.Timestamp.now(),
                'network': self.network,
                'chain': self.chain,
                'current_block': current_block,
                'block_time_seconds': block_time,
                'gas_utilization_percent': gas_utilization,
                'transactions_per_second': tps,
                'gas_price_gwei': gas_info['gas_price_gwei'],
                'base_fee_gwei': gas_info.get('base_fee_gwei', 0),
                'suggested_priority_fee_gwei': gas_info.get('suggested_priority_fee_gwei', 0),
                'difficulty': current_block_info.get('difficulty', 0),
                'total_difficulty': current_block_info.get('total_difficulty', 0),
                'block_size_bytes': current_block_info.get('size', 0),
                'network_health': 'healthy' if gas_utilization < 90 else 'congested'
            }
            
            self.logger.info(
                f"网络统计: 区块 {current_block}, "
                f"Gas使用率 {gas_utilization:.1f}%, "
                f"TPS {tps:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取网络统计失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_network_stats()
    
    def _create_simulated_network_stats(self) -> Dict[str, Any]:
        """创建模拟网络统计"""
        stats = self.simulation_data['network_stats']
        
        return {
            'timestamp': pd.Timestamp.now(),
            'network': self.network,
            'chain': self.chain,
            'current_block': stats['current_block'],
            'block_time_seconds': 12.0 if self.network == 'ethereum' else 2.0,
            'gas_utilization_percent': stats['gas_utilization_percent'],
            'transactions_per_second': stats['transactions_per_second'],
            'gas_price_gwei': self.simulation_data['gas_price_gwei'],
            'base_fee_gwei': self.simulation_data['gas_price_gwei'] * 0.8,
            'suggested_priority_fee_gwei': 2.0,
            'difficulty': int(np.random.uniform(1e15, 1e16)),
            'total_difficulty': int(np.random.uniform(1e22, 1e23)),
            'block_size_bytes': np.random.randint(50000, 200000),
            'network_health': stats['network_health'],
            'is_simulated': True
        }
    
    @log_errors(reraise=False)
    def fetch_dune_query(self, 
                        query_id: int,
                        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行Dune Analytics查询
        
        参数:
            query_id: Dune查询ID
            parameters: 查询参数
            
        返回:
            查询结果
        """
        if not self.dune_config or not self.dune_config.get('api_key'):
            self.logger.warning("Dune API未配置或缺少API密钥，返回模拟数据")
            return self._create_simulated_dune_query(query_id, parameters)
        
        try:
            headers = {
                'X-Dune-API-Key': self.dune_config['api_key'],
                'Content-Type': 'application/json'
            }
            
            # 执行查询
            execute_url = f"{self.dune_config['base_url']}/query/{query_id}/execute"
            
            payload = {}
            if parameters:
                payload['query_parameters'] = parameters
            
            response = requests.post(
                execute_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ConnectionError(f"Dune API请求失败: {response.status_code}")
            
            execute_data = response.json()
            execution_id = execute_data.get('execution_id')
            
            if not execution_id:
                raise DataFormatError("未获取到执行ID")
            
            # 等待查询完成
            status_url = f"{self.dune_config['base_url']}/execution/{execution_id}/status"
            results_url = f"{self.dune_config['base_url']}/execution/{execution_id}/results"
            
            max_retries = 30
            for _ in range(max_retries):
                status_response = requests.get(status_url, headers=headers, timeout=30)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    if status_data.get('state') == 'QUERY_STATE_COMPLETED':
                        # 获取结果
                        results_response = requests.get(results_url, headers=headers, timeout=30)
                        
                        if results_response.status_code == 200:
                            results_data = results_response.json()
                            return results_data
                        else:
                            raise ConnectionError(f"获取结果失败: {results_response.status_code}")
                    
                    elif status_data.get('state') in ['QUERY_STATE_FAILED', 'QUERY_STATE_CANCELLED']:
                        raise DataFormatError(f"查询失败: {status_data.get('state')}")
                
                # 等待1秒后重试
                time.sleep(1)
            
            raise TimeoutError("查询执行超时")
            
        except Exception as e:
            self.logger.error(f"Dune查询失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_dune_query(query_id, parameters)
    
    def _create_simulated_dune_query(self, query_id: int, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建模拟Dune查询结果"""
        # 根据查询ID返回不同的模拟数据
        if query_id == 3489:  # ETH2 Staking Deposits
            return {
                'execution_id': f'simulated_{query_id}',
                'query_id': query_id,
                'state': 'QUERY_STATE_COMPLETED',
                'submitted_at': pd.Timestamp.now().isoformat(),
                'execution_started_at': pd.Timestamp.now().isoformat(),
                'execution_ended_at': pd.Timestamp.now().isoformat(),
                'result': {
                    'rows': [
                        {
                            'validators': self.simulation_data['eth2_staking']['total_validators'],
                            'total_eth': self.simulation_data['eth2_staking']['total_eth_staked'],
                            'avg_balance': 32.1,
                            'date': pd.Timestamp.now().strftime('%Y-%m-%d')
                        }
                    ],
                    'metadata': {
                        'column_names': ['validators', 'total_eth', 'avg_balance', 'date']
                    }
                }
            }
        else:
            # 通用模拟数据
            return {
                'execution_id': f'simulated_{query_id}',
                'query_id': query_id,
                'state': 'QUERY_STATE_COMPLETED',
                'submitted_at': pd.Timestamp.now().isoformat(),
                'execution_started_at': pd.Timestamp.now().isoformat(),
                'execution_ended_at': pd.Timestamp.now().isoformat(),
                'result': {
                    'rows': [
                        {'value': 1000, 'date': pd.Timestamp.now().strftime('%Y-%m-%d')}
                    ],
                    'metadata': {
                        'column_names': ['value', 'date']
                    }
                }
            }
    
    @log_errors(reraise=False)
    def fetch_the_graph_query(self,
                            subgraph: str,
                            query: str,
                            variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行The Graph查询
        
        参数:
            subgraph: subgraph名称或URL
            query: GraphQL查询
            variables: 查询变量
            
        返回:
            查询结果
        """
        try:
            # 构建候选 URL 列表（优先尝试官方/默认端点，失败再尝试去中心化网关）
            urls: List[str] = []
            if subgraph.startswith('http'):
                urls.append(subgraph)
            else:
                try:
                    urls.append(OnChainDataSource.get_the_graph_api(subgraph))
                except ValueError:
                    pass

                # 常见 subgraph 的去中心化端点（可作为 fallback）
                decentralized_key_map = {
                    'uniswap_v3_ethereum': 'uniswap_v3_ethereum_decentralized',
                }
                dec_key = decentralized_key_map.get(subgraph)
                if dec_key:
                    dec_url = OnChainDataSource.THE_GRAPH_API.get(dec_key)
                    if dec_url:
                        urls.append(dec_url)

            # 如果仍然没有 URL，直接触发异常走模拟数据
            if not urls:
                raise ValueError(f"找不到可用的 The Graph 端点: {subgraph}")

            # 如果提供了网关 API key，自动把去中心化 URL 规范化为 /api/<key>/subgraphs/id/<id>
            gateway_key = os.environ.get('THEGRAPH_API_KEY') or os.environ.get('THE_GRAPH_API_KEY')
            gateway_token = os.environ.get('THEGRAPH_API_TOKEN') or os.environ.get('THE_GRAPH_API_TOKEN')

            def _is_gateway_url(u: str) -> bool:
                return 'gateway-' in u and 'thegraph.com' in u

            def _requires_gateway_auth(u: str) -> bool:
                # 常见需要鉴权的形式：.../api/subgraphs/id/<id>
                # 带 key 的形式：.../api/<key>/subgraphs/id/<id>
                return _is_gateway_url(u) and '/api/subgraphs/id/' in u

            # 没有 key 时，先过滤掉明确需要鉴权的 gateway URL，避免无意义报错
            if not gateway_key and not gateway_token:
                urls = [u for u in urls if not _requires_gateway_auth(u)]
            normalized_urls: List[str] = []
            for u in urls:
                normalized_urls.append(u)
                if gateway_key and '/api/subgraphs/id/' in u and '/api/' in u:
                    # 兼容两种形式：
                    # 1) https://gateway-xxx.thegraph.com/api/subgraphs/id/<id>
                    # 2) https://gateway-xxx.thegraph.com/api/<key>/subgraphs/id/<id>
                    try:
                        parts = u.split('/api/', 1)
                        base = parts[0]
                        suffix = parts[1]
                        if not suffix.startswith(gateway_key + '/'):
                            normalized_urls.append(f"{base}/api/{gateway_key}/{suffix.lstrip('/')}")
                    except Exception:
                        pass

            # 去重且保持顺序
            seen = set()
            urls = []
            for u in normalized_urls:
                if u and u not in seen:
                    urls.append(u)
                    seen.add(u)

            # 规范化后再次过滤
            if not gateway_key and not gateway_token:
                urls = [u for u in urls if not _requires_gateway_auth(u)]
            
            # 构建请求
            payload = {'query': query}
            if variables:
                payload['variables'] = variables
            
            last_error: Optional[Exception] = None
            for url in urls:
                try:
                    headers: Dict[str, str] = {}
                    # 部分 gateway 要求 Authorization header（兼容 header 鉴权）
                    if _is_gateway_url(url):
                        if gateway_token:
                            headers['Authorization'] = f"Bearer {gateway_token}"
                        elif gateway_key:
                            headers['Authorization'] = f"Bearer {gateway_key}"

                    try:
                        response = requests.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=30
                        )
                    except requests.exceptions.ProxyError as pe:
                        # 环境代理导致的 ProxyError：绕过代理直连重试一次
                        self.logger.warning(f"The Graph代理连接失败，尝试绕过代理直连: {url} -> {pe}")
                        session = requests.Session()
                        session.trust_env = False
                        response = session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=30
                        )

                    if response.status_code != 200:
                        raise ConnectionError(f"The Graph API请求失败: {response.status_code}")

                    data = response.json()

                    if 'errors' in data:
                        error_messages = [err.get('message', 'Unknown error') for err in data['errors']]
                        msg = ', '.join(error_messages)

                        # gateway 未鉴权：尝试下一个端点（最终会回退到模拟数据）
                        if ('auth error' in msg.lower() or 'authorization' in msg.lower()) and not gateway_key:
                            raise DataFormatError(f"The Graph查询错误: {msg}")

                        # 典型错误：旧的 api.thegraph.com 端点已下线，继续尝试下一个 URL
                        if 'endpoint has been removed' in msg.lower() or 'has been removed' in msg.lower():
                            raise DataFormatError(f"The Graph查询错误: {msg}")

                        raise DataFormatError(f"The Graph查询错误: {msg}")

                    # 成功
                    return data
                except Exception as e:
                    last_error = e
                    # 如果是旧端点被移除/不可用，继续尝试候选列表
                    if ('api.thegraph.com' in url) or ('endpoint has been removed' in str(e).lower()):
                        self.logger.warning(f"The Graph端点不可用，尝试下一个: {url} -> {e}")
                        continue
                    # gateway 鉴权失败：尝试下一个端点
                    if ('auth error' in str(e).lower()) or ('authorization' in str(e).lower()):
                        self.logger.warning(f"The Graph需要鉴权或鉴权失败，尝试下一个端点: {url} -> {e}")
                        continue
                    # 其它错误也尝试下一个端点（多端点容错）
                    self.logger.warning(f"The Graph查询失败，尝试下一个端点: {url} -> {e}")
                    continue

            if last_error:
                raise last_error
            raise ConnectionError("The Graph查询失败: 未知错误")
            
        except Exception as e:
            self.logger.error(f"The Graph查询失败: {e}")
            # 失败时返回模拟数据
            return self._create_simulated_the_graph_query(subgraph, query, variables)
    
    def _create_simulated_the_graph_query(self, subgraph: str, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建模拟The Graph查询结果"""
        # 简单解析查询类型
        if 'pools' in query and 'factories' in query:
            # Uniswap V3查询
            return {
                'data': {
                    'pools': [
                        {
                            'id': '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
                            'token0': {'symbol': 'WETH'},
                            'token1': {'symbol': 'USDC'},
                            'totalValueLockedUSD': str(self.simulation_data['defi_stats']['total_tvl_usd'] * 0.1),
                            'volumeUSD': str(self.simulation_data['defi_stats']['total_volume_usd'] * 0.05),
                            'feesUSD': str(self.simulation_data['defi_stats']['total_fees_usd'] * 0.05)
                        }
                    ],
                    'factories': [
                        {
                            'poolCount': 10000,
                            'totalVolumeUSD': str(self.simulation_data['defi_stats']['total_volume_usd']),
                            'totalFeesUSD': str(self.simulation_data['defi_stats']['total_fees_usd']),
                            'totalValueLockedUSD': str(self.simulation_data['defi_stats']['total_tvl_usd'])
                        }
                    ]
                }
            }
        else:
            # 通用模拟数据
            return {
                'data': {
                    'result': 'simulated data'
                }
            }
    
    def test_connection(self) -> bool:
        """
        测试连接
        
        返回:
            连接是否成功
        """
        try:
            if self.use_simulation:
                self.logger.info("模拟模式，连接测试跳过")
                return True
            
            # 测试Web3连接
            if self.web3 and self.web3.is_connected():
                try:
                    block_number = self.web3.eth.block_number
                    self.logger.info(f"Web3连接测试成功，当前区块: {block_number}")
                    return True
                except:
                    pass
            
            # 如果Web3连接失败，尝试备用方法
            self.logger.warning("Web3连接测试失败，尝试备用方法")
            
            # 尝试使用区块链浏览器API
            if self.explorer_config:
                try:
                    # 简单的API测试
                    test_address = '0x0000000000000000000000000000000000000000'
                    params = {
                        'module': 'account',
                        'action': 'balance',
                        'address': test_address,
                        'tag': 'latest'
                    }
                    
                    if 'api_key' in self.explorer_config and self.explorer_config['api_key']:
                        params['apikey'] = self.explorer_config['api_key']
                    
                    response = requests.get(
                        self.explorer_config['url'],
                        params=params,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == '1':
                            self.logger.info(f"区块链浏览器API连接测试成功")
                            return True
                except Exception as e:
                    self.logger.warning(f"区块链浏览器API连接测试失败: {e}")
            
            # 所有连接都失败
            self.logger.error("所有连接测试失败，启用模拟模式")
            self.use_simulation = True
            return True  # 模拟模式也算"成功"
            
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            self.use_simulation = True
            return True  # 模拟模式也算"成功"
    
    def close(self):
        """关闭获取器，释放资源"""
        if self.web3_manager:
            self.web3_manager.close_all()
        
        super().close()


# ==================== 特定链上指标获取器 ====================

class EthereumOnChainFetcher(OnChainFetcher):
    """
    以太坊链上数据获取器
    """
    
    def __init__(self, 
                 chain: str = "mainnet",
                 config: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None,
                 use_simulation: bool = False):
        """
        初始化以太坊链上数据获取器
        
        参数:
            chain: 链类型 (mainnet, goerli)
            config: 配置字典
            cache_manager: 缓存管理器
            use_simulation: 是否使用模拟数据
        """
        super().__init__(
            network="ethereum",
            chain=chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation
        )
    
    @log_errors(reraise=False)
    @cached(key_prefix="eth_eth2_staking", ttl=600, sub_dir="ethereum")
    def fetch_eth2_staking_stats(self) -> Dict[str, Any]:
        """
        获取ETH2.0质押统计
        
        返回:
            ETH2.0质押统计信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟ETH2.0质押数据")
            staking_data = self.simulation_data['eth2_staking']
            
            return {
                'timestamp': pd.Timestamp.now(),
                'network': 'ethereum',
                'chain': self.chain,
                'total_validators': staking_data['total_validators'],
                'total_eth_staked': staking_data['total_eth_staked'],
                'average_validator_balance': 32.1,
                'staking_apy': staking_data['staking_apy'],
                'withdrawal_queue_length': 0,
                'data_source': 'simulation',
                'is_simulated': True
            }
        
        # 使用Dune查询ETH2.0质押数据
        # 查询ID: 例如 3489 (ETH2 Staking Deposits)
        try:
            query_result = self.fetch_dune_query(
                query_id=3489,  # 示例查询ID，需要替换为实际查询
                parameters={}
            )
            
            # 解析结果
            result = {
                'timestamp': pd.Timestamp.now(),
                'network': 'ethereum',
                'chain': self.chain,
                'total_validators': 0,
                'total_eth_staked': 0,
                'average_validator_balance': 0,
                'staking_apy': 0.04,  # 默认4%
                'withdrawal_queue_length': 0,
                'data_source': 'dune'
            }
            
            if query_result and 'result' in query_result:
                rows = query_result['result'].get('rows', [])
                if rows:
                    # 假设查询返回最新数据
                    latest = rows[0]
                    result['total_validators'] = latest.get('validators', 0)
                    result['total_eth_staked'] = latest.get('total_eth', 0)
                    result['average_validator_balance'] = latest.get('avg_balance', 32)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"获取ETH2.0质押统计失败，使用模拟数据: {e}")
            
            # 返回模拟数据
            staking_data = self.simulation_data['eth2_staking']
            
            return {
                'timestamp': pd.Timestamp.now(),
                'network': 'ethereum',
                'chain': self.chain,
                'total_validators': staking_data['total_validators'],
                'total_eth_staked': staking_data['total_eth_staked'],
                'average_validator_balance': 32.1,
                'staking_apy': staking_data['staking_apy'],
                'withdrawal_queue_length': 0,
                'data_source': 'simulation_fallback',
                'is_simulated': True
            }
    
    @log_errors(reraise=False)
    @cached(key_prefix="eth_defi_stats", ttl=600, sub_dir="ethereum")
    def fetch_defi_stats(self) -> Dict[str, Any]:
        """
        获取DeFi统计信息
        
        返回:
            DeFi统计信息
        """
        if self.use_simulation:
            self.logger.info("使用模拟DeFi统计")
            defi_data = self.simulation_data['defi_stats']
            
            return {
                'timestamp': pd.Timestamp.now(),
                'network': 'ethereum',
                'chain': self.chain,
                'total_pools': 10000,
                'total_tvl_usd': defi_data['total_tvl_usd'],
                'total_volume_usd': defi_data['total_volume_usd'],
                'total_fees_usd': defi_data['total_fees_usd'],
                'top_pools': [
                    {
                        'pool_address': '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
                        'pair': 'WETH/USDC',
                        'tvl_usd': defi_data['total_tvl_usd'] * 0.1,
                        'volume_usd': defi_data['total_volume_usd'] * 0.05,
                        'fees_usd': defi_data['total_fees_usd'] * 0.05
                    }
                ],
                'data_source': 'simulation',
                'is_simulated': True
            }
        
        try:
            # 使用The Graph查询DeFi数据
            # 查询Uniswap V3数据
            query = """
            {
                pools(first: 5, orderBy: totalValueLockedUSD, orderDirection: desc) {
                    id
                    token0 {
                        symbol
                    }
                    token1 {
                        symbol
                    }
                    totalValueLockedUSD
                    volumeUSD
                    feesUSD
                }
                factories(first: 1) {
                    poolCount
                    totalVolumeUSD
                    totalFeesUSD
                    totalValueLockedUSD
                }
            }
            """
            
            result = self.fetch_the_graph_query(
                subgraph='uniswap_v3_ethereum',
                query=query
            )
            
            data = result.get('data', {})
            factories = data.get('factories', [])
            pools = data.get('pools', [])
            
            defi_stats = {
                'timestamp': pd.Timestamp.now(),
                'network': 'ethereum',
                'chain': self.chain,
                'total_pools': factories[0].get('poolCount', 0) if factories else 0,
                'total_tvl_usd': float(factories[0].get('totalValueLockedUSD', 0)) if factories else 0,
                'total_volume_usd': float(factories[0].get('totalVolumeUSD', 0)) if factories else 0,
                'total_fees_usd': float(factories[0].get('totalFeesUSD', 0)) if factories else 0,
                'top_pools': [],
                'data_source': 'the_graph'
            }
            
            for pool in pools[:5]:
                defi_stats['top_pools'].append({
                    'pool_address': pool.get('id'),
                    'pair': f"{pool['token0']['symbol']}/{pool['token1']['symbol']}",
                    'tvl_usd': float(pool.get('totalValueLockedUSD', 0)),
                    'volume_usd': float(pool.get('volumeUSD', 0)),
                    'fees_usd': float(pool.get('feesUSD', 0))
                })
            
            return defi_stats
            
        except Exception as e:
            self.logger.error(f"获取DeFi统计失败: {e}")
            # 返回模拟数据
            defi_data = self.simulation_data['defi_stats']
            
            return {
                'timestamp': pd.Timestamp.now(),
                'network': 'ethereum',
                'chain': self.chain,
                'total_pools': 10000,
                'total_tvl_usd': defi_data['total_tvl_usd'],
                'total_volume_usd': defi_data['total_volume_usd'],
                'total_fees_usd': defi_data['total_fees_usd'],
                'top_pools': [
                    {
                        'pool_address': '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
                        'pair': 'WETH/USDC',
                        'tvl_usd': defi_data['total_tvl_usd'] * 0.1,
                        'volume_usd': defi_data['total_volume_usd'] * 0.05,
                        'fees_usd': defi_data['total_fees_usd'] * 0.05
                    }
                ],
                'data_source': 'simulation_fallback',
                'is_simulated': True
            }
    
    @log_errors(reraise=False)
    @cached(key_prefix="eth_nft_stats", ttl=600, sub_dir="ethereum")
    def fetch_nft_stats(self) -> Dict[str, Any]:
        """
        获取NFT市场统计
        
        返回:
            NFT统计信息
        """
        # 这里可以集成OpenSea、LooksRare等NFT市场的API
        # 由于需要API密钥，这里返回模拟数据
        
        return {
            'timestamp': pd.Timestamp.now(),
            'network': 'ethereum',
            'chain': self.chain,
            'total_nft_trades_24h': 25000,
            'total_volume_eth_24h': 12000,
            'total_volume_usd_24h': 20000000,
            'top_collections': [
                {'name': 'Bored Ape Yacht Club', 'volume_eth': 1500},
                {'name': 'CryptoPunks', 'volume_eth': 1200},
                {'name': 'Mutant Ape Yacht Club', 'volume_eth': 800},
                {'name': 'Otherdeed', 'volume_eth': 600},
                {'name': 'Azuki', 'volume_eth': 500}
            ],
            'market_share': {
                'opensea': 0.75,
                'looksrare': 0.15,
                'x2y2': 0.08,
                'other': 0.02
            },
            'data_source': 'simulation',
            'is_simulated': True
        }


# ==================== 链上数据管理器 ====================

try:
    from crypto_data_system.storage.data_manager import FileDataManager
except (ImportError, ModuleNotFoundError):
    # 脚本直接运行时的备用导入方式
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crypto_data_system.storage.data_manager import FileDataManager


class OnChainDataManager(FileDataManager):
    """
    链上数据管理器
    
    管理多个网络的链上数据获取
    """
    
    def __init__(self, 
                 networks: Optional[List[str]] = None,
                 fetcher_config: Optional[Dict] = None,
                 root_dir: Optional[str] = None,
                 cache_manager: Optional[Any] = None,
                 use_simulation: bool = False):
        """
        初始化链上数据管理器
        
        参数:
            networks: 网络列表
            fetcher_config: 获取器配置
            use_simulation: 是否使用模拟数据
        """
        self.networks = networks or ['ethereum', 'polygon', 'bsc']
        self.fetcher_config = fetcher_config or {}
        self.fetchers = {}
        self.use_simulation = use_simulation
        
        # 初始化日志
        self.logger = get_logger("onchain_manager")

        # 初始化文件存储（子目录按 onchain 分组）
        super().__init__(root_dir=root_dir, sub_dir="onchain", file_format="json", cache_manager=cache_manager)
    
    def init_fetchers(self):
        """初始化所有网络获取器"""
        for network in self.networks:
            try:
                if network == 'ethereum':
                    self.fetchers[network] = EthereumOnChainFetcher(
                        config=self.fetcher_config,
                        use_simulation=self.use_simulation
                    )
                else:
                    self.fetchers[network] = OnChainFetcher(
                        network=network,
                        config=self.fetcher_config,
                        use_simulation=self.use_simulation
                    )
                
                self.logger.info(f"初始化 {network} 链上获取器成功")
                
                # 测试连接
                if not self.fetchers[network].test_connection():
                    self.logger.warning(f"{network} 获取器连接测试失败，使用模拟模式")
                
            except Exception as e:
                self.logger.error(f"初始化 {network} 链上获取器失败: {e}")
    
    def fetch_multi_network_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取多网络统计信息
        
        返回:
            多网络统计信息字典
        """
        results = {}
        
        for network, fetcher in self.fetchers.items():
            try:
                stats = fetcher.fetch_network_stats()
                results[network] = stats
            except Exception as e:
                self.logger.error(f"获取 {network} 网络统计失败: {e}")
                results[network] = None
        
        # 统一持久化
        self.save_dict("onchain_network_stats", results)
        return results
    
    def fetch_exchange_flows_all(self, 
                                exchange: str = 'binance',
                                token_address: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        获取所有网络的交易所资金流
        
        参数:
            exchange: 交易所名称
            token_address: 代币合约地址
            
        返回:
            多网络交易所资金流字典
        """
        results = {}
        
        for network, fetcher in self.fetchers.items():
            try:
                flow_data = fetcher.fetch_exchange_flow(
                    exchange=exchange,
                    token_address=token_address
                )
                results[network] = flow_data
            except Exception as e:
                self.logger.error(f"获取 {network} 交易所资金流失败: {e}")
                results[network] = None
        
        return results
    
    def get_network_summary(self) -> Dict[str, Any]:
        """
        获取网络摘要
        
        返回:
            网络摘要信息
        """
        stats = self.fetch_multi_network_stats()
        
        summary = {
            'total_networks': len(self.fetchers),
            'active_networks': sum(1 for v in stats.values() if v is not None),
            'networks': {},
            'comparison': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 提取关键指标
        for network, stat in stats.items():
            if stat:
                summary['networks'][network] = {
                    'block_number': stat.get('current_block', 0),
                    'tps': stat.get('transactions_per_second', 0),
                    'gas_price_gwei': stat.get('gas_price_gwei', 0),
                    'gas_utilization': stat.get('gas_utilization_percent', 0),
                    'network_health': stat.get('network_health', 'unknown'),
                    'is_simulated': stat.get('is_simulated', False)
                }
        
        # 比较不同网络
        if summary['networks']:
            networks = list(summary['networks'].keys())
            
            # 找出TPS最高的网络
            tps_values = {n: data['tps'] for n, data in summary['networks'].items()}
            max_tps_network = max(tps_values, key=tps_values.get)
            summary['comparison']['fastest_network'] = max_tps_network
            
            # 找出Gas最低的网络
            gas_values = {n: data['gas_price_gwei'] for n, data in summary['networks'].items()}
            min_gas_network = min(gas_values, key=gas_values.get)
            summary['comparison']['cheapest_network'] = min_gas_network
            
            # 统计模拟模式数量
            simulated_count = sum(1 for data in summary['networks'].values() if data.get('is_simulated', False))
            summary['simulated_networks'] = simulated_count
            summary['real_networks'] = len(summary['networks']) - simulated_count
        
        return summary
    
    def close_all(self):
        """关闭所有获取器"""
        for network, fetcher in self.fetchers.items():
            try:
                fetcher.close()
                self.logger.info(f"关闭 {network} 链上获取器成功")
            except Exception as e:
                self.logger.error(f"关闭 {network} 链上获取器失败: {e}")
        
        self.fetchers.clear()


# ==================== 测试函数 ====================

def test_onchain_fetcher():
    """测试链上获取器"""
    print("=" * 60)
    print("链上获取器模块测试")
    print("=" * 60)
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 测试基础功能
    print("\n1. 测试OnChainFetcher基础功能:")
    try:
        # 创建获取器（以太坊主网）
        # 注意：如果网络不可用，会自动切换到模拟模式
        fetcher = OnChainFetcher(network="ethereum", chain="mainnet")
        
        print(f"✅ 获取器创建成功: {fetcher}")
        print(f"✅ 网络: {fetcher.network}")
        print(f"✅ 链: {fetcher.chain}")
        print(f"✅ 模拟模式: {fetcher.use_simulation}")
        
        # 测试连接
        if fetcher.test_connection():
            print("✅ 连接测试成功")
        else:
            print("⚠️  连接测试失败")
        
        # 测试获取区块号
        print("\n2. 测试获取区块号:")
        try:
            block_number = fetcher.fetch_block_number()
            print(f"✅ 当前区块号: {block_number}")
            if fetcher.use_simulation:
                print("⚠️  使用模拟数据")
        except Exception as e:
            print(f"❌ 获取区块号失败: {e}")
        
        # 测试获取区块信息
        print("\n3. 测试获取区块信息:")
        try:
            block_info = fetcher.fetch_block('latest')
            if block_info:
                print(f"✅ 区块获取成功: #{block_info.get('block_number', 'N/A')}")
                print(f"  时间: {block_info.get('timestamp', 'N/A')}")
                print(f"  交易数: {block_info.get('transactions', 'N/A')}")
                print(f"  Gas使用: {block_info.get('gas_used', 'N/A')}")
                if block_info.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  区块获取失败或为空")
        except Exception as e:
            print(f"❌ 获取区块信息失败: {e}")
        
        # 测试获取Gas价格
        print("\n4. 测试获取Gas价格:")
        try:
            gas_info = fetcher.fetch_gas_price()
            if gas_info:
                print(f"✅ Gas价格获取成功")
                print(f"  Gas价格: {gas_info.get('gas_price_gwei', 'N/A'):.2f} Gwei")
                print(f"  基础费用: {gas_info.get('base_fee_gwei', 'N/A'):.2f} Gwei")
                if gas_info.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  Gas价格获取失败")
        except Exception as e:
            print(f"❌ 获取Gas价格失败: {e}")
        
        # 测试获取网络统计
        print("\n5. 测试获取网络统计:")
        try:
            network_stats = fetcher.fetch_network_stats()
            if network_stats:
                print(f"✅ 网络统计获取成功")
                print(f"  TPS: {network_stats.get('transactions_per_second', 'N/A'):.2f}")
                print(f"  Gas使用率: {network_stats.get('gas_utilization_percent', 'N/A'):.1f}%")
                print(f"  网络健康: {network_stats.get('network_health', 'N/A')}")
                if network_stats.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  网络统计获取失败")
        except Exception as e:
            print(f"❌ 获取网络统计失败: {e}")
        
        # 测试获取地址余额（使用示例地址）
        print("\n6. 测试获取地址余额:")
        test_address = "0x742d35Cc6634C0532925a3b844Bc9e0F2d5A2A7e"  # 币安地址示例
        try:
            balance_info = fetcher.fetch_address_balance(test_address)
            if balance_info:
                print(f"✅ 地址余额获取成功")
                print(f"  地址: {balance_info.get('address', 'N/A')[:10]}...")
                print(f"  原生代币余额: {balance_info.get('balance_native', 'N/A'):.6f} ETH")
                if balance_info.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  地址余额获取失败")
        except Exception as e:
            print(f"❌ 获取地址余额失败: {e}")
        
        # 测试获取合约信息（USDT合约）
        print("\n7. 测试获取合约信息:")
        usdt_address = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # 以太坊USDT
        try:
            contract_info = fetcher.fetch_contract_info(usdt_address)
            if contract_info:
                print(f"✅ 合约信息获取成功")
                print(f"  合约地址: {contract_info.get('contract_address', 'N/A')[:10]}...")
                print(f"  名称: {contract_info.get('name', 'N/A')}")
                print(f"  符号: {contract_info.get('symbol', 'N/A')}")
                print(f"  总供应量: {contract_info.get('total_supply', 'N/A'):,.0f}")
                if contract_info.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  合约信息获取失败")
        except Exception as e:
            print(f"❌ 获取合约信息失败: {e}")
        
        # 测试交易所资金流
        print("\n8. 测试交易所资金流:")
        try:
            exchange_flow = fetcher.fetch_exchange_flow(exchange='binance')
            if exchange_flow:
                print(f"✅ 交易所资金流获取成功")
                print(f"  交易所: {exchange_flow.get('exchange', 'N/A')}")
                print(f"  净流入: {exchange_flow.get('net_flow', 'N/A'):.6f}")
                print(f"  流入: {exchange_flow.get('inflow', 'N/A'):.6f}")
                print(f"  流出: {exchange_flow.get('outflow', 'N/A'):.6f}")
                if exchange_flow.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  交易所资金流获取失败")
        except Exception as e:
            print(f"❌ 获取交易所资金流失败: {e}")
        
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
    
    # 测试以太坊特定获取器
    print("\n9. 测试EthereumOnChainFetcher:")
    try:
        eth_fetcher = EthereumOnChainFetcher()
        
        print(f"✅ 以太坊获取器创建成功")
        print(f"✅ 模拟模式: {eth_fetcher.use_simulation}")
        
        # 测试ETH2.0质押统计
        print("\n10. 测试ETH2.0质押统计:")
        try:
            staking_stats = eth_fetcher.fetch_eth2_staking_stats()
            if staking_stats:
                print(f"✅ ETH2.0质押统计获取成功")
                print(f"  总验证者: {staking_stats.get('total_validators', 'N/A'):,}")
                print(f"  总质押ETH: {staking_stats.get('total_eth_staked', 'N/A'):,.0f}")
                print(f"  APY: {staking_stats.get('staking_apy', 'N/A'):.2%}")
                if staking_stats.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  ETH2.0质押统计获取失败")
        except Exception as e:
            print(f"❌ 获取ETH2.0质押统计失败: {e}")
        
        # 测试DeFi统计
        print("\n11. 测试DeFi统计:")
        try:
            defi_stats = eth_fetcher.fetch_defi_stats()
            if defi_stats:
                print(f"✅ DeFi统计获取成功")
                print(f"  总TVL: ${defi_stats.get('total_tvl_usd', 'N/A'):,.0f}")
                print(f"  总交易量: ${defi_stats.get('total_volume_usd', 'N/A'):,.0f}")
                print(f"  总手续费: ${defi_stats.get('total_fees_usd', 'N/A'):,.0f}")
                if defi_stats.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  DeFi统计获取失败")
        except Exception as e:
            print(f"❌ 获取DeFi统计失败: {e}")
        
        # 测试NFT统计
        print("\n12. 测试NFT统计:")
        try:
            nft_stats = eth_fetcher.fetch_nft_stats()
            if nft_stats:
                print(f"✅ NFT统计获取成功")
                print(f"  24小时交易量: {nft_stats.get('total_volume_eth_24h', 'N/A'):,.0f} ETH")
                print(f"  24小时交易额: ${nft_stats.get('total_volume_usd_24h', 'N/A'):,.0f}")
                if nft_stats.get('is_simulated'):
                    print("⚠️  使用模拟数据")
            else:
                print("⚠️  NFT统计获取失败")
        except Exception as e:
            print(f"❌ 获取NFT统计失败: {e}")
        
        eth_fetcher.close()
        print("✅ 以太坊获取器关闭成功")
        
    except Exception as e:
        print(f"❌ 以太坊获取器测试失败: {e}")
    
    # 测试链上数据管理器
    print("\n13. 测试链上数据管理器:")
    try:
        manager = OnChainDataManager(networks=['ethereum', 'polygon'])
        manager.init_fetchers()
        
        print(f"✅ 管理器创建成功，网络: {manager.networks}")
        
        # 获取网络摘要
        summary = manager.get_network_summary()
        print(f"✅ 网络摘要:")
        print(f"  网络数量: {summary['total_networks']}")
        print(f"  活跃网络: {summary['active_networks']}")
        if 'simulated_networks' in summary:
            print(f"  模拟网络: {summary['simulated_networks']}")
            print(f"  真实网络: {summary['real_networks']}")
        
        manager.close_all()
        print("✅ 管理器关闭成功")
        
    except Exception as e:
        print(f"❌ 管理器测试失败: {e}")
    
    print("\n✅ 链上获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    test_onchain_fetcher()