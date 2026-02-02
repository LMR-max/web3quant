#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加密货币数据系统 Web 应用

Flask 服务器，提供 REST API 和 Web UI 界面
支持所有数据获取、分析、可视化功能
"""

import os
import sys
import json
import logging
import uuid
import time
import subprocess
import csv
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import traceback
import pandas as pd
import re
from pathlib import Path

from crypto_data_system.storage.data_manager import FileDataManager
from crypto_data_system.storage.data_integrity import DataIntegrityVerifier
from crypto_data_system.utils.symbol_classifier import classify_symbol

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# 导入系统模块
from crypto_data_system import (
    create_fetcher,
    create_data_manager,
    get_logger,
    CacheManager,
    CacheConfig,
    ExchangeFlowAnalyzer,
    AddressBehaviorAnalyzer,
    LargeMoveAnalyzer,
    MEVAnalyzer,
    GasAnalyzer,
    ProtocolAnalyzer,
    CapitalCycleAnalyzer,
    TokenDistributionAnalyzer,
    NFTAnalyzer,
    PriceRelationAnalyzer,
)

# ==================== Flask 应用配置 ====================

# 获取静态文件和模板的绝对路径
static_folder = os.path.join(project_root, 'web_static')
template_folder = os.path.join(project_root, 'web_templates')

app = Flask(__name__, 
           static_folder=static_folder,
           static_url_path='/static',
           template_folder=template_folder)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 用于 Web 前端持久化“最近一次合并后的结果”
_web_store = FileDataManager(sub_dir="web")

# 本地最终数据索引（data_manager_storage）缓存
_local_index_cache_lock = Lock()
_local_index_cache = {
    'cache_key': None,
    'cached_at': 0.0,
    'payload': None,
}

# 全局系统实例
class GlobalSystem:
    def __init__(self):
        self.config = {}
        self.cache_config = CacheConfig()
        self.cache_manager = CacheManager(self.cache_config)
        self.fetchers = {}

    def get_fetcher(self, exchange: str, market_type: str):
        key = f"{exchange}_{market_type}"
        if key not in self.fetchers:
             self.fetchers[key] = create_fetcher(
                exchange=exchange,
                market_type=market_type,
                config=self.config,
                cache_manager=self.cache_manager
            )
        return self.fetchers[key]

global_system = GlobalSystem()


def _parse_json_maybe(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return json.loads(text)
    return None


def _parse_addresses_maybe(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    if isinstance(value, str):
        parts = re.split(r"[\s,;]+", value.strip())
        return [p for p in parts if p]
    return []

@app.route('/api/alphagen/meta', methods=['GET'])
def get_alphagen_meta():
    """Returns available operators, window candidates, and columns from a specific CSV."""
    from alphagen_style.dsl import list_operator_specs, WINDOW_CANDIDATES
    
    # Defaults
    cols = []
    
    # If file_uri or path provided, read header
    path_arg = request.args.get('path')
    if path_arg:
        try:
            full_path = os.path.abspath(path_arg)
            if os.path.exists(full_path):
                # Read just header
                df = pd.read_csv(full_path, nrows=0)
                cols = df.columns.tolist()
        except Exception as e:
            logger.error(f"Error reading meta from {path_arg}: {e}")

    return jsonify({
        "operators": list_operator_specs(),
        "windows": WINDOW_CANDIDATES,
        "columns": cols
    })


@app.route('/api/spot/orderbook', methods=['GET'])
def get_spot_orderbook():
    """获取现货盘口数据"""
    exchange = request.args.get('exchange', 'binance')
    symbol = request.args.get('symbol', 'BTC/USDT')
    limit = int(request.args.get('limit', 20))
    
    try:
        # 获取 Fetcher
        fetcher = global_system.get_fetcher(exchange, 'spot')
        if not fetcher:
            return jsonify({'success': False, 'error': f'Failed to create fetcher for {exchange}'})
        
        # 获取盘口
        order_book = fetcher.fetch_order_book(symbol, limit=limit)
        
        if order_book:
            summary = order_book.get_summary(levels=limit)
            
            # 处理 timestamp 可能为 NaT 的情况
            ts = order_book.timestamp
            ts_val = 0
            if pd.notna(ts):
                try:
                    ts_val = ts.timestamp() * 1000
                except Exception:
                    ts_val = time.time() * 1000
            else:
                ts_val = time.time() * 1000

            return jsonify({
                'success': True,
                'data': {
                    'timestamp': ts_val,
                    'symbol': order_book.symbol,
                    'bids': order_book.bids,
                    'asks': order_book.asks,
                    'summary': summary
                }
            })
        else:
             return jsonify({'success': False, 'error': 'Failed to fetch order book data'})
            
    except Exception as e:
        logger.error(f"Error fetching order book: {e}")
        return jsonify({'success': False, 'error': str(e)})


def _parse_symbol_from_storage_key(symbol_key: str) -> str:
    """将存储文件名中的 symbol key 尽量还原为常见 CCXT symbol 形式。

    例：
      BTC_USDT -> BTC/USDT
      BTC_USDT_USDT -> BTC/USDT:USDT
    其他（期权合约等）保持原样。
    """
    s = str(symbol_key or '').strip()


def _timeframe_to_ms(timeframe: str) -> Optional[int]:
    tf = str(timeframe or '').strip().lower()
    m = re.fullmatch(r"(\d+)([mhdw])", tf)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    if n <= 0:
        return None
    if unit == 'm':
        return n * 60_000
    if unit == 'h':
        return n * 3_600_000
    if unit == 'd':
        return n * 86_400_000
    if unit == 'w':
        return n * 7 * 86_400_000
    return None


def _normalize_symbol_variants(symbol: str) -> List[str]:
    s = str(symbol or '').strip()
    if not s:
        return []
    # 常见输入：BTC/USDT、BTC/USDT:USDT
    s_up = s.upper()
    return [
        s_up.replace('/', '_').replace(':', '_'),
    ]


def _find_local_merged_file(market: str, exchange: str, symbol: str, timeframe: Optional[str] = None) -> tuple[Optional[Path], Optional[str]]:
    """在 data_manager_storage 下查找某 symbol/timeframe 的本地 merged 文件（优先 Parquet）。

    先尝试标准 Parquet 目录；找不到再尝试 legacy JSON；最后递归搜索。
    """
    market_l = str(market or 'spot').lower()
    exchange_l = str(exchange or 'binance').lower()
    root = Path(os.path.join(os.getcwd(), 'data_manager_storage', market_l, exchange_l))
    if not root.exists():
        return None, None

    symbol_variants = _normalize_symbol_variants(symbol)
    tfs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'] if not timeframe else [str(timeframe).strip().lower()]

    # 1) 标准 Parquet 路径（最快）
    for variant in symbol_variants:
        for tf in tfs:
            candidate = root / variant / tf / 'ohlcv_merged.parquet'
            if candidate.exists():
                return candidate, tf

    # 2) 直系路径（legacy JSON）
    for variant in symbol_variants:
        for tf in tfs:
            candidate = root / f"{variant}_{tf}_merged.json"
            if candidate.exists():
                return candidate, tf

    # 3) 递归搜索（兼容 variant 子目录，如 linear/cross 等）
    for variant in symbol_variants:
        for tf in tfs:
            pattern = f"{variant}_{tf}_merged.json"
            try:
                for p in root.rglob(pattern):
                    if p.is_file():
                        return p, tf
            except Exception:
                # rglob 在某些权限/长路径环境可能失败
                continue

    # 4) 递归搜索 Parquet
    try:
        for p in root.rglob('ohlcv_merged.parquet'):
            if not p.is_file():
                continue
            try:
                tf = p.parent.name
                symbol_key = p.parent.parent.name
                if symbol_key in symbol_variants and (not timeframe or tf in tfs):
                    return p, tf
            except Exception:
                continue
    except Exception:
        pass

    return None, None


def _extract_sorted_timestamps_ms(rows: Any) -> List[int]:
    if not isinstance(rows, list) or not rows:
        return []
    out: List[int] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        ts = r.get('timestamp')
        if ts is None:
            continue
        try:
            n = int(float(ts))
        except Exception:
            continue
        # 兼容秒级
        if n < 1_000_000_000_000:
            n *= 1000
        out.append(n)
    if not out:
        return []
    out = sorted(set(out))
    return out


def _iso_utc_from_ms(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v < 1_000_000_000_000:
        v *= 1000
    try:
        return datetime.fromtimestamp(v / 1000.0, tz=timezone(timedelta(hours=8))).isoformat()
    except Exception:
        return None


def _iso_utc_from_s(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    try:
        return datetime.fromtimestamp(v, tz=timezone(timedelta(hours=8))).isoformat()
    except Exception:
        return None


def _detect_ohlcv_gaps(timestamps_ms: List[int], step_ms: int, now_ms: Optional[int] = None, include_internal: bool = True, include_tail: bool = True, max_gaps: int = 200) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []
    if step_ms <= 0:
        return gaps
    if not timestamps_ms:
        return gaps

    # internal gaps
    if include_internal:
        prev = timestamps_ms[0]
        for ts in timestamps_ms[1:]:
            diff = ts - prev
            if diff > step_ms:
                miss_start = prev + step_ms
                miss_end = ts - step_ms
                if miss_start <= miss_end:
                    missing_bars = int((miss_end - miss_start) // step_ms) + 1
                    gaps.append({
                        'kind': 'internal',
                        'start_ms': miss_start,
                        'end_ms': miss_end,
                        'missing_bars': missing_bars,
                    })
                    if len(gaps) >= max_gaps:
                        break
            prev = ts

    # tail gap
    if include_tail and len(gaps) < max_gaps:
        last_ts = timestamps_ms[-1]
        cur = int(now_ms if now_ms is not None else time.time() * 1000)
        # 向下对齐到 timeframe 边界，避免“未收盘”K线
        aligned_now = cur - (cur % step_ms)
        if aligned_now - last_ts > step_ms:
            miss_start = last_ts + step_ms
            miss_end = aligned_now
            if miss_start <= miss_end:
                missing_bars = int((miss_end - miss_start) // step_ms) + 1
                gaps.append({
                    'kind': 'tail',
                    'start_ms': miss_start,
                    'end_ms': miss_end,
                    'missing_bars': missing_bars,
                })

    return gaps


def _fetch_ohlcv_as_rows(exchange: str, market: str, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    """拉取指定时间段 OHLCV，并标准化为 {timestamp, open, high, low, close, volume} 列表（timestamp=ms）。"""

    # onchain/social 不支持 OHLCV
    if str(market).lower() in ('onchain', 'social'):
        return []

    fetcher = get_or_create_fetcher(str(exchange).lower(), str(market).lower())
    if hasattr(fetcher, 'add_symbols'):
        try:
            fetcher.add_symbols([symbol])
        except Exception:
            pass

    # epoch(ms) -> UTC naive datetime（避免本地时区引入偏移 & tz 混用）
    start_dt = datetime.utcfromtimestamp(int(start_ms) / 1000)
    end_dt = datetime.utcfromtimestamp(int(end_ms) / 1000)

    # 优先 bulk（更适合区间/补洞）
    if hasattr(fetcher, 'fetch_ohlcv_bulk') and callable(getattr(fetcher, 'fetch_ohlcv_bulk')):
        try:
            df = fetcher.fetch_ohlcv_bulk(
                symbol=symbol,
                start_date=start_dt,
                end_date=end_dt,
                timeframe=timeframe,
                max_bars_per_request=1000,
            )
            if df is None or getattr(df, 'empty', False):
                return []
            
            # 使用向量化操作优化性能
            try:
                # 确保 DataFrame 副本，避免 SettingWithCopyWarning
                df_op = df.copy()

                # 统一生成 timestamp(ms) 列：兼容 timestamp 在 index / datetime 列 / 数值(ms|s|ns)
                if 'timestamp' not in df_op.columns:
                    if isinstance(df_op.index, pd.DatetimeIndex):
                        df_op = df_op.copy()
                        df_op['timestamp'] = (df_op.index.astype('int64') // 1_000_000).astype('int64')
                    elif df_op.index is not None and str(getattr(df_op.index, 'name', '')).lower() in ('timestamp', 'datetime', 'date'):
                        try:
                            idx_dt = pd.to_datetime(df_op.index, errors='coerce')
                            df_op = df_op.copy()
                            df_op['timestamp'] = (idx_dt.astype('int64') // 1_000_000).astype('int64')
                        except Exception:
                            raise KeyError('timestamp')
                    elif 'datetime' in df_op.columns:
                        dt = pd.to_datetime(df_op['datetime'], errors='coerce')
                        df_op['timestamp'] = (dt.astype('int64') // 1_000_000).astype('int64')
                    elif 'date' in df_op.columns:
                        dt = pd.to_datetime(df_op['date'], errors='coerce')
                        df_op['timestamp'] = (dt.astype('int64') // 1_000_000).astype('int64')
                    else:
                        raise KeyError('timestamp')
                else:
                    ts_col = df_op['timestamp']
                    if pd.api.types.is_datetime64_any_dtype(ts_col):
                        df_op['timestamp'] = (pd.to_datetime(ts_col, errors='coerce').astype('int64') // 1_000_000).astype('int64')
                    else:
                        ts_num = pd.to_numeric(ts_col, errors='coerce')
                        if ts_num.notna().any():
                            mx = float(ts_num.max())
                            # ns -> ms
                            if mx > 1_000_000_000_000_000:
                                df_op['timestamp'] = (ts_num // 1_000_000).astype('int64')
                            # seconds -> ms
                            elif mx < 1_000_000_000_000:
                                df_op['timestamp'] = (ts_num * 1000).astype('int64')
                            else:
                                df_op['timestamp'] = ts_num.astype('int64')
                        else:
                            dt = pd.to_datetime(ts_col, errors='coerce')
                            df_op['timestamp'] = (dt.astype('int64') // 1_000_000).astype('int64')
                
                # 过滤时间范围
                df_filtered = df_op[
                    (df_op['timestamp'] >= start_ms) & 
                    (df_op['timestamp'] <= end_ms)
                ].copy()
                
                if df_filtered.empty:
                    return []
                
                # 确保需要的列存在
                cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                for col in cols:
                    if col not in df_filtered.columns:
                        df_filtered[col] = 0.0
                
                # 确保数值类型
                for col in cols[1:]:  # opn/high/low/close/volume
                     df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0.0)

                # 批量转换为字典
                return df_filtered[cols].to_dict('records')
            
            except Exception as e_vec:
                logger.debug(f"Vectorized processing failed ({e_vec}), falling back to iterative method")
                # 如果向量化失败，回退到原来的迭代逻辑（保留原始逻辑作为 fallback）
                ts_ms = df_op['timestamp'] if 'timestamp' in df_op.columns else pd.Series([], dtype='int64')
                out = []
                for i in range(len(df_op)):
                    tms = int(ts_ms.iloc[i]) if i < len(ts_ms) else None
                    if tms is None:
                        continue
                    if tms < start_ms or tms > end_ms:
                        continue
                    out.append({
                        'timestamp': tms,
                        'open': float(df_op['open'].iloc[i]) if 'open' in df_op.columns else float(df_op.iloc[i].get('open', 0)),
                        'high': float(df_op['high'].iloc[i]) if 'high' in df_op.columns else float(df_op.iloc[i].get('high', 0)),
                        'low': float(df_op['low'].iloc[i]) if 'low' in df_op.columns else float(df_op.iloc[i].get('low', 0)),
                        'close': float(df_op['close'].iloc[i]) if 'close' in df_op.columns else float(df_op.iloc[i].get('close', 0)),
                        'volume': float(df_op['volume'].iloc[i]) if 'volume' in df_op.columns else float(df_op.iloc[i].get('volume', 0)),
                    })
                return out
        except Exception as e:
            logger.warning(f"fetch_ohlcv_bulk failed, fallback: {e}")

    # 回退：fetch-range（最多 1000 bars，一般用于小缺口）
    if hasattr(fetcher, 'fetch_ohlcv') and callable(getattr(fetcher, 'fetch_ohlcv')):
        try:
            raw = fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=start_ms, limit=1000)
            out = []
            for item in (raw or []):
                if isinstance(item, (list, tuple)) and len(item) >= 6:
                    tms = int(item[0])
                    if tms < 1_000_000_000_000:
                        tms *= 1000
                    if tms < start_ms or tms > end_ms:
                        continue
                    out.append({
                        'timestamp': tms,
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5]),
                    })
            return out
        except Exception as e:
            logger.warning(f"fetch_ohlcv fallback failed: {e}")
            return []

    return []


def _save_ohlcv_parquet(market: str, exchange: str, symbol: str, timeframe: str, rows: List[Dict[str, Any]]) -> Optional[Path]:
    """将 OHLCV rows 合并保存为 Parquet（标准路径）。"""
    try:
        if not rows:
            return None
        market_l = str(market or 'spot').lower()
        exchange_l = str(exchange or 'binance').lower()
        symbol_key = str(symbol or '').replace('/', '_').replace(':', '_').strip() or 'unknown'
        timeframe_l = str(timeframe or '').strip().lower() or '1h'

        root = Path(os.path.join(os.getcwd(), 'data_manager_storage'))
        out_dir = root / market_l / exchange_l / symbol_key / timeframe_l
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'ohlcv_merged.parquet'

        df_new = pd.DataFrame(rows)
        if df_new.empty or 'timestamp' not in df_new.columns:
            return None

        ts = pd.to_numeric(df_new['timestamp'], errors='coerce')
        ts = ts.where(ts >= 1_000_000_000_000, ts * 1000)
        df_new = df_new.copy()
        df_new['timestamp'] = ts
        df_new = df_new.dropna(subset=['timestamp'])
        df_new['timestamp'] = df_new['timestamp'].astype('int64')

        if out_path.exists():
            try:
                df_old = pd.read_parquet(out_path)
            except Exception:
                df_old = pd.DataFrame()
            df = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new
        else:
            df = df_new

        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='last')

        df.to_parquet(out_path, index=False, compression='snappy')
        return out_path
    except Exception as e:
        logger.warning(f"Parquet save failed: {e}")
        return None
    if not s:
        return s
    parts = s.split('_')
    if len(parts) == 2 and all(parts):
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 3 and all(parts):
        return f"{parts[0]}/{parts[1]}:{parts[2]}"
    return s


def _try_extract_time_range_from_json(path: Path, max_bytes: int = 20 * 1024 * 1024):
    """尽量从 JSON 文件中提取 (start_ts_ms, end_ts_ms, count)。

    为避免扫描过慢：仅在文件大小 <= max_bytes 时尝试完整 json.load。
    """
    try:
        if not path.exists() or not path.is_file():
            return None
        if path.stat().st_size > max_bytes:
            return None

        with path.open('r', encoding='utf-8') as f:
            obj = json.load(f)

        data = None
        if isinstance(obj, list):
            data = obj
        elif isinstance(obj, dict):
            if isinstance(obj.get('data'), list):
                data = obj.get('data')
            elif isinstance(obj.get('results'), dict):
                # web merged results 结构：{results: {symbol: {data:[...]}}}
                # 这里不展开，留给上层
                return None

        if not isinstance(data, list) or len(data) == 0:
            return None

        def _get_ts_ms(item):
            if item is None:
                return None
            # 常见：dict {'timestamp': ms}
            if isinstance(item, dict):
                ts = item.get('timestamp')
                if ts is None:
                    ts = item.get('time') or item.get('t')
                if ts is None:
                    ts = item.get('datetime')
                try:
                    if ts is None:
                        return None
                    if isinstance(ts, (int, float)):
                        tsn = int(ts)
                        return tsn if tsn > 10**12 else tsn * 1000
                    # datetime string
                    return int(pd.Timestamp(ts).timestamp() * 1000)
                except Exception:
                    return None

            # 常见：OHLCV list/tuple [ts, o, h, l, c, v]
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                ts = item[0]
                try:
                    if isinstance(ts, (int, float)):
                        tsn = int(ts)
                        return tsn if tsn > 10**12 else tsn * 1000
                    return int(pd.Timestamp(ts).timestamp() * 1000)
                except Exception:
                    return None
            return None

        first_ts = _get_ts_ms(data[0])
        last_ts = _get_ts_ms(data[-1])
        if first_ts is None or last_ts is None:
            return None

        start_ts = int(min(first_ts, last_ts))
        end_ts = int(max(first_ts, last_ts))
        return {
            'start_ts': start_ts,
            'end_ts': end_ts,
            'count': len(data)
        }
    except Exception:
        return None


def _scan_local_storage_index(
    root_dir: Path,
    market: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframe: Optional[str] = None,
    max_files: int = 5000,
    include_range: bool = False,
):
    """扫描 data_manager_storage 下的最终数据文件，生成索引。使用了新的 DataInventory 工具。"""
    try:
        from crypto_data_system.storage.data_inventory import DataInventory
        inventory = DataInventory(str(root_dir))
        df = inventory.scan()
        
        if df.empty:
             return {
                'entries': [],
                'total_files': 0,
                'symbols': [],
                'markets': [],
                'exchanges': [],
                'timeframes': [],
            }
            
        # 过滤 (Filters)
        if market:
             df = df[df['type'].astype(str).str.lower() == str(market).lower()]
        if exchange:
             df = df[df['exchange'].astype(str).str.lower() == str(exchange).lower()]
        if timeframe:
             df = df[df['timeframe'].astype(str).str.lower() == str(timeframe).lower()]
             
        # 统计唯一值
        all_symbols = sorted(df['symbol'].unique().tolist())
        all_markets = sorted(df['type'].unique().tolist())
        all_exchanges = sorted(df['exchange'].unique().tolist())
        all_timeframes = sorted(df['timeframe'].unique().tolist())
        
        # 限制返回数量 (如果前端只需要 symbol 列表，可以不用返回 entries)
        entries = df.head(max_files).to_dict('records')
        
        # 格式化时间 (转换为中国时间 UTC+8 显示)
        for entry in entries:
            # Pandas time to CST String
            try:
                if entry.get('start_time') and not pd.isna(entry.get('start_time')):
                    # 假设 start_time 是 UTC (naive timestamp from unit='ms')
                    # 加 8 小时
                    start_cst = entry['start_time'] + pd.Timedelta(hours=8)
                    entry['start_date'] = str(start_cst) # .strftime('%Y-%m-%d %H:%M:%S')
                
                if entry.get('end_time') and not pd.isna(entry.get('end_time')):
                    end_cst = entry['end_time'] + pd.Timedelta(hours=8)
                    entry['end_date'] = str(end_cst)
                    
                if entry.get('last_modified') and not pd.isna(entry.get('last_modified')):
                    # last_modified 通常是本地系统时间 (如果是国内机器已经是CST，如果是UTC容器则需调整)
                    # 假设文件系统时间已经是本地时间
                    entry['last_modified'] = str(entry['last_modified'])
            except Exception:
                pass
                
            # Rename 'type' to 'market' for frontend compatibility
            entry['market'] = entry.get('type')
            
        return {
            'entries': entries,
            'total_files': len(entries),
            'symbols': all_symbols,
            'markets': all_markets,
            'exchanges': all_exchanges,
            'timeframes': all_timeframes,
        }
            
    except Exception as e:
        logger.error(f"Scan inventory failed: {e}")
        # Fallback to empty
        return {
            'entries': [], 
            'error': str(e)
        }




# ==================== 异步任务管理 ====================

# 任务队列和状态存储
task_store = {}  # task_id -> {status, progress, results, error}
task_store_lock = Lock()
task_executor = ThreadPoolExecutor(max_workers=3)

# AlphaGen / RL 任务队列（与数据获取/保存任务隔离，避免互相阻塞）
alphagen_task_store = {}  # task_id -> AlphaGenTask
alphagen_task_store_lock = Lock()
alphagen_executor = ThreadPoolExecutor(max_workers=1)

# ML 任务队列（与数据获取/保存任务隔离）
ml_task_store = {}  # task_id -> MLTask
ml_task_store_lock = Lock()
ml_executor = ThreadPoolExecutor(max_workers=1)


class AlphaGenTask:
    """AlphaGen/RL 异步任务类（训练/导出）"""

    def __init__(self, task_id: str, kind: str, params: Dict[str, Any]):
        self.task_id = task_id
        self.kind = kind
        self.params = params
        self.status = 'queued'  # queued, running, completed, error
        self.progress = 0
        self.result: Dict[str, Any] = {}
        self.logs: List[str] = []
        self.error: Optional[str] = None
        self.start_time_ts = datetime.now()
        self.cancelled = False
        self.timeout_sec = 6 * 3600
        self.proc = None

    def add_log(self, line: str, *, max_lines: int = 500) -> None:
        s = str(line).rstrip('\n')
        if not s:
            return
        self.logs.append(s)
        if len(self.logs) > int(max_lines):
            self.logs = self.logs[-int(max_lines):]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'kind': self.kind,
            'params': self.params,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'logs': self.logs,
            'elapsed_seconds': int((datetime.now() - self.start_time_ts).total_seconds()),
            'cancelled': self.cancelled,
            'timeout_sec': self.timeout_sec,
        }


class MLTask:
    """ML 训练/评估任务"""

    def __init__(self, task_id: str, params: Dict[str, Any]):
        self.task_id = task_id
        self.params = params
        self.status = 'queued'  # queued, running, completed, error
        self.progress = 0
        self.result: Dict[str, Any] = {}
        self.logs: List[str] = []
        self.error: Optional[str] = None
        self.start_time_ts = datetime.now()
        self.cancelled = False

    def add_log(self, line: str, *, max_lines: int = 500) -> None:
        s = str(line).rstrip('\n')
        if not s:
            return
        self.logs.append(s)
        if len(self.logs) > int(max_lines):
            self.logs = self.logs[-int(max_lines):]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'params': self.params,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'logs': self.logs,
            'elapsed_seconds': int((datetime.now() - self.start_time_ts).total_seconds()),
            'cancelled': self.cancelled,
        }


def _alphagen_base_dir() -> Path:
    base = Path(project_root) / 'data_manager_storage' / 'web' / 'alphagen'
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe_join(base_dir: Path, rel_path: str) -> Path:
    # 防止路径穿越
    rel = str(rel_path or '').replace('\\', '/').lstrip('/').strip()
    p = (base_dir / rel).resolve()
    b = base_dir.resolve()
    if b not in p.parents and p != b:
        raise ValueError('Invalid path')
    return p


def _resolve_alphagen_panel_path(panel_path: str) -> Path:
    raw = str(panel_path or '').strip()
    if not raw:
        raise ValueError('panel path is required')
    p = Path(raw)
    if not p.is_absolute():
        p = (Path(project_root) / raw).resolve()
    else:
        p = p.resolve()

    roots = [
        Path(project_root).resolve(),
        (Path(project_root) / 'data_manager_storage').resolve(),
        _alphagen_base_dir().resolve(),
    ]
    if not any((r == p) or (r in p.parents) for r in roots):
        raise ValueError('panel path outside allowed roots')
    return p


def _parse_ts_value(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # numeric timestamp
    try:
        if re.fullmatch(r'-?\d+(\.\d+)?', s):
            val = float(s)
            if abs(val) > 1e12:  # ms
                return datetime.utcfromtimestamp(val / 1000.0)
            if abs(val) > 1e9:  # seconds
                return datetime.utcfromtimestamp(val)
    except Exception:
        pass

    try:
        ts = pd.to_datetime(s, errors='coerce', utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _panel_quick_stats(path: Path, *, max_rows_scan: int = 2_000_000, sample_rows: int = 200) -> Dict[str, Any]:
    st = path.stat()
    info: Dict[str, Any] = {
        'path': str(path.resolve()),
        'size_bytes': int(st.st_size),
        'modified_at': _iso_utc_from_s(st.st_mtime),
    }

    rows_scanned = 0
    rows_truncated = False
    head_rows: List[List[str]] = []
    tail_rows: deque = deque(maxlen=sample_rows)
    columns: List[str] = []

    with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            info.update({'columns': [], 'row_count': 0, 'rows_scanned': 0, 'rows_truncated': False})
            return info
        columns = [str(c).strip() for c in header]
        for row in reader:
            rows_scanned += 1
            if rows_scanned <= sample_rows:
                head_rows.append(row)
            tail_rows.append(row)
            if rows_scanned >= max_rows_scan:
                rows_truncated = True
                break

    row_count = f'>={rows_scanned}' if rows_truncated else rows_scanned
    info.update({
        'columns': columns,
        'row_count': row_count,
        'rows_scanned': rows_scanned,
        'rows_truncated': rows_truncated,
    })

    lower_cols = [c.lower() for c in columns]
    time_candidates = ['timestamp', 'ts', 'ts_ms', 'datetime', 'date', 'time']
    time_col = None
    for cand in time_candidates:
        if cand in lower_cols:
            time_col = columns[lower_cols.index(cand)]
            break

    if time_col:
        idx = columns.index(time_col)
        samples = []
        for row in head_rows:
            if idx < len(row):
                samples.append(row[idx])
        for row in list(tail_rows):
            if idx < len(row):
                samples.append(row[idx])

        parsed = [ts for ts in (_parse_ts_value(v) for v in samples) if ts is not None]
        if parsed:
            info['time_column'] = time_col
            info['time_range'] = {
                'start': min(parsed).isoformat(),
                'end': max(parsed).isoformat(),
                'approx': True,
            }

    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in lower_cols]
    info['required_missing'] = missing
    info['ok'] = len(missing) == 0
    return info


def _infer_multi_pattern(panel_path: str) -> Dict[str, Any]:
    raw = str(panel_path or '').strip()
    if not raw:
        return {'pattern': None, 'error': 'panel is required'}

    # if explicit glob
    if any(ch in raw for ch in ['*', '?', '[']):
        return {'pattern': raw, 'source': 'user_glob'}

    p = Path(raw)
    if not p.is_absolute():
        p = (Path(project_root) / raw).resolve()
    else:
        p = p.resolve()

    roots = [
        Path(project_root).resolve(),
        (Path(project_root) / 'data_manager_storage').resolve(),
        _alphagen_base_dir().resolve(),
    ]
    if not any((r == p) or (r in p.parents) for r in roots):
        return {'pattern': None, 'error': 'panel path outside allowed roots'}

    if p.exists() and p.is_file():
        base_dir = p.parent
        base_name = p.name
        m = re.match(r'^alphagen_panel_(.+)_(\d+[a-zA-Z]+)\.csv$', base_name)
        if m:
            tf = m.group(2)
            pattern = str((base_dir / f"alphagen_panel_*_{tf}.csv").resolve())
            return {'pattern': pattern, 'source': 'inferred'}
        # fallback: match all csv in same dir
        return {'pattern': str((base_dir / '*.csv').resolve()), 'source': 'fallback_dir'}

    # If path doesn't exist, treat it as a relative pattern in project root
    return {'pattern': str((Path(project_root) / raw).resolve()), 'source': 'relative'}


def _list_spot_merged_files(*, exchange: str = 'binance') -> List[Dict[str, Any]]:
    """List local spot OHLCV sources for AlphaGen panel building.

    Supports both legacy JSON and canonical parquet produced by the data system:
    - data_manager_storage/spot/<exchange>/*_merged.json
    - data_manager_storage/spot/<exchange>/<SYMBOL_KEY>/<TIMEFRAME>/ohlcv_merged.parquet
    """
    out: List[Dict[str, Any]] = []
    root = Path(project_root) / 'data_manager_storage' / 'spot' / str(exchange).lower()
    if not root.exists():
        return out

    inv_map: Dict[str, Dict[str, Any]] = {}
    try:
        from crypto_data_system.storage.data_inventory import DataInventory

        inv = DataInventory(str(Path(project_root) / 'data_manager_storage'))
        df = inv.scan(consolidate=False)
        if not df.empty:
            df = df[(df['type'].astype(str).str.lower() == 'spot') & (df['exchange'].astype(str).str.lower() == str(exchange).lower())]
            for _, row in df.iterrows():
                try:
                    p = Path(str(row.get('file_path') or '')).resolve()
                    inv_map[str(p)] = row.to_dict()
                except Exception:
                    continue
    except Exception:
        inv_map = {}

    def _calc_gap_rate(info: Dict[str, Any], tf: Optional[str]) -> Optional[float]:
        try:
            if not info or not tf:
                return None
            start = info.get('start_time')
            end = info.get('end_time')
            count = info.get('count')
            if start is None or end is None or count is None:
                return None
            from crypto_data_system.utils.date_utils import calculate_timeframe_seconds

            sec = calculate_timeframe_seconds(str(tf))
            if not sec:
                return None
            start_ts = pd.to_datetime(start).to_pydatetime()
            end_ts = pd.to_datetime(end).to_pydatetime()
            if end_ts <= start_ts:
                return None
            expected = int((end_ts - start_ts).total_seconds() / sec) + 1
            if expected <= 0:
                return None
            gap_rate = 1.0 - (float(count) / float(expected))
            return max(0.0, min(1.0, gap_rate))
        except Exception:
            return None

    def _symbol_from_key(symbol_key: str) -> Optional[str]:
        s = str(symbol_key or '').strip()
        if not s:
            return None
        s = s.upper()
        if '/' in s:
            return s
        if '_' in s:
            parts = [p for p in s.split('_') if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        return s

    # 1) canonical parquet (preferred)
    try:
        for p in sorted(root.rglob('ohlcv_merged.parquet')):
            try:
                timeframe = p.parent.name
                symbol_key = p.parent.parent.name
            except Exception:
                timeframe = None
                symbol_key = None

            inv = inv_map.get(str(p.resolve()))
            start_time = inv.get('start_time') if inv else None
            end_time = inv.get('end_time') if inv else None
            count = inv.get('count') if inv else None
            gap_rate = _calc_gap_rate(inv, timeframe) if inv else None

            panel_recommend = None
            if symbol_key and timeframe:
                panel_name = f"alphagen_panel_{symbol_key}_{timeframe}.csv"
                panel_recommend = str((_alphagen_base_dir() / 'panels' / panel_name).resolve())

            out.append({
                'name': p.name,
                'path': str(p.resolve()),
                'symbol': _symbol_from_key(symbol_key) if symbol_key else None,
                'timeframe': timeframe,
                'size': int(p.stat().st_size) if p.exists() else None,
                'modified_at': _iso_utc_from_s(p.stat().st_mtime) if p.exists() else None,
                'format': 'parquet',
                'start_time': str(start_time) if start_time is not None else None,
                'end_time': str(end_time) if end_time is not None else None,
                'count': int(count) if count is not None else None,
                'gap_rate': gap_rate,
                'panel_recommend': panel_recommend,
            })
    except Exception:
        pass

    # 2) legacy merged json
    for p in sorted(root.glob('*_merged.json')):
        name = p.name
        parts = name.split('_')
        timeframe = None
        symbol = None
        try:
            # e.g. BTC_USDT_1m_merged.json
            if len(parts) >= 4 and parts[-2] == 'merged.json':
                # not expected; keep fallback
                pass
        except Exception:
            pass

        # Robust parse: take last 2 tokens as "<tf>_merged.json".
        try:
            if name.endswith('_merged.json'):
                core = name[:-len('_merged.json')]
                toks = core.split('_')
                if len(toks) >= 3:
                    timeframe = toks[-1]
                    base = toks[0]
                    quote = toks[1]
                    symbol = f"{base}/{quote}"
        except Exception:
            timeframe = None
            symbol = None

        out.append({
            'name': name,
            'path': str(p.resolve()),
            'symbol': symbol,
            'timeframe': timeframe,
            'size': int(p.stat().st_size) if p.exists() else None,
                'modified_at': _iso_utc_from_s(p.stat().st_mtime) if p.exists() else None,
            'format': 'json',
        })

    # De-dup by path
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for it in out:
        p = it.get('path')
        if not p or p in seen:
            continue
        seen.add(p)
        deduped.append(it)
    return deduped


def _infer_swap_aux_paths(*, exchange: str, symbol: str) -> Dict[str, Optional[str]]:
    """Infer local swap funding/OI files for given symbol.

    Uses existing storage convention:
      data_manager_storage/swap/<exchange>/linear/<BASE>_<QUOTE>_funding_history.json
      data_manager_storage/swap/<exchange>/linear/<BASE>_<QUOTE>_open_interest_1h.json
    """
    ex = str(exchange or 'binance').lower()
    sym = str(symbol or '').strip().upper().replace('-', '/').replace('_', '/').replace(' ', '')
    if '/' not in sym:
        return {'funding': None, 'open_interest': None}
    base, quote = sym.split('/', 1)
    stem = f"{base}_{quote}"
    root = Path(project_root) / 'data_manager_storage' / 'swap' / ex / 'linear'
    funding = root / f"{stem}_funding_history.json"
    oi = root / f"{stem}_open_interest_1h.json"
    return {
        'funding': str(funding.resolve()) if funding.exists() else None,
        'open_interest': str(oi.resolve()) if oi.exists() else None,
    }


def _run_subprocess_task(task: AlphaGenTask, cmd: List[str], *, cwd: Optional[str] = None) -> None:
    """运行子进程并把 stdout/stderr 作为日志写入任务。"""
    try:
        task.status = 'running'
        task.progress = 1
        task.add_log(' '.join(cmd))

        proc = subprocess.Popen(
            cmd,
            cwd=cwd or project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        task.proc = proc
        assert proc.stdout is not None

        deadline = time.time() + task.timeout_sec if task.timeout_sec else None

        while True:
            if task.cancelled:
                try:
                    proc.terminate()
                except Exception:
                    pass
                task.status = 'cancelled'
                task.error = 'cancelled by user'
                task.add_log(task.error)
                task.progress = 0
                return

            if deadline and time.time() > deadline:
                try:
                    proc.terminate()
                except Exception:
                    pass
                task.status = 'timeout'
                task.error = f'timeout after {task.timeout_sec}s'
                task.add_log(task.error)
                task.progress = 0
                return

            try:
                out, _ = proc.communicate(timeout=1)
                if out:
                    for line in out.splitlines():
                        task.add_log(line)
                rc = proc.returncode
                if rc != 0:
                    raise RuntimeError(f'subprocess exited with code {rc}')
                task.progress = 100
                task.status = 'completed'
                return
            except subprocess.TimeoutExpired as e:
                if e.output:
                    for line in e.output.splitlines():
                        task.add_log(line)
                continue
    except Exception as e:
        task.status = 'error'
        task.error = f'{type(e).__name__}: {e}'
        task.add_log(task.error)
        task.progress = 0



class SaveTask:
    """保存任务类"""
    def __init__(self, task_id: str, market: str, exchange: str, symbols: List[str], 
                 timeframe: str, start_time: Optional[int] = None, end_time: Optional[int] = None):
        self.task_id = task_id
        self.market = market
        self.exchange = exchange
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.status = 'queued'  # queued, running, completed, error
        self.progress = 0
        self.results = {}
        self.error = None
        self.start_time_ts = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'results': self.results,
            'error': self.error,
            'symbols_total': len(self.symbols),
            'symbols_processed': len([r for r in self.results.values() if r.get('status')]),
            'elapsed_seconds': int((datetime.now() - self.start_time_ts).total_seconds())
        }


class FetchTask:
    """获取数据任务类"""
    def __init__(self, task_id: str, market: str, exchange: str, symbols: List[str],
                 timeframe: str, limit: Optional[int] = None, 
                 start_time: Optional[int] = None, end_time: Optional[int] = None,
                 data_type: Optional[str] = None, include: Optional[List[str]] = None,
                 auto_save: bool = True, storage_format: str = 'parquet'):
        self.task_id = task_id
        self.market = market
        self.exchange = exchange
        self.symbols = symbols
        self.timeframe = timeframe
        self.limit = limit
        self.start_time = start_time
        self.end_time = end_time
        self.data_type = data_type or 'ohlcv'
        self.include = include
        self.auto_save = bool(auto_save)
        self.storage_format = str(storage_format or 'parquet').lower()
        self.status = 'queued'
        self.progress = 0
        self.results = {}
        self.error = None
        self.start_time_ts = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'results': self.results,
            'error': self.error,
            'data_type': getattr(self, 'data_type', 'ohlcv'),
            'auto_save': getattr(self, 'auto_save', True),
            'storage_format': getattr(self, 'storage_format', 'parquet'),
            'symbols_total': len(self.symbols),
            'symbols_processed': len([r for r in self.results.values() if r.get('status')]),
            'elapsed_seconds': int((datetime.now() - self.start_time_ts).total_seconds())
        }


def execute_save_task(task: SaveTask):
    """后台线程执行保存任务"""
    try:
        task.status = 'running'
        market_lower = str(task.market).lower()

        def _safe_filename_component(value: str, max_len: int = 180) -> str:
            """Make a string safe for use as a Windows filename component."""
            s = str(value or '').strip()
            if not s:
                return 'empty'
            # Windows forbidden: <>:"/\|?* and control chars
            s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s)
            # Avoid trailing dots/spaces (Windows disallows)
            s = s.rstrip(' .')
            if not s:
                s = 'empty'
            if len(s) > max_len:
                s = s[:max_len]
            return s

        # onchain/social：使用文件存储直接保存（非 OHLCV）
        if market_lower in ('onchain', 'social'):
            fetcher = get_or_create_fetcher(task.exchange, task.market)
            store = FileDataManager(sub_dir=f"{market_lower}/{task.exchange}")

            total = len(task.symbols)
            for idx, symbol in enumerate(task.symbols):
                try:
                    if market_lower == 'onchain':
                        payload = _fetch_onchain_snapshot(fetcher, symbol)
                    else:
                        period = _infer_social_period(task.start_time, task.end_time, task.timeframe)
                        payload = _fetch_social_snapshot(fetcher, symbol, period)

                    safe_symbol = _safe_filename_component(symbol)
                    safe_tf = _safe_filename_component(task.timeframe)
                    ok = store.save_timestamped(f"{safe_symbol}_{safe_tf}", payload, timestamp=True)
                    task.results[symbol] = {
                        'status': 'success' if ok else 'error',
                        'message': 'Data saved' if ok else 'Save failed'
                    }
                except Exception as e:
                    logger.warning(f"Error saving {symbol}: {e}")
                    task.results[symbol] = {
                        'status': 'error',
                        'error': str(e)
                    }

                task.progress = int((idx + 1) / total * 100)

            task.status = 'completed'
            task.progress = 100
            return

        manager = get_or_create_manager(task.market, task.exchange, getattr(task, 'storage_format', 'parquet'))
        manager.add_symbols(task.symbols)

        total = len(task.symbols)
        for idx, symbol in enumerate(task.symbols):
            try:
                if hasattr(manager, 'fetch_and_save'):
                    start_dt = datetime.utcfromtimestamp(task.start_time/1000) if task.start_time else None
                    end_dt = datetime.utcfromtimestamp(task.end_time/1000) if task.end_time else None
                    ok = manager.fetch_and_save(symbol, task.timeframe, start_date=start_dt, end_date=end_dt)
                    task.results[symbol] = {
                        'status': 'success' if ok else 'error',
                        'message': 'Data saved' if ok else 'Save failed'
                    }
                else:
                    task.results[symbol] = {
                        'status': 'skipped',
                        'message': 'DataManager does not support fetch_and_save'
                    }
            except Exception as e:
                logger.warning(f"Error saving {symbol}: {e}")
                task.results[symbol] = {
                    'status': 'error',
                    'error': str(e)
                }

            # 更新进度
            task.progress = int((idx + 1) / total * 100)

        task.status = 'completed'
        task.progress = 100

    except Exception as e:
        logger.error(f"Task {task.task_id} failed: {e}")
        task.status = 'error'
        task.error = str(e)


def execute_fetch_task(task: FetchTask):
    """后台线程执行获取数据任务"""
    try:
        task.status = 'running'
        market_lower = str(task.market).lower()
        data_type = str(getattr(task, 'data_type', 'ohlcv') or 'ohlcv').lower()
        include = getattr(task, 'include', None)
        fetcher = get_or_create_fetcher(task.exchange, task.market)

        def _to_ts_ms(ts_val):
            if ts_val is None:
                return None
            if isinstance(ts_val, (int, float)):
                return int(ts_val) if ts_val > 1e12 else int(ts_val * 1000)

            # pandas 的 NaT/NaN 会有 timestamp 方法但会抛错
            try:
                if pd.isna(ts_val):
                    return None
            except Exception:
                pass

            if hasattr(ts_val, 'timestamp') and callable(getattr(ts_val, 'timestamp')):
                try:
                    return int(ts_val.timestamp() * 1000)
                except Exception:
                    return None
            try:
                ts = pd.Timestamp(ts_val)
                if pd.isna(ts):
                    return None
                return int(ts.timestamp() * 1000)
            except Exception:
                return None

        def _serialize_trade(trade_obj, fallback_ts_ms: int) -> Dict[str, Any]:
            if trade_obj is None:
                return {}
            if hasattr(trade_obj, 'to_dict'):
                d = trade_obj.to_dict()
            elif hasattr(trade_obj, '__dict__'):
                d = dict(trade_obj.__dict__)
            elif isinstance(trade_obj, dict):
                d = dict(trade_obj)
            else:
                d = {'value': str(trade_obj)}

            ts_ms = _to_ts_ms(d.get('timestamp')) or fallback_ts_ms
            d['timestamp'] = ts_ms
            d['datetime'] = _iso_utc_from_ms(ts_ms)
            return d

        def _serialize_orderbook_summary(orderbook_obj, fallback_ts_ms: int) -> Dict[str, Any]:
            if orderbook_obj is None:
                return {}

            ts_ms = fallback_ts_ms
            if hasattr(orderbook_obj, 'timestamp'):
                ts_ms = _to_ts_ms(getattr(orderbook_obj, 'timestamp', None)) or fallback_ts_ms
            elif isinstance(orderbook_obj, dict):
                ts_ms = _to_ts_ms(orderbook_obj.get('timestamp')) or fallback_ts_ms

            summary = {}
            # 尝试获取原始 bids/asks 用于前端深度图
            raw_bids = []
            raw_asks = []
            if hasattr(orderbook_obj, 'bids'):
                raw_bids = orderbook_obj.bids[:50] if isinstance(orderbook_obj.bids, list) else []
            if hasattr(orderbook_obj, 'asks'):
                raw_asks = orderbook_obj.asks[:50] if isinstance(orderbook_obj.asks, list) else []
            
            # 若 bids/asks 为空但有 to_dict
            if not raw_bids and hasattr(orderbook_obj, 'to_dict'):
                d = orderbook_obj.to_dict()
                raw_bids = d.get('bids', [])[:50]
                raw_asks = d.get('asks', [])[:50]

            if hasattr(orderbook_obj, 'get_summary'):
                try:
                    summary = orderbook_obj.get_summary(levels=5)
                except Exception:
                    summary = {}
            elif isinstance(orderbook_obj, dict):
                summary = {
                    'best_bid': orderbook_obj.get('best_bid'),
                    'best_ask': orderbook_obj.get('best_ask'),
                    'spread': orderbook_obj.get('spread'),
                }
                raw_bids = orderbook_obj.get('bids', [])[:50]
                raw_asks = orderbook_obj.get('asks', [])[:50]

            out = {
                'timestamp': ts_ms,
                'datetime': _iso_utc_from_ms(ts_ms),
                'best_bid': summary.get('best_bid'),
                'best_ask': summary.get('best_ask'),
                'spread': summary.get('spread'),
                'spread_percent': summary.get('spread_percent'),
                'bid_total': summary.get('bid_total'),
                'ask_total': summary.get('ask_total'),
                'imbalance': summary.get('imbalance'),
                'bids_top5': summary.get('bids_top5'),
                'asks_top5': summary.get('asks_top5'),
                # 新增用于可视化的数据
                'bids': raw_bids,
                'asks': raw_asks
            }
            return out

        def _json_safe(obj, fallback_ts_ms: int):
            """把常见对象（dataclass/pd.Timestamp 等）转换为 JSON 可序列化结构。"""
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj

            # 时间对象
            ts_ms = _to_ts_ms(obj)
            if ts_ms is not None and isinstance(obj, (pd.Timestamp, datetime)):
                return ts_ms

            # dataclass / 模型
            if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                try:
                    obj = obj.to_dict()
                except Exception:
                    obj = dict(getattr(obj, '__dict__', {}) or {})

            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if isinstance(v, (pd.Timestamp, datetime)):
                        ms = _to_ts_ms(v)
                        out[k] = ms if ms is not None else None
                        if k == 'timestamp' and ms is not None:
                            out['datetime'] = _iso_utc_from_ms(ms)
                    else:
                        out[k] = _json_safe(v, fallback_ts_ms)
                return out

            if isinstance(obj, (list, tuple)):
                return [_json_safe(v, fallback_ts_ms) for v in obj]

            # 兜底
            return str(obj)

        # onchain/social：按“指标/舆情”快照获取（非 OHLCV）
        if market_lower in ('onchain', 'social'):
            total = len(task.symbols)
            for idx, symbol in enumerate(task.symbols):
                try:
                    if market_lower == 'onchain':
                        payload = _fetch_onchain_snapshot(fetcher, symbol)
                    else:
                        period = _infer_social_period(task.start_time, task.end_time, task.timeframe)
                        payload = _fetch_social_snapshot(fetcher, symbol, period)

                    task.results[symbol] = {
                        'status': 'success',
                        'data': [payload]
                    }
                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
                    task.results[symbol] = {
                        'status': 'error',
                        'error': str(e)
                    }

                task.progress = int((idx + 1) / total * 100)

            task.status = 'completed'
            task.progress = 100
            return

        # swap/future：支持按 data_type 批量获取（非 OHLCV）
        if market_lower in ('swap', 'future') and data_type != 'ohlcv':
            total = len(task.symbols)
            for idx, symbol in enumerate(task.symbols):
                try:
                    now_ms = int(datetime.now().timestamp() * 1000)
                    include_set = set((include or []))

                    if data_type == 'ticker':
                        t = fetcher.fetch_ticker(symbol)
                        ts_ms = _to_ts_ms(t.get('timestamp')) or now_ms
                        payload = {
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'last': t.get('last'),
                            'bid': t.get('bid'),
                            'ask': t.get('ask'),
                            'high': t.get('high'),
                            'low': t.get('low'),
                            'volume': t.get('baseVolume') or t.get('quoteVolume'),
                            'raw': _json_safe(t, now_ms),
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'orderbook':
                        ob = fetcher.fetch_orderbook(symbol, limit=task.limit or 50)
                        payload = _serialize_orderbook_summary(ob, now_ms)
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'trades':
                        trades = fetcher.fetch_trades(symbol, limit=task.limit or 50)
                        trades_ser = [_serialize_trade(t, now_ms) for t in (trades or [])][:200]
                        # 表格友好：展示最后一条成交的关键字段
                        last_trade = trades_ser[-1] if trades_ser else {}
                        payload = {
                            'timestamp': last_trade.get('timestamp') or now_ms,
                            'datetime': last_trade.get('datetime') or _iso_utc_from_ms(now_ms),
                            'price': last_trade.get('price'),
                            'amount': last_trade.get('amount'),
                            'side': last_trade.get('side'),
                            'trade_id': last_trade.get('trade_id') or last_trade.get('id'),
                            'trades': trades_ser,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'funding_rate':
                        if not hasattr(fetcher, 'fetch_funding_rate'):
                            raise AttributeError('fetch_funding_rate not supported')

                        # swap 优先返回历史序列（用于可视化折线），否则回退为单点
                        if market_lower == 'swap' and hasattr(fetcher, 'fetch_funding_rate_history') and task.start_time:
                            limit = int(task.limit or 200)
                            limit = max(1, min(limit, 1000))
                            hist = fetcher.fetch_funding_rate_history(symbol, since=task.start_time, limit=limit)
                            hist_s = _json_safe(hist, now_ms) if hist is not None else []
                            rows = []
                            for fr_d in (hist_s or []):
                                if not isinstance(fr_d, dict):
                                    continue
                                ts_ms = _to_ts_ms(fr_d.get('funding_time')) or _to_ts_ms(fr_d.get('timestamp'))
                                if ts_ms is None:
                                    continue
                                if task.end_time and ts_ms > int(task.end_time):
                                    continue
                                rows.append({
                                    'timestamp': int(ts_ms),
                                    'datetime': _iso_utc_from_ms(ts_ms),
                                    'funding_rate': fr_d.get('funding_rate'),
                                    'predicted_rate': fr_d.get('predicted_rate'),
                                    'interval_hours': fr_d.get('interval_hours'),
                                })
                            rows.sort(key=lambda r: r.get('timestamp', 0))
                            task.results[symbol] = {'status': 'success', 'data': rows, 'count': len(rows)}
                        else:
                            fr = fetcher.fetch_funding_rate(symbol)
                            fr_d = _json_safe(fr, now_ms) if fr is not None else {}
                            ts_ms = _to_ts_ms(fr_d.get('timestamp')) or now_ms
                            payload = {
                                'timestamp': ts_ms,
                                'datetime': _iso_utc_from_ms(ts_ms),
                                'funding_rate': fr_d.get('funding_rate'),
                                'predicted_rate': fr_d.get('predicted_rate'),
                                'interval_hours': fr_d.get('interval_hours'),
                                'funding_time': fr_d.get('funding_time'),
                                'raw': fr_d,
                            }
                            task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'open_interest':
                        if not hasattr(fetcher, 'fetch_open_interest'):
                            raise AttributeError('fetch_open_interest not supported')

                        # swap 优先返回历史序列（用于可视化折线），否则回退为单点
                        if market_lower == 'swap' and hasattr(fetcher, 'fetch_open_interest_history') and task.start_time:
                            limit = int(task.limit or 200)
                            limit = max(1, min(limit, 1000))
                            hist = fetcher.fetch_open_interest_history(
                                symbol,
                                timeframe=task.timeframe,
                                since=task.start_time,
                                limit=limit,
                            )
                            hist_s = _json_safe(hist, now_ms) if hist is not None else []
                            rows = []
                            for oi_d in (hist_s or []):
                                if not isinstance(oi_d, dict):
                                    continue
                                ts_ms = _to_ts_ms(oi_d.get('timestamp'))
                                if ts_ms is None:
                                    continue
                                if task.end_time and ts_ms > int(task.end_time):
                                    continue
                                rows.append({
                                    'timestamp': int(ts_ms),
                                    'datetime': _iso_utc_from_ms(ts_ms),
                                    'open_interest': oi_d.get('open_interest'),
                                    'open_interest_value': oi_d.get('open_interest_value'),
                                    'volume_24h': oi_d.get('volume_24h'),
                                    'turnover_24h': oi_d.get('turnover_24h'),
                                })
                            rows.sort(key=lambda r: r.get('timestamp', 0))
                            task.results[symbol] = {'status': 'success', 'data': rows, 'count': len(rows)}
                        else:
                            oi = fetcher.fetch_open_interest(symbol)
                            oi_d = _json_safe(oi, now_ms) if oi is not None else {}
                            ts_ms = _to_ts_ms(oi_d.get('timestamp')) or now_ms
                            payload = {
                                'timestamp': ts_ms,
                                'datetime': _iso_utc_from_ms(ts_ms),
                                'open_interest': oi_d.get('open_interest'),
                                'open_interest_value': oi_d.get('open_interest_value'),
                                'volume_24h': oi_d.get('volume_24h'),
                                'turnover_24h': oi_d.get('turnover_24h'),
                                'raw': oi_d,
                            }
                            task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'mark_price':
                        if not hasattr(fetcher, 'fetch_mark_price'):
                            raise AttributeError('fetch_mark_price not supported')
                        mp = fetcher.fetch_mark_price(symbol)
                        mp_d = _json_safe(mp, now_ms) if mp is not None else {}
                        ts_ms = _to_ts_ms(mp_d.get('timestamp')) or now_ms
                        payload = {
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'mark_price': mp_d.get('mark_price'),
                            'index_price': mp_d.get('index_price'),
                            'funding_rate': mp_d.get('funding_rate'),
                            'raw': mp_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'contract_info':
                        ci = fetcher.fetch_contract_info(symbol) if hasattr(fetcher, 'fetch_contract_info') else {}
                        ci_d = _json_safe(ci, now_ms) if ci is not None else {}
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'contract_size': ci_d.get('contract_size'),
                            'settle_currency': ci_d.get('settle_currency') or ci_d.get('settle'),
                            'max_leverage': ci_d.get('max_leverage'),
                            'raw': ci_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'liquidations':
                        if not hasattr(fetcher, 'fetch_liquidation_info'):
                            raise AttributeError('fetch_liquidation_info not supported')
                        liqs = fetcher.fetch_liquidation_info(symbol, limit=task.limit or 50)
                        liqs_d = _json_safe(liqs, now_ms) if liqs is not None else []
                        last_liq = liqs_d[0] if isinstance(liqs_d, list) and liqs_d else {}
                        ts_ms = _to_ts_ms(last_liq.get('timestamp')) or now_ms
                        payload = {
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'side': last_liq.get('side'),
                            'quantity': last_liq.get('quantity'),
                            'price': last_liq.get('price'),
                            'value': last_liq.get('value'),
                            'liquidations': liqs_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'market_info':
                        if not hasattr(fetcher, 'fetch_market_info'):
                            raise AttributeError('fetch_market_info not supported')
                        mi = fetcher.fetch_market_info(symbol)
                        mi_d = _json_safe(mi, now_ms) if mi is not None else {}
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'raw': mi_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'basis':
                        if not hasattr(fetcher, 'fetch_basis'):
                            raise AttributeError('fetch_basis not supported')
                        b = fetcher.fetch_basis(symbol)
                        b_d = _json_safe(b, now_ms) if b is not None else {}
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'basis': b_d.get('basis') if isinstance(b_d, dict) else None,
                            'basis_percent': b_d.get('basis_percent') if isinstance(b_d, dict) else None,
                            'raw': b_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'term_structure':
                        if not hasattr(fetcher, 'fetch_term_structure'):
                            raise AttributeError('fetch_term_structure not supported')
                        ts = fetcher.fetch_term_structure(symbol)
                        ts_d = _json_safe(ts, now_ms) if ts is not None else {}
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'raw': ts_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'settlement_price':
                        if not hasattr(fetcher, 'fetch_settlement_price'):
                            raise AttributeError('fetch_settlement_price not supported')
                        sp = fetcher.fetch_settlement_price(symbol)
                        sp_d = _json_safe(sp, now_ms) if sp is not None else {}
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'settlement_price': sp_d.get('settlement_price') if isinstance(sp_d, dict) else None,
                            'raw': sp_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'snapshot':
                        # 优先走 fetcher 的聚合快照（若存在），否则按能力拼装
                        snap = {}
                        if hasattr(fetcher, 'fetch_market_snapshot'):
                            snap = fetcher.fetch_market_snapshot(
                                symbol=symbol,
                                timeframe=task.timeframe,
                                ohlcv_limit=min(int(task.limit or 200), 200),
                                trades_limit=min(int(task.limit or 200), 200),
                                orderbook_limit=min(int(task.limit or 50), 50),
                                include=include,
                            )
                        if not isinstance(snap, dict):
                            snap = {}

                        # 统一序列化
                        snap_s = _json_safe(snap, now_ms)
                        ticker = snap_s.get('ticker') if isinstance(snap_s.get('ticker'), dict) else {}
                        orderbook = snap_s.get('orderbook')
                        trades = snap.get('trades') if isinstance(snap.get('trades'), list) else []
                        trades_ser = [_serialize_trade(t, now_ms) for t in trades][:200]
                        ob_summary = _serialize_orderbook_summary(snap.get('orderbook'), now_ms) if orderbook is not None else {}

                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'last': ticker.get('last'),
                            'bid': ticker.get('bid'),
                            'ask': ticker.get('ask'),
                            'funding_rate': (snap_s.get('funding_rate') or {}).get('funding_rate') if isinstance(snap_s.get('funding_rate'), dict) else None,
                            'open_interest': (snap_s.get('open_interest') or {}).get('open_interest') if isinstance(snap_s.get('open_interest'), dict) else None,

                            'ticker': ticker,
                            'orderbook': ob_summary,
                            'trades': trades_ser,
                            'funding_rate_raw': snap_s.get('funding_rate'),
                            'open_interest_raw': snap_s.get('open_interest'),
                            'mark_price': snap_s.get('mark_price'),
                            'liquidations': snap_s.get('liquidations'),
                            'contract_info': snap_s.get('contract_info'),
                            'market_info': snap_s.get('market_info'),
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    else:
                        task.results[symbol] = {'status': 'error', 'error': f"Unsupported {market_lower} data_type: {data_type}"}

                except Exception as e:
                    logger.warning(f"Error fetching {symbol} ({market_lower} {data_type}): {e}")
                    task.results[symbol] = {'status': 'error', 'error': str(e)}

                task.progress = int((idx + 1) / total * 100)

            task.status = 'completed'
            task.progress = 100
            logger.info(f"Fetch task {task.task_id} completed successfully ({market_lower} {data_type})")
            return

        # option：支持按 data_type 批量获取（非 OHLCV）
        if market_lower == 'option' and data_type != 'ohlcv':
            total = len(task.symbols)
            for idx, symbol in enumerate(task.symbols):
                try:
                    now_ms = int(datetime.now().timestamp() * 1000)

                    if data_type == 'orderbook':
                        ob = fetcher.fetch_orderbook(symbol, limit=task.limit or 50)
                        payload = _serialize_orderbook_summary(ob, now_ms)
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'trades':
                        trades = fetcher.fetch_trades(symbol, limit=task.limit or 50)
                        trades_ser = [_serialize_trade(t, now_ms) for t in (trades or [])][:200]
                        last_trade = trades_ser[-1] if trades_ser else {}
                        payload = {
                            'timestamp': last_trade.get('timestamp') or now_ms,
                            'datetime': last_trade.get('datetime') or _iso_utc_from_ms(now_ms),
                            'price': last_trade.get('price'),
                            'amount': last_trade.get('amount'),
                            'side': last_trade.get('side'),
                            'trade_id': last_trade.get('trade_id') or last_trade.get('id'),
                            'trades': trades_ser,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'contract_info':
                        if not hasattr(fetcher, 'fetch_contract_info'):
                            raise AttributeError('fetch_contract_info not supported')
                        ci = fetcher.fetch_contract_info(symbol)
                        ci_d = _json_safe(ci or {}, now_ms)
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'raw': ci_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'option_price':
                        if not hasattr(fetcher, 'fetch_option_price'):
                            raise AttributeError('fetch_option_price not supported')
                        p = fetcher.fetch_option_price(symbol)
                        p_d = _json_safe(p or {}, now_ms)
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'mark_price': (p_d or {}).get('mark_price') if isinstance(p_d, dict) else None,
                            'bid': (p_d or {}).get('bid') if isinstance(p_d, dict) else None,
                            'ask': (p_d or {}).get('ask') if isinstance(p_d, dict) else None,
                            'theoretical_price': (p_d or {}).get('theoretical_price') if isinstance(p_d, dict) else None,
                            'iv': (p_d or {}).get('implied_volatility') if isinstance(p_d, dict) else None,
                            'raw': p_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'greeks':
                        if not hasattr(fetcher, 'fetch_greeks'):
                            raise AttributeError('fetch_greeks not supported')
                        g = fetcher.fetch_greeks(symbol)
                        g_d = _json_safe(g, now_ms) if g is not None else {}
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'delta': (g_d or {}).get('delta') if isinstance(g_d, dict) else None,
                            'gamma': (g_d or {}).get('gamma') if isinstance(g_d, dict) else None,
                            'theta': (g_d or {}).get('theta') if isinstance(g_d, dict) else None,
                            'vega': (g_d or {}).get('vega') if isinstance(g_d, dict) else None,
                            'rho': (g_d or {}).get('rho') if isinstance(g_d, dict) else None,
                            'iv': (g_d or {}).get('iv') if isinstance(g_d, dict) else None,
                            'raw': g_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'snapshot':
                        snap = {}
                        if hasattr(fetcher, 'fetch_market_snapshot'):
                            snap = fetcher.fetch_market_snapshot(
                                symbol=symbol,
                                trades_limit=min(int(task.limit or 200), 200),
                                orderbook_limit=min(int(task.limit or 50), 50),
                                include=include,
                            )
                        if not isinstance(snap, dict):
                            snap = {}

                        snap_s = _json_safe(snap, now_ms)
                        ci = snap_s.get('contract_info') if isinstance(snap_s.get('contract_info'), dict) else {}
                        g = snap_s.get('greeks') if isinstance(snap_s.get('greeks'), dict) else {}
                        p = snap_s.get('option_price') if isinstance(snap_s.get('option_price'), dict) else {}
                        ob_summary = _serialize_orderbook_summary(snap.get('orderbook'), now_ms) if snap.get('orderbook') is not None else {}
                        trades = snap.get('trades') if isinstance(snap.get('trades'), list) else []
                        trades_ser = [_serialize_trade(t, now_ms) for t in trades][:200]

                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'mark_price': p.get('mark_price') if isinstance(p, dict) else None,
                            'bid': p.get('bid') if isinstance(p, dict) else None,
                            'ask': p.get('ask') if isinstance(p, dict) else None,
                            'iv': (g.get('iv') if isinstance(g, dict) else None) or (p.get('implied_volatility') if isinstance(p, dict) else None),
                            'delta': g.get('delta') if isinstance(g, dict) else None,

                            'contract_info': ci,
                            'option_price': p,
                            'greeks': g,
                            'orderbook': ob_summary,
                            'trades': trades_ser,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    else:
                        task.results[symbol] = {'status': 'error', 'error': f"Unsupported option data_type: {data_type}"}

                except Exception as e:
                    logger.warning(f"Error fetching {symbol} (option {data_type}): {e}")
                    task.results[symbol] = {'status': 'error', 'error': str(e)}

                task.progress = int((idx + 1) / total * 100)

            task.status = 'completed'
            task.progress = 100
            logger.info(f"Fetch task {task.task_id} completed successfully (option {data_type})")
            return

        # margin：支持按 data_type 批量获取（非 OHLCV）
        if market_lower == 'margin' and data_type != 'ohlcv':
            total = len(task.symbols)
            for idx, symbol in enumerate(task.symbols):
                try:
                    now_ms = int(datetime.now().timestamp() * 1000)

                    if data_type == 'ticker':
                        t = fetcher.fetch_ticker(symbol)
                        ts_ms = _to_ts_ms((t or {}).get('timestamp')) or now_ms
                        payload = {
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'last': (t or {}).get('last'),
                            'bid': (t or {}).get('bid'),
                            'ask': (t or {}).get('ask'),
                            'high': (t or {}).get('high'),
                            'low': (t or {}).get('low'),
                            'volume': (t or {}).get('baseVolume') or (t or {}).get('quoteVolume'),
                            'raw': _json_safe(t or {}, now_ms),
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'orderbook':
                        ob = fetcher.fetch_orderbook(symbol, limit=task.limit or 50)
                        payload = _serialize_orderbook_summary(ob, now_ms)
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'trades':
                        trades = fetcher.fetch_trades(symbol, limit=task.limit or 50)
                        trades_ser = [_serialize_trade(t, now_ms) for t in (trades or [])][:200]
                        last_trade = trades_ser[-1] if trades_ser else {}
                        payload = {
                            'timestamp': last_trade.get('timestamp') or now_ms,
                            'datetime': last_trade.get('datetime') or _iso_utc_from_ms(now_ms),
                            'price': last_trade.get('price'),
                            'amount': last_trade.get('amount'),
                            'side': last_trade.get('side'),
                            'trade_id': last_trade.get('trade_id') or last_trade.get('id'),
                            'trades': trades_ser,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'market_info':
                        mi = fetcher.fetch_market_info(symbol)
                        mi_d = _json_safe(mi or {}, now_ms)
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'raw': mi_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'margin_levels':
                        if not hasattr(fetcher, 'fetch_margin_levels'):
                            raise AttributeError('fetch_margin_levels not supported')
                        ml = fetcher.fetch_margin_levels(symbol)
                        ml_d = _json_safe(ml or {}, now_ms)
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'max_leverage': (ml_d or {}).get('max_leverage') if isinstance(ml_d, dict) else None,
                            'maintenance_margin_rate': (ml_d or {}).get('maintenance_margin_rate') if isinstance(ml_d, dict) else None,
                            'raw': ml_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'borrow_rate':
                        if not hasattr(fetcher, 'fetch_borrow_rate'):
                            raise AttributeError('fetch_borrow_rate not supported')
                        # 尝试从 market_info 推断 base currency
                        mi = fetcher.fetch_market_info(symbol)
                        base = ((mi or {}).get('base') or '').upper() if isinstance(mi, dict) else ''
                        br = fetcher.fetch_borrow_rate(base) if base else {}
                        br_d = _json_safe(br or {}, now_ms)
                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'currency': (br_d or {}).get('currency') if isinstance(br_d, dict) else None,
                            'rate': (br_d or {}).get('rate') if isinstance(br_d, dict) else None,
                            'annual_rate': (br_d or {}).get('annual_rate') if isinstance(br_d, dict) else None,
                            'raw': br_d,
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'snapshot':
                        snap = {}
                        if hasattr(fetcher, 'fetch_market_snapshot'):
                            snap = fetcher.fetch_market_snapshot(
                                symbol=symbol,
                                timeframe=task.timeframe,
                                ohlcv_limit=min(int(task.limit or 200), 200),
                                trades_limit=min(int(task.limit or 200), 200),
                                orderbook_limit=min(int(task.limit or 50), 50),
                                include=include,
                            )
                        if not isinstance(snap, dict):
                            snap = {}

                        snap_s = _json_safe(snap, now_ms)
                        ticker = snap_s.get('ticker') if isinstance(snap_s.get('ticker'), dict) else {}
                        trades = snap.get('trades') if isinstance(snap.get('trades'), list) else []
                        trades_ser = [_serialize_trade(t, now_ms) for t in trades][:200]
                        ob_summary = _serialize_orderbook_summary(snap.get('orderbook'), now_ms) if snap.get('orderbook') is not None else {}

                        payload = {
                            'timestamp': now_ms,
                            'datetime': _iso_utc_from_ms(now_ms),
                            'last': ticker.get('last'),
                            'bid': ticker.get('bid'),
                            'ask': ticker.get('ask'),
                            'max_leverage': (snap_s.get('margin_levels') or {}).get('max_leverage') if isinstance(snap_s.get('margin_levels'), dict) else None,
                            'borrow_rate': (snap_s.get('borrow_rate') or {}).get('rate') if isinstance(snap_s.get('borrow_rate'), dict) else None,

                            'ticker': ticker,
                            'orderbook': ob_summary,
                            'trades': trades_ser,
                            'market_info': snap_s.get('market_info'),
                            'margin_levels': snap_s.get('margin_levels'),
                            'borrow_rate_raw': snap_s.get('borrow_rate'),
                            'liquidation_price': snap_s.get('liquidation_price'),
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    else:
                        task.results[symbol] = {'status': 'error', 'error': f"Unsupported margin data_type: {data_type}"}

                except Exception as e:
                    logger.warning(f"Error fetching {symbol} (margin {data_type}): {e}")
                    task.results[symbol] = {'status': 'error', 'error': str(e)}

                task.progress = int((idx + 1) / total * 100)

            task.status = 'completed'
            task.progress = 100
            logger.info(f"Fetch task {task.task_id} completed successfully (margin {data_type})")
            return

        # spot：支持按 data_type 拉取非 OHLCV 数据
        if market_lower == 'spot' and data_type != 'ohlcv':
            if hasattr(fetcher, 'add_symbols'):
                fetcher.add_symbols(task.symbols)

            total = len(task.symbols)
            for idx, symbol in enumerate(task.symbols):
                try:
                    now_ms = int(datetime.now().timestamp() * 1000)

                    if data_type == 'ticker':
                        t = fetcher.fetch_ticker(symbol)
                        ts_ms = _to_ts_ms(t.get('timestamp')) or now_ms
                        row = {
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'last': t.get('last'),
                            'bid': t.get('bid'),
                            'ask': t.get('ask'),
                            'high': t.get('high'),
                            'low': t.get('low'),
                            'baseVolume': t.get('baseVolume'),
                            'quoteVolume': t.get('quoteVolume'),
                            'percentage': t.get('percentage'),
                        }
                        task.results[symbol] = {'status': 'success', 'data': [row], 'count': 1}

                    elif data_type == 'orderbook':
                        depth = task.limit if isinstance(task.limit, int) and task.limit else 50
                        depth = max(5, min(int(depth), 500))
                        ob = fetcher.fetch_orderbook(symbol, limit=depth)
                        if ob is None:
                            task.results[symbol] = {'status': 'success', 'data': [], 'count': 0}
                        else:
                            ts_ms = _to_ts_ms(getattr(ob, 'timestamp', None)) or now_ms
                            # 使用统一序列化方法，确保包含 bids/asks
                            payload = _serialize_orderbook_summary(ob, now_ms)
                            
                            # 兼容旧字段
                            row = {
                                'timestamp': payload['timestamp'],
                                'datetime': payload['datetime'],
                                'best_bid': payload['best_bid'],
                                'best_ask': payload['best_ask'],
                                'spread': payload['spread'],
                                'spread_percent': payload['spread_percent'],
                                'bid_total': payload['bid_total'],
                                'ask_total': payload['ask_total'],
                                'imbalance': payload['imbalance'],
                                'bids': payload.get('bids', []),
                                'asks': payload.get('asks', [])
                            }
                            task.results[symbol] = {'status': 'success', 'data': [row], 'count': 1}

                    elif data_type == 'trades':
                        trade_limit = task.limit if isinstance(task.limit, int) and task.limit else 100
                        trade_limit = max(1, min(int(trade_limit), 1000))
                        trades = fetcher.fetch_trades(symbol, since=None, limit=trade_limit)
                        rows = []
                        for tr in trades or []:
                            if hasattr(tr, 'to_dict'):
                                d = tr.to_dict()
                            elif hasattr(tr, '__dict__'):
                                d = dict(tr.__dict__)
                            elif isinstance(tr, dict):
                                d = dict(tr)
                            else:
                                d = {'value': str(tr)}

                            ts_ms = _to_ts_ms(d.get('timestamp')) or now_ms
                            d['timestamp'] = ts_ms
                            d['datetime'] = _iso_utc_from_ms(ts_ms)
                            rows.append(d)
                        # 统一为 payload 结构
                        if rows:
                            last_ts = rows[-1]['timestamp']
                            last_dt = rows[-1]['datetime']
                        else:
                            last_ts = now_ms
                            last_dt = _iso_utc_from_ms(last_ts)
                        payload = {
                            'timestamp': last_ts,
                            'datetime': last_dt,
                            'count': len(rows),
                            'trades': rows
                        }
                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    elif data_type == 'market_info':
                        info = fetcher.fetch_market_info(symbol)
                        ts_ms = now_ms
                        row = {'timestamp': ts_ms, 'datetime': _iso_utc_from_ms(ts_ms)}
                        if isinstance(info, dict):
                            row.update(info)
                        task.results[symbol] = {'status': 'success', 'data': [row], 'count': 1}

                    elif data_type == 'snapshot':
                        # 更“全”的结构：返回一份嵌套对象（同时带少量摘要字段方便表格展示）
                        snap = None
                        if hasattr(fetcher, 'fetch_market_snapshot') and callable(getattr(fetcher, 'fetch_market_snapshot')):
                            # 默认 include：不传则由 fetcher 给默认集合；若用户传 include 则尊重
                            ohlcv_limit = 50
                            trades_limit = 50
                            orderbook_limit = 50
                            if isinstance(task.limit, int) and task.limit and task.limit > 0:
                                # 对 snapshot：limit 用于控制 trades 数量（同时也可能用于 ohlcv 深度），上限保护
                                trades_limit = max(1, min(int(task.limit), 1000))
                                ohlcv_limit = max(1, min(int(task.limit), 500))
                                orderbook_limit = max(5, min(int(task.limit), 500))

                            snap = fetcher.fetch_market_snapshot(
                                symbol=symbol,
                                timeframe=task.timeframe,
                                ohlcv_limit=ohlcv_limit,
                                trades_limit=trades_limit,
                                orderbook_limit=orderbook_limit,
                                include=include,
                            )

                        if not isinstance(snap, dict):
                            snap = {}

                        ticker = snap.get('ticker') if isinstance(snap.get('ticker'), dict) else {}
                        stats_24h = snap.get('stats_24h') if isinstance(snap.get('stats_24h'), dict) else {}
                        orderbook_obj = snap.get('orderbook')
                        trades_list = snap.get('trades') if isinstance(snap.get('trades'), list) else []
                        market_info = snap.get('market_info') if isinstance(snap.get('market_info'), dict) else {}

                        ob_summary = _serialize_orderbook_summary(orderbook_obj, now_ms)
                        trades_ser = [_serialize_trade(t, now_ms) for t in trades_list][:50]

                        # 可选：ohlcv（若 include 里包含 ohlcv）
                        ohlcv_ser = []
                        if 'ohlcv' in (include or []) and isinstance(snap.get('ohlcv'), list):
                            for item in snap.get('ohlcv'):
                                try:
                                    if hasattr(item, 'timestamp') and hasattr(item, 'open'):
                                        ts_ms = _to_ts_ms(getattr(item, 'timestamp', None))
                                        if ts_ms is None:
                                            continue
                                        ohlcv_ser.append({
                                            'timestamp': ts_ms,
                                            'datetime': _iso_utc_from_ms(ts_ms),
                                            'open': float(getattr(item, 'open', 0)),
                                            'high': float(getattr(item, 'high', 0)),
                                            'low': float(getattr(item, 'low', 0)),
                                            'close': float(getattr(item, 'close', 0)),
                                            'volume': float(getattr(item, 'volume', 0)),
                                        })
                                    elif isinstance(item, (list, tuple)) and len(item) >= 6:
                                        ts_ms = _to_ts_ms(item[0])
                                        if ts_ms is None:
                                            continue
                                        ohlcv_ser.append({
                                            'timestamp': ts_ms,
                                            'datetime': _iso_utc_from_ms(ts_ms),
                                            'open': float(item[1]),
                                            'high': float(item[2]),
                                            'low': float(item[3]),
                                            'close': float(item[4]),
                                            'volume': float(item[5]),
                                        })
                                except Exception:
                                    continue

                        ts_ms = now_ms
                        payload = {
                            # 表格友好字段（摘要）
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'last': ticker.get('last'),
                            'bid': ticker.get('bid'),
                            'ask': ticker.get('ask'),
                            'spread': ob_summary.get('spread'),
                            '24h_change_pct': stats_24h.get('price_change_percent'),

                            # 更完整的嵌套结构
                            'ticker': ticker,
                            'stats_24h': stats_24h,
                            'orderbook': ob_summary,
                            'trades': trades_ser,
                            'market_info': market_info,
                        }
                        if ohlcv_ser:
                            payload['ohlcv'] = ohlcv_ser

                        task.results[symbol] = {'status': 'success', 'data': [payload], 'count': 1}

                    else:
                        task.results[symbol] = {'status': 'error', 'error': f"Unsupported spot data_type: {data_type}"}

                except Exception as e:
                    logger.warning(f"Error fetching {symbol} ({data_type}): {e}")
                    task.results[symbol] = {'status': 'error', 'error': str(e)}

                task.progress = int((idx + 1) / total * 100)

            task.status = 'completed'
            task.progress = 100
            logger.info(f"Fetch task {task.task_id} completed successfully (spot {data_type})")
            return

        # 兜底：若请求的是非 OHLCV data_type，但前面没有对应市场分支处理，
        # 则返回明确错误，避免误走通用 OHLCV 流程覆盖结果。
        if data_type != 'ohlcv':
            total = len(task.symbols)
            for idx, symbol in enumerate(task.symbols):
                task.results[symbol] = {
                    'status': 'error',
                    'error': f"Unsupported {market_lower} data_type: {data_type}"
                }
                task.progress = int((idx + 1) / total * 100)

            task.status = 'completed'
            task.progress = 100
            logger.info(f"Fetch task {task.task_id} completed with unsupported data_type ({market_lower} {data_type})")
            return
        
        if hasattr(fetcher, 'add_symbols'):
            fetcher.add_symbols(task.symbols)
        
        total = len(task.symbols)
        task.progress = 0
        for idx, symbol in enumerate(task.symbols):
            try:
                if hasattr(fetcher, 'fetch_ohlcv'):
                    since_ms = task.start_time
                    end_ms = task.end_time
                    
                    used_bulk = False
                    if since_ms and end_ms and hasattr(fetcher, 'fetch_ohlcv_bulk'):
                        start_dt = datetime.utcfromtimestamp(since_ms/1000)
                        end_dt = datetime.utcfromtimestamp(end_ms/1000)
                        data_list = fetcher.fetch_ohlcv_bulk(
                            symbol=symbol,
                            start_date=start_dt,
                            end_date=end_dt,
                            timeframe=task.timeframe,
                            max_bars_per_request=1000
                        )
                        used_bulk = True
                    else:
                        data_list = fetcher.fetch_ohlcv(symbol, task.timeframe, since=since_ms, limit=task.limit)

                    # 立即处理并写入结果，避免缓存巨量 raw 数据
                    ohlcv = []

                    # 处理 DataFrame 返回
                    if hasattr(data_list, 'to_dict') and hasattr(data_list, 'columns'):
                        try:
                            df_copy = data_list.reset_index()
                            records = df_copy.to_dict(orient='records')
                            for rec in records:
                                ts_val = rec.get('timestamp') or rec.get('index')
                                if ts_val is None:
                                    continue

                                if hasattr(ts_val, 'timestamp'):
                                    ts_ms = int(ts_val.timestamp() * 1000)
                                elif isinstance(ts_val, (int, float)):
                                    ts_ms = int(ts_val) if ts_val > 1e12 else int(ts_val * 1000)
                                else:
                                    ts_ms = int(pd.Timestamp(ts_val).timestamp() * 1000)

                                ohlcv.append({
                                    'timestamp': ts_ms,
                                    'datetime': _iso_utc_from_ms(ts_ms),
                                    'open': float(rec.get('open', 0)),
                                    'high': float(rec.get('high', 0)),
                                    'low': float(rec.get('low', 0)),
                                    'close': float(rec.get('close', 0)),
                                    'volume': float(rec.get('volume', 0))
                                })
                        except Exception as df_err:
                            logger.error(f"Error processing DataFrame for {symbol}: {df_err}", exc_info=True)
                    else:
                        for item in data_list:
                            try:
                                if hasattr(item, 'timestamp'):
                                    ts = item.timestamp
                                    ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, 'timestamp') else int(ts)
                                    if not used_bulk:
                                        if since_ms and ts_ms < since_ms:
                                            continue
                                        if end_ms and ts_ms > end_ms:
                                            continue
                                    row = {
                                        'timestamp': ts_ms,
                                        'datetime': _iso_utc_from_ms(ts_ms),
                                        'open': float(item.open),
                                        'high': float(item.high),
                                        'low': float(item.low),
                                        'close': float(item.close),
                                        'volume': float(item.volume)
                                    }

                                    # 扩展字段（若存在）
                                    for k in (
                                        'quote_volume', 'trades',
                                        'taker_buy_base_volume', 'taker_buy_quote_volume',
                                        'vwap',
                                        'rsi',
                                        'sma_20', 'sma_50', 'sma_200',
                                        'ema_12', 'ema_26',
                                        'macd', 'macd_signal', 'macd_histogram',
                                        'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                                    ):
                                        if hasattr(item, k):
                                            try:
                                                v = getattr(item, k)
                                                if v is None or (hasattr(pd, 'isna') and pd.isna(v)):
                                                    continue
                                                row[k] = float(v) if k != 'trades' else int(v)
                                            except Exception:
                                                continue

                                    ohlcv.append(row)
                                else:
                                    ts_ms = item[0]
                                    if not used_bulk:
                                        if since_ms and ts_ms < since_ms:
                                            continue
                                        if end_ms and ts_ms > end_ms:
                                            continue
                                    ohlcv.append({
                                        'timestamp': ts_ms,
                                        'datetime': _iso_utc_from_ms(ts_ms),
                                        'open': float(item[1]),
                                        'high': float(item[2]),
                                        'low': float(item[3]),
                                        'close': float(item[4]),
                                        'volume': float(item[5])
                                    })
                            except Exception as item_error:
                                logger.warning(f"Error processing OHLCV item for {symbol}: {item_error}")
                                continue

                    task.results[symbol] = {
                        'status': 'success',
                        'data': ohlcv,
                        'count': len(ohlcv)
                    }

                    if getattr(task, 'auto_save', False) and str(getattr(task, 'storage_format', 'parquet')).lower() == 'parquet':
                        _save_ohlcv_parquet(task.market, task.exchange, symbol, task.timeframe, ohlcv)
                else:
                    task.results[symbol] = {
                        'status': 'skipped',
                        'error': 'Fetcher does not support fetch_ohlcv'
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                task.results[symbol] = {
                    'status': 'error',
                    'error': str(e)
                }

            # 更新进度 (0-100%)
            task.progress = int((idx + 1) / total * 100)
        
        # 确保任务完成状态设置
        task.status = 'completed'
        task.progress = 100
        logger.info(f"Fetch task {task.task_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Fetch task {task.task_id} failed: {e}", exc_info=True)
        task.status = 'error'
        task.error = str(e)
        task.progress = 0


# ==================== 常量配置 ====================

SUPPORTED_EXCHANGES = [
    'binance', 'okx', 'bybit', 'kucoin', 'gate',
    'huobi', 'upbit', 'bithumb', 'kraken', 'coinbase'
]

SUPPORTED_MARKETS = [
    'spot', 'swap', 'future', 'option', 'margin', 'onchain', 'social'
]

# 预定义的交易对列表（作为备选方案）
PREDEFINED_SYMBOLS = {
    'binance': {
        'spot': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
            'ADA/USDT', 'DOGE/USDT', 'POLKADOT/USDT', 'DOT/USDT', 'LINK/USDT',
            'LTC/USDT', 'AVAX/USDT', 'UNI/USDT', 'MATIC/USDT', 'FIL/USDT',
            'TRX/USDT', 'NEAR/USDT', 'ICP/USDT', 'OP/USDT', 'ATOM/USDT',
            'ARB/USDT', 'BLUR/USDT', 'GMX/USDT', 'SEI/USDT', 'MANTA/USDT',
            'ZETA/USDT', 'AAVE/USDT', 'APT/USDT', 'SUI/USDT', 'FLOW/USDT',
            'PEPE/USDT', 'SHIB/USDT', 'WIF/USDT', 'BONK/USDT', 'FLOKI/USDT',
            'SAND/USDT', 'MANA/USDT', 'GALA/USDT', 'ENJ/USDT', 'AXIE/USDT',
        ],
        'future': [
            'BTC/USD:BTC-260327', 'ETH/USD:ETH-260327', 'XRP/USD:XRP-260327', 'BNB/USD:BNB-260327', 'SOL/USD:SOL-260327',
            'BTC/USD:BTC-260626', 'ETH/USD:ETH-260626', 'XRP/USD:XRP-260626', 'BNB/USD:BNB-260626', 'SOL/USD:SOL-260626',
        ],
        'option': [
            'BTC-240628-50000-C', 'BTC-240628-50000-P',
            'ETH-240628-3000-C', 'ETH-240628-3000-P',
        ],
        'swap': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
            'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'LINK/USDT', 'LTC/USDT',
        ],
        'margin': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
        ],
    },
    'okx': {
        'spot': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
            'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'LINK/USDT', 'LTC/USDT',
            'AVAX/USDT', 'UNI/USDT', 'MATIC/USDT', 'FIL/USDT', 'TRX/USDT',
        ],
        'future': [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT',
        ],
        'swap': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
        ],
        'option': [
            'BTC/USDT:USDT', 'ETH/USDT:USDT',
        ],
    },
    'bybit': {
        'spot': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
        ],
        'future': [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
        ],
        'swap': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT',
        ],
    },
}

# 缓存全局对象
_fetchers_cache = {}
_managers_cache = {}


# ==================== 辅助函数 ====================

def get_or_create_fetcher(exchange: str, market: str):
    """获取或创建 Fetcher 实例"""
    key = f"{exchange}:{market}"
    if key not in _fetchers_cache:
        try:
            _fetchers_cache[key] = create_fetcher(exchange, market)
            logger.info(f"Created fetcher: {key}")
        except Exception as e:
            logger.error(f"Failed to create fetcher {key}: {e}")
            raise
    return _fetchers_cache[key]


def get_or_create_manager(market: str, exchange: str = 'binance', storage_format: str = 'parquet'):
    """获取或创建 DataManager 实例"""
    fmt = str(storage_format or 'parquet').lower()
    key = f"{market}:{exchange}:{fmt}"
    if key not in _managers_cache:
        try:
            save_json_merged = True if fmt == 'json' else False
            _managers_cache[key] = create_data_manager(market, exchange=exchange, save_json_merged=save_json_merged)
            logger.info(f"Created manager: {key}")
        except Exception as e:
            logger.error(f"Failed to create manager {key}: {e}")
            raise
    return _managers_cache[key]


def _to_ts_ms(ts_val) -> Optional[int]:
    if ts_val is None:
        return None
    if hasattr(ts_val, 'timestamp') and callable(getattr(ts_val, 'timestamp')):
        try:
            return int(ts_val.timestamp() * 1000)
        except Exception:
            pass
    if isinstance(ts_val, (int, float)):
        return int(ts_val) if ts_val > 1e12 else int(ts_val * 1000)
    try:
        return int(pd.Timestamp(ts_val).timestamp() * 1000)
    except Exception:
        return None


def _infer_social_period(start_ms: Optional[int], end_ms: Optional[int], timeframe: str) -> str:
    """把 UI 的时间范围/周期粗略映射到 social_fetcher 的 period (24h/7d/30d)。"""
    try:
        if start_ms and end_ms and isinstance(start_ms, (int, float)) and isinstance(end_ms, (int, float)):
            delta_ms = max(0, int(end_ms) - int(start_ms))
            delta_days = delta_ms / (1000 * 60 * 60 * 24)
            if delta_days <= 2:
                return '24h'
            if delta_days <= 14:
                return '7d'
            return '30d'
    except Exception:
        pass

    tf = str(timeframe or '').lower()
    if tf in ('1h', '4h', '1d'):
        return '24h'
    if tf in ('1w',):
        return '7d'
    return '30d'


def _normalize_snapshot_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    ts_ms = _to_ts_ms(raw.get('timestamp'))
    if ts_ms is None:
        ts_ms = int(datetime.now().timestamp() * 1000)

    out: Dict[str, Any] = {
        'timestamp': ts_ms,
        'datetime': _iso_utc_from_ms(ts_ms),
    }

    for k, v in raw.items():
        if k == 'timestamp':
            continue
        if isinstance(v, (dict, list, tuple, set)):
            continue
        if hasattr(v, 'isoformat') and callable(getattr(v, 'isoformat')):
            try:
                out[k] = v.isoformat()
                continue
            except Exception:
                pass
        out[k] = v

    return out


def _fetch_onchain_snapshot(fetcher, symbol: str) -> Dict[str, Any]:
    sym = str(symbol).lower()
    if sym in ('network_stats', 'network', 'stats'):
        raw = fetcher.fetch_network_stats()
        return _normalize_snapshot_dict(raw)
    if sym in ('gas_price', 'gas'):
        raw = fetcher.fetch_gas_price()
        return _normalize_snapshot_dict(raw)
    if sym in ('block_number', 'blocknumber', 'height'):
        if not hasattr(fetcher, 'fetch_block_number'):
            return _normalize_snapshot_dict({'error': 'fetch_block_number not supported', 'requested': symbol})
        raw = {'block_number': fetcher.fetch_block_number()}
        return _normalize_snapshot_dict(raw)
    if sym in ('block_latest', 'latest_block', 'block:latest'):
        if not hasattr(fetcher, 'fetch_block'):
            return _normalize_snapshot_dict({'error': 'fetch_block not supported', 'requested': symbol})
        raw = fetcher.fetch_block('latest')
        return _normalize_snapshot_dict(raw)
    if sym.startswith('block:'):
        try:
            if not hasattr(fetcher, 'fetch_block'):
                return _normalize_snapshot_dict({'error': 'fetch_block not supported', 'requested': symbol})
            block_id = sym.split(':', 1)[1].strip()
            block_id = int(block_id) if block_id.isdigit() else block_id
            raw = fetcher.fetch_block(block_id)
            out = _normalize_snapshot_dict(raw)
            out['requested'] = symbol
            return out
        except Exception:
            pass
    if sym.startswith('tx:'):
        try:
            if not hasattr(fetcher, 'fetch_transaction'):
                return _normalize_snapshot_dict({'error': 'fetch_transaction not supported', 'requested': symbol})
            tx_hash = sym.split(':', 1)[1].strip()
            raw = fetcher.fetch_transaction(tx_hash)
            out = _normalize_snapshot_dict(raw)
            out['requested'] = symbol
            return out
        except Exception:
            pass
    if sym.startswith('balance:'):
        # balance:<address> 或 balance:<address>:<token_address>
        try:
            if not hasattr(fetcher, 'fetch_address_balance'):
                return _normalize_snapshot_dict({'error': 'fetch_address_balance not supported', 'requested': symbol})
            parts = sym.split(':')
            address = parts[1].strip() if len(parts) > 1 else ''
            token_address = parts[2].strip() if len(parts) > 2 else None
            raw = fetcher.fetch_address_balance(address=address, token_address=token_address)
            out = _normalize_snapshot_dict(raw)
            out['requested'] = symbol
            return out
        except Exception:
            pass
    if sym.startswith('contract:'):
        try:
            if not hasattr(fetcher, 'fetch_contract_info'):
                return _normalize_snapshot_dict({'error': 'fetch_contract_info not supported', 'requested': symbol})
            contract_address = sym.split(':', 1)[1].strip()
            raw = fetcher.fetch_contract_info(contract_address)
            out = _normalize_snapshot_dict(raw)
            out['requested'] = symbol
            return out
        except Exception:
            pass
    if sym in ('eth2_staking', 'eth2', 'staking', 'eth2_staking_stats'):
        if not hasattr(fetcher, 'fetch_eth2_staking_stats'):
            return _normalize_snapshot_dict({'error': 'fetch_eth2_staking_stats not supported', 'requested': symbol})
        raw = fetcher.fetch_eth2_staking_stats()
        return _normalize_snapshot_dict(raw)
    if sym in ('defi_stats', 'defi', 'tvl'):
        if not hasattr(fetcher, 'fetch_defi_stats'):
            return _normalize_snapshot_dict({'error': 'fetch_defi_stats not supported', 'requested': symbol})
        raw = fetcher.fetch_defi_stats()
        return _normalize_snapshot_dict(raw)
    if sym in ('nft_stats', 'nft'):
        if not hasattr(fetcher, 'fetch_nft_stats'):
            return _normalize_snapshot_dict({'error': 'fetch_nft_stats not supported', 'requested': symbol})
        raw = fetcher.fetch_nft_stats()
        return _normalize_snapshot_dict(raw)
    if sym in ('multi_network_stats', 'multi_network'):
        if not hasattr(fetcher, 'fetch_multi_network_stats'):
            return _normalize_snapshot_dict({'error': 'fetch_multi_network_stats not supported', 'requested': symbol})
        raw = fetcher.fetch_multi_network_stats()
        return _normalize_snapshot_dict(raw)
    # 默认：尝试按网络统计返回（避免空）
    raw = fetcher.fetch_network_stats()
    snap = _normalize_snapshot_dict(raw)
    snap['requested'] = symbol
    return snap


def _fetch_social_snapshot(fetcher, symbol: str, period: str) -> Dict[str, Any]:
    metric = fetcher.fetch_metrics(symbol=symbol, period=period)
    raw = metric.to_dict() if hasattr(metric, 'to_dict') else (metric.__dict__ if hasattr(metric, '__dict__') else {})
    # social 的 timestamp 通常是 ISO 字符串
    ts_ms = _to_ts_ms(raw.get('timestamp'))
    if ts_ms is None:
        ts_ms = int(datetime.now().timestamp() * 1000)

    out: Dict[str, Any] = {
        'timestamp': ts_ms,
        'datetime': _iso_utc_from_ms(ts_ms),
        'platform': raw.get('platform', getattr(fetcher, 'platform', None)),
        'period': raw.get('period', period),
        'sentiment_score': raw.get('sentiment_score', 0.0),
        'post_count': raw.get('post_count', 0),
        'engagement_rate': raw.get('engagement_rate', 0.0),
        'mention_count': raw.get('mention_count', raw.get('post_count', 0)),
        'unique_users': raw.get('unique_users', 0),
        'total_likes': raw.get('total_likes', 0),
        'total_comments': raw.get('total_comments', 0),
        'total_shares': raw.get('total_shares', 0),
    }
    return out


# ==================== API 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index_new.html')


@app.route('/api/config')
def api_config():
    """获取系统配置"""
    return jsonify({
        'exchanges': SUPPORTED_EXCHANGES,
        'markets': SUPPORTED_MARKETS,
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/onchain/analyze', methods=['POST'])
def api_onchain_analyze():
    """链上看板分析统一入口"""
    try:
        payload = request.json or {}
        kind = str(payload.get('kind') or '').strip().lower()
        if not kind:
            return jsonify({'success': False, 'error': 'kind is required'}), 400

        network = str(payload.get('network') or 'ethereum').lower()
        chain = str(payload.get('chain') or 'mainnet').lower()
        use_simulation = bool(payload.get('simulation') or False)

        dune_query = payload.get('dune_query')
        dune_query_id = int(dune_query) if dune_query else None
        dune_params = _parse_json_maybe(payload.get('dune_params'))

        if kind == 'exchange_flow':
            analyzer = ExchangeFlowAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            exchange = str(payload.get('exchange') or 'binance').lower()
            hours = int(payload.get('hours') or 24)
            stablecoin = bool(payload.get('stablecoin') or False)
            if stablecoin:
                result = analyzer.fetch_stablecoin_flow(
                    exchange=exchange,
                    hours=hours,
                    dune_query_id=dune_query_id,
                    dune_params=dune_params,
                )
            else:
                result = analyzer.fetch_exchange_flow(
                    exchange=exchange,
                    hours=hours,
                    dune_query_id=dune_query_id,
                    dune_params=dune_params,
                )
        elif kind == 'address_behavior':
            analyzer = AddressBehaviorAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            hours = int(payload.get('hours') or 24)
            retention_hours = int(payload.get('retention_hours') or 24)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                addresses = _parse_addresses_maybe(payload.get('addresses'))
                result = analyzer.analyze_watchlist(
                    addresses,
                    hours=hours,
                    retention_hours=retention_hours,
                )
        elif kind == 'large_moves':
            analyzer = LargeMoveAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            hours = int(payload.get('hours') or 24)
            min_value = float(payload.get('min_value') or 100.0)
            token_address = str(payload.get('token') or '').strip() or None
            contract_calls = bool(payload.get('contract_calls') or False)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                addresses = _parse_addresses_maybe(payload.get('addresses'))
                if contract_calls:
                    result = analyzer.analyze_contract_call_concentration(
                        addresses,
                        hours=hours,
                        min_value=min_value,
                    )
                else:
                    result = analyzer.analyze_large_transfers(
                        addresses,
                        hours=hours,
                        min_value=min_value,
                        token_address=token_address,
                    )
        elif kind == 'mev':
            analyzer = MEVAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                result = analyzer.analyze_placeholder()
        elif kind == 'gas':
            analyzer = GasAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                hours = int(payload.get('hours') or 1)
                max_blocks = int(payload.get('max_blocks') or 200)
                result = analyzer.analyze_blocks(hours=hours, max_blocks=max_blocks)
        elif kind == 'protocol':
            analyzer = ProtocolAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                subgraph = str(payload.get('subgraph') or '').strip()
                query = str(payload.get('query') or '').strip()
                variables = _parse_json_maybe(payload.get('variables'))
                if not subgraph or not query:
                    return jsonify({'success': False, 'error': 'subgraph and query are required when dune_query is empty'}), 400
                result = analyzer.analyze_the_graph(subgraph, query, variables)
        elif kind == 'capital_cycle':
            analyzer = CapitalCycleAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                hours = int(payload.get('hours') or 24)
                addresses = _parse_addresses_maybe(payload.get('addresses'))
                result = analyzer.analyze_watchlist(addresses, hours=hours)
        elif kind == 'token_distribution':
            analyzer = TokenDistributionAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                token_address = str(payload.get('token') or '').strip()
                if not token_address:
                    return jsonify({'success': False, 'error': 'token is required'}), 400
                addresses = _parse_addresses_maybe(payload.get('addresses'))
                whale_threshold = float(payload.get('whale_threshold') or 100000.0)
                top_n = int(payload.get('top_n') or 10)
                result = analyzer.analyze_watchlist(
                    token_address=token_address,
                    addresses=addresses,
                    whale_threshold=whale_threshold,
                    top_n=top_n,
                )
        elif kind == 'nft':
            analyzer = NFTAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            if dune_query_id:
                result = analyzer.analyze_dune(dune_query_id, dune_params)
            else:
                subgraph = str(payload.get('subgraph') or '').strip()
                query = str(payload.get('query') or '').strip()
                variables = _parse_json_maybe(payload.get('variables'))
                if not subgraph or not query:
                    return jsonify({'success': False, 'error': 'subgraph and query are required when dune_query is empty'}), 400
                result = analyzer.analyze_the_graph(subgraph, query, variables)
        elif kind == 'price_relation':
            analyzer = PriceRelationAnalyzer(network=network, chain=chain, use_simulation=use_simulation)
            if not dune_query_id:
                return jsonify({'success': False, 'error': 'dune_query is required'}), 400
            metric_key = str(payload.get('metric_key') or '').strip()
            if not metric_key:
                return jsonify({'success': False, 'error': 'metric_key is required'}), 400
            exchange = str(payload.get('exchange') or 'binance')
            symbol = str(payload.get('symbol') or 'BTC/USDT')
            timeframe = str(payload.get('timeframe') or '1h')
            limit = int(payload.get('limit') or 500)
            result = analyzer.analyze_dune_vs_price(
                dune_query_id=dune_query_id,
                metric_key=metric_key,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                parameters=dune_params,
            )
        else:
            return jsonify({'success': False, 'error': f'unknown kind: {kind}'}), 400

        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"onchain analyze failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fetcher/info', methods=['POST'])
def api_fetcher_info():
    """获取 Fetcher 信息"""
    try:
        data = request.json
        exchange = data.get('exchange', 'binance')
        market = data.get('market', 'spot')
        
        fetcher = get_or_create_fetcher(exchange, market)
        
        return jsonify({
            'success': True,
            'exchange': exchange,
            'market': market,
            'type': fetcher.__class__.__name__,
            'symbols_count': len(fetcher.symbols) if hasattr(fetcher, 'symbols') else 0,
            'message': f'Fetcher 信息获取成功'
        })
    except Exception as e:
        logger.error(f"Error in fetcher_info: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/api/fetcher/symbols', methods=['POST'])
def api_fetcher_symbols():
    """获取交易对列表"""
    try:
        data = request.json
        exchange = data.get('exchange', 'binance')
        market = data.get('market', 'spot')
        
        fetcher = get_or_create_fetcher(exchange, market)
        
        # 获取符号列表
        symbols = []
        if hasattr(fetcher, 'fetch_symbols'):
            symbols = fetcher.fetch_symbols()
        elif hasattr(fetcher, 'symbols'):
            symbols = list(fetcher.symbols)
        
        return jsonify({
            'success': True,
            'symbols': symbols[:100],  # 返回前100个
            'total': len(symbols),
            'message': f'获得 {len(symbols)} 个交易对'
        })
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/fetcher/tickers', methods=['POST'])
def api_fetcher_tickers():
    """获取行情数据"""
    try:
        data = request.json
        exchange = data.get('exchange', 'binance')
        market = data.get('market', 'spot')
        symbols = data.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        fetcher = get_or_create_fetcher(exchange, market)
        
        # 添加交易对
        if hasattr(fetcher, 'add_symbols'):
            fetcher.add_symbols(symbols)
        
        # 获取行情
        tickers = {}
        if hasattr(fetcher, 'fetch_all_tickers'):
            tickers_data = fetcher.fetch_all_tickers()
            tickers = {
                k: {
                    'symbol': k,
                    'bid': v.get('bid', 0),
                    'ask': v.get('ask', 0),
                    'last': v.get('last', 0),
                    'high': v.get('high', 0),
                    'low': v.get('low', 0),
                    'volume': v.get('volume', 0),
                    'timestamp': datetime.now().isoformat()
                } for k, v in tickers_data.items()
            }
        elif hasattr(fetcher, 'fetch_ticker'):
            tickers = {}
            for symbol in symbols:
                try:
                    ticker = fetcher.fetch_ticker(symbol)
                    tickers[symbol] = {
                        'symbol': symbol,
                        'bid': ticker.get('bid', 0),
                        'ask': ticker.get('ask', 0),
                        'last': ticker.get('last', 0),
                        'high': ticker.get('high', 0),
                        'low': ticker.get('low', 0),
                        'volume': ticker.get('volume', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
        
        return jsonify({
            'success': True,
            'exchange': exchange,
            'market': market,
            'data': tickers,
            'count': len(tickers),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/api/fetcher/ohlcv', methods=['POST'])
def api_fetcher_ohlcv():
    """获取 OHLCV 数据"""
    try:
        data = request.json
        exchange = data.get('exchange', 'binance')
        market = data.get('market', 'spot')
        symbol = data.get('symbol', 'BTC/USDT')
        timeframe = data.get('timeframe', '1h')
        limit = data.get('limit')  # None 表示不限制
        since = data.get('since')  # 可选：毫秒时间戳 / ISO 字符串 / datetime

        # onchain/social 不支持 OHLCV
        if str(market).lower() in ('onchain', 'social'):
            return jsonify({
                'success': False,
                'error': f"Market '{market}' does not support OHLCV",
                'hint': '请选择 spot/swap/future/option/margin 等市场，或使用对应的链上/舆情接口'
            }), 400

        # 规范化 limit
        if limit is not None:
            try:
                limit = int(limit)
                if limit <= 0:
                    limit = None
            except Exception:
                limit = None

        def _to_ts_ms(ts_val):
            """尽可能把时间转换为毫秒时间戳"""
            if ts_val is None:
                return None

            # datetime-like
            if hasattr(ts_val, 'timestamp') and callable(getattr(ts_val, 'timestamp')):
                return int(ts_val.timestamp() * 1000)

            # numeric
            if isinstance(ts_val, (int, float)):
                return int(ts_val) if ts_val > 1e12 else int(ts_val * 1000)

            # string / others
            try:
                return int(pd.Timestamp(ts_val).timestamp() * 1000)
            except Exception:
                return None
        
        fetcher = get_or_create_fetcher(exchange, market)
        
        # 添加交易对
        if hasattr(fetcher, 'add_symbols'):
            fetcher.add_symbols([symbol])
        
        # 获取 OHLCV
        ohlcv = []
        if hasattr(fetcher, 'fetch_ohlcv'):
            try:
                data_list = fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )

                for item in (data_list or []):
                    # 1) 数据模型对象（OHLCVData/SwapOHLCVData 等）
                    if hasattr(item, 'timestamp') and hasattr(item, 'open'):
                        ts_ms = _to_ts_ms(getattr(item, 'timestamp', None))
                        if ts_ms is None:
                            continue
                        row = {
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'open': float(getattr(item, 'open', 0)),
                            'high': float(getattr(item, 'high', 0)),
                            'low': float(getattr(item, 'low', 0)),
                            'close': float(getattr(item, 'close', 0)),
                            'volume': float(getattr(item, 'volume', 0)),
                        }

                        # 扩展字段（若存在）
                        for k in (
                            'quote_volume', 'trades',
                            'taker_buy_base_volume', 'taker_buy_quote_volume',
                            'vwap',
                            'rsi',
                            'sma_20', 'sma_50', 'sma_200',
                            'ema_12', 'ema_26',
                            'macd', 'macd_signal', 'macd_histogram',
                            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                        ):
                            if hasattr(item, k):
                                try:
                                    v = getattr(item, k)
                                    if v is None or (hasattr(pd, 'isna') and pd.isna(v)):
                                        continue
                                    row[k] = float(v) if k != 'trades' else int(v)
                                except Exception:
                                    continue

                        ohlcv.append(row)
                        continue

                    # 2) 数组/元组: [ts, o, h, l, c, v]
                    if isinstance(item, (list, tuple)) and len(item) >= 6:
                        ts_ms = _to_ts_ms(item[0])
                        if ts_ms is None:
                            continue
                        ohlcv.append({
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'open': float(item[1]),
                            'high': float(item[2]),
                            'low': float(item[3]),
                            'close': float(item[4]),
                            'volume': float(item[5]),
                        })
                        continue

                    # 3) dict: {timestamp, open, ...}
                    if isinstance(item, dict):
                        ts_ms = _to_ts_ms(item.get('timestamp') or item.get('datetime') or item.get('time'))
                        if ts_ms is None:
                            continue
                        ohlcv.append({
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'open': float(item.get('open', 0)),
                            'high': float(item.get('high', 0)),
                            'low': float(item.get('low', 0)),
                            'close': float(item.get('close', 0)),
                            'volume': float(item.get('volume', 0)),
                        })
            except Exception as e:
                logger.warning(f"Error fetching OHLCV: {e}")
        
        return jsonify({
            'success': True,
            'exchange': exchange,
            'market': market,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': ohlcv,
            'count': len(ohlcv),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching OHLCV: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/api/fetcher/snapshot', methods=['POST'])
def api_fetcher_snapshot():
    """获取某个 symbol 的市场快照（现货等可用）。

    该接口用于一次性拿到：ticker / orderbook / trades / market_info / ohlcv 等。
    include 为空时返回合理默认集合。
    """
    try:
        data = request.json or {}
        exchange = data.get('exchange', 'binance')
        market = data.get('market', 'spot')
        symbol = data.get('symbol', 'BTC/USDT')
        timeframe = data.get('timeframe', '1h')
        include = data.get('include')

        # limits
        ohlcv_limit = data.get('ohlcv_limit', 200)
        trades_limit = data.get('trades_limit', 200)
        orderbook_limit = data.get('orderbook_limit', 50)
        try:
            ohlcv_limit = int(ohlcv_limit)
        except Exception:
            ohlcv_limit = 200
        try:
            trades_limit = int(trades_limit)
        except Exception:
            trades_limit = 200
        try:
            orderbook_limit = int(orderbook_limit)
        except Exception:
            orderbook_limit = 50

        # onchain/social：不走该接口
        if str(market).lower() in ('onchain', 'social'):
            return jsonify({
                'success': False,
                'error': f"Market '{market}' does not support snapshot",
                'hint': '请使用对应链上/舆情接口，或切换到 spot/swap/future/option/margin'
            }), 400

        fetcher = get_or_create_fetcher(exchange, market)
        if hasattr(fetcher, 'add_symbols'):
            fetcher.add_symbols([symbol])

        # 统一把 timestamp 转 ms
        def _to_ts_ms(ts_val):
            if ts_val is None:
                return None
            if hasattr(ts_val, 'timestamp') and callable(getattr(ts_val, 'timestamp')):
                return int(ts_val.timestamp() * 1000)
            if isinstance(ts_val, (int, float)):
                return int(ts_val) if ts_val > 1e12 else int(ts_val * 1000)
            try:
                return int(pd.Timestamp(ts_val).timestamp() * 1000)
            except Exception:
                return None

        def _serialize_trade(t):
            if t is None:
                return None
            if hasattr(t, 'to_dict'):
                d = t.to_dict()
            elif hasattr(t, '__dict__'):
                d = dict(t.__dict__)
            elif isinstance(t, dict):
                d = dict(t)
            else:
                return {'value': str(t)}

            ts_ms = _to_ts_ms(d.get('timestamp'))
            if ts_ms is not None:
                d['timestamp'] = ts_ms
                d['datetime'] = _iso_utc_from_ms(ts_ms)
            return d

        def _serialize_orderbook(ob):
            if ob is None:
                return None
            if hasattr(ob, 'to_dict'):
                d = ob.to_dict()
            elif hasattr(ob, '__dict__'):
                d = dict(ob.__dict__)
            elif isinstance(ob, dict):
                d = dict(ob)
            else:
                return {'value': str(ob)}

            ts_ms = _to_ts_ms(d.get('timestamp'))
            if ts_ms is not None:
                d['timestamp'] = ts_ms
                d['datetime'] = _iso_utc_from_ms(ts_ms)
            return d

        # 快照（优先用 fetcher 自带聚合方法）
        raw_snapshot = None
        if hasattr(fetcher, 'fetch_market_snapshot') and callable(getattr(fetcher, 'fetch_market_snapshot')):
            raw_snapshot = fetcher.fetch_market_snapshot(
                symbol=symbol,
                timeframe=timeframe,
                ohlcv_limit=ohlcv_limit,
                trades_limit=trades_limit,
                orderbook_limit=orderbook_limit,
                include=include,
            )

        if raw_snapshot is None:
            raw_snapshot = {
                'exchange': exchange,
                'market_type': market,
                'symbol': symbol,
            }

        # 逐项序列化（保证 JSON 可返回）
        out: Dict[str, Any] = {}
        for k, v in (raw_snapshot or {}).items():
            if k == 'ohlcv' and isinstance(v, list):
                o = []
                for item in v:
                    if hasattr(item, 'timestamp') and hasattr(item, 'open'):
                        ts_ms = _to_ts_ms(getattr(item, 'timestamp', None))
                        if ts_ms is None:
                            continue
                        o.append({
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'open': float(getattr(item, 'open', 0)),
                            'high': float(getattr(item, 'high', 0)),
                            'low': float(getattr(item, 'low', 0)),
                            'close': float(getattr(item, 'close', 0)),
                            'volume': float(getattr(item, 'volume', 0)),
                        })
                        continue
                    if isinstance(item, (list, tuple)) and len(item) >= 6:
                        ts_ms = _to_ts_ms(item[0])
                        if ts_ms is None:
                            continue
                        o.append({
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'open': float(item[1]),
                            'high': float(item[2]),
                            'low': float(item[3]),
                            'close': float(item[4]),
                            'volume': float(item[5]),
                        })
                        continue
                    if isinstance(item, dict):
                        ts_ms = _to_ts_ms(item.get('timestamp') or item.get('datetime') or item.get('time'))
                        if ts_ms is None:
                            continue
                        o.append({
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'open': float(item.get('open', 0)),
                            'high': float(item.get('high', 0)),
                            'low': float(item.get('low', 0)),
                            'close': float(item.get('close', 0)),
                            'volume': float(item.get('volume', 0)),
                        })
                out[k] = o
                continue

            if k == 'trades' and isinstance(v, list):
                out[k] = [x for x in (_serialize_trade(t) for t in v) if x is not None]
                continue

            if k == 'orderbook':
                out[k] = _serialize_orderbook(v)
                continue

            if k in ('time', 'timestamp'):
                out[k] = _to_ts_ms(v) if v is not None else None
                continue

            # dict / scalar
            if isinstance(v, dict):
                vv = dict(v)
                ts_ms = _to_ts_ms(vv.get('timestamp'))
                if ts_ms is not None:
                    vv['timestamp'] = ts_ms
                    vv['datetime'] = _iso_utc_from_ms(ts_ms)
                out[k] = vv
                continue

            out[k] = v

        return jsonify({
            'success': True,
            'exchange': exchange,
            'market': market,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': out,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Error fetching snapshot: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/api/manager/fetch', methods=['POST'])
def api_manager_fetch():
    """使用 DataManager 获取和保存数据"""
    try:
        data = request.json
        market = data.get('market', 'spot')
        exchange = data.get('exchange', 'binance')
        symbols = data.get('symbols', ['BTC/USDT'])
        timeframe = data.get('timeframe', '1h')
        limit = data.get('limit', 100)

        # onchain/social 不支持 OHLCV 管理流程
        if str(market).lower() in ('onchain', 'social'):
            return jsonify({
                'success': False,
                'error': f"Market '{market}' does not support OHLCV fetch/save",
                'hint': '请切换到 spot/swap/future/option/margin 或使用对应链上/舆情接口'
            }), 400
        
        if isinstance(symbols, str):
            symbols = [symbols]

        # 规范化 limit
        if limit is not None:
            try:
                limit = int(limit)
                if limit <= 0:
                    limit = None
            except Exception:
                limit = None
        
        manager = get_or_create_manager(market, exchange)
        
        # 添加交易对
        manager.add_symbols(symbols)
        
        # 获取并保存 OHLCV
        results = {}
        for symbol in symbols:
            try:
                if hasattr(manager, 'fetch_all_ohlcv'):
                    # 使用 Manager 的方法
                    manager.fetch_all_ohlcv(timeframe, limit)
                    results[symbol] = {
                        'status': 'success',
                        'symbol': symbol,
                        'message': f'数据已保存'
                    }
                else:
                    results[symbol] = {
                        'status': 'skipped',
                        'symbol': symbol,
                        'message': 'DataManager 不支持此操作'
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                results[symbol] = {
                    'status': 'error',
                    'symbol': symbol,
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'market': market,
            'exchange': exchange,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in manager_fetch: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/manager/info', methods=['POST'])
def api_manager_info():
    """获取 DataManager 信息"""
    try:
        data = request.json
        market = data.get('market', 'spot')
        exchange = data.get('exchange', 'binance')
        
        manager = get_or_create_manager(market, exchange)
        
        return jsonify({
            'success': True,
            'market': market,
            'exchange': exchange,
            'type': manager.__class__.__name__,
            'symbols_count': len(manager.symbols) if hasattr(manager, 'symbols') else 0,
            'message': 'DataManager 信息获取成功'
        })
    except Exception as e:
        logger.error(f"Error in manager_info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/status')
def api_status():
    """获取系统状态"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'cached_fetchers': len(_fetchers_cache),
        'cached_managers': len(_managers_cache),
        'version': '1.0.0'
    })


@app.route('/api/local/data', methods=['POST'])
def api_local_data():
    """直接读取本地已落盘的某个交易对数据。

    用于"数据分析"页面本地优先可视化：直接读文件而不在线 fetch。

    POST JSON params:
      - market: 市场类型（spot/swap/future/option/margin...）
      - exchange: 交易所（binance/okx/bybit...）
      - symbol: 交易对（如 BTC/USDT）
      - timeframe: 时间框架（如 1m/1h/1d）— 可选，默认按首个可用值加载

    Returns:
      - success: bool
      - data: OHLCV/指标数组
      - meta: {market, exchange, symbol, timeframe, source, row_count, ...}
    """
    try:
        data = request.json or {}
        market = str(data.get('market') or 'spot').lower()
        exchange = str(data.get('exchange') or 'binance').lower()
        symbol = str(data.get('symbol') or 'BTC/USDT').upper()
        timeframe = data.get('timeframe', '').strip().lower() or None

        found_file, found_tf = _find_local_merged_file(market=market, exchange=exchange, symbol=symbol, timeframe=timeframe)
        if not found_file:
            return jsonify({
                'success': False,
                'error': f'No local merged data for {symbol} in {market}/{exchange}'
            }), 400

        root = Path(os.path.join(os.getcwd(), 'data_manager_storage', market, exchange))

        try:
            if found_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(found_file)
                if df is None:
                    df = pd.DataFrame()
                if 'timestamp' not in df.columns:
                    # 兼容 timestamp 在 index 的情况
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.copy()
                        df['timestamp'] = (df.index.astype('int64') // 1_000_000).astype('int64')
                        df = df.reset_index(drop=True)
                df = df.where(pd.notna(df), None)
                data_rows = df.to_dict(orient='records')
            else:
                with found_file.open('r', encoding='utf-8') as f:
                    payload = json.load(f)
                data_rows = payload if isinstance(payload, list) else payload.get('data', [])
                if not isinstance(data_rows, list):
                    data_rows = []
        except Exception as e:
            logger.error(f"Error reading local data: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to read local file: {str(e)}'
            }), 500

        return jsonify({
            'success': True,
            'data': data_rows,
            'meta': {
                'market': market,
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': found_tf,
                'source': 'local_storage',
                'row_count': len(data_rows),
                'file_path': str(found_file.relative_to(root.parent.parent)) if root.exists() else str(found_file)
            }
        })

    except Exception as e:
        logger.error(f"Error in api_local_data: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/api/local/data-integrity', methods=['POST'])
def api_local_data_integrity():
    """本地数据完整性检查 API"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        file_path = data.get('file_path')
        timeframe = data.get('timeframe', '1m')
        
        # 安全性检查：只允许检查 data_manager_storage 下的文件
        safe_root = Path(os.path.join(os.getcwd(), 'data_manager_storage')).resolve()
        target_path = Path(file_path).resolve()
        
        if not str(target_path).startswith(str(safe_root)):
             return jsonify({'success': False, 'error': 'Access denied: Path outside storage root'}), 403
             
        report = DataIntegrityVerifier.verify_file(target_path, timeframe)
        
        return jsonify({
            'success': True,
            'report': report
        })
    except Exception as e:
        logger.error(f"Integrity check error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/local/delete-file', methods=['POST'])
def api_local_delete_file():
    """删除本地数据文件（仅允许 data_manager_storage 下的文件）。"""
    try:
        data = request.json or {}
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({'success': False, 'error': 'No file_path provided'}), 400

        safe_root = Path(os.path.join(os.getcwd(), 'data_manager_storage')).resolve()
        target_path = Path(file_path).resolve()

        if not str(target_path).startswith(str(safe_root)):
            return jsonify({'success': False, 'error': 'Access denied: Path outside storage root'}), 403

        if not target_path.exists() or not target_path.is_file():
            return jsonify({'success': False, 'error': 'File not found'}), 404

        try:
            target_path.unlink()
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to delete file: {e}'}), 500

        return jsonify({
            'success': True,
            'deleted': str(target_path)
        })
    except Exception as e:
        logger.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/local/convert-format', methods=['POST'])
def api_local_convert_format():
    """转换本地数据文件格式（json <-> parquet）。"""
    try:
        data = request.json or {}
        file_path = data.get('file_path')
        target_format = str(data.get('target_format') or 'parquet').lower()

        if not file_path:
            return jsonify({'success': False, 'error': 'No file_path provided'}), 400
        if target_format not in ('parquet', 'json'):
            return jsonify({'success': False, 'error': 'target_format must be parquet or json'}), 400

        safe_root = Path(os.path.join(os.getcwd(), 'data_manager_storage')).resolve()
        target_path = Path(file_path).resolve()
        if not str(target_path).startswith(str(safe_root)):
            return jsonify({'success': False, 'error': 'Access denied: Path outside storage root'}), 403
        if not target_path.exists() or not target_path.is_file():
            return jsonify({'success': False, 'error': 'File not found'}), 404

        src_format = 'parquet' if target_path.suffix == '.parquet' else 'json'
        if src_format == target_format:
            return jsonify({'success': True, 'message': 'Already in target format', 'file': str(target_path)})

        # Load to DataFrame
        if src_format == 'parquet':
            df = pd.read_parquet(target_path)
        else:
            with target_path.open('r', encoding='utf-8') as f:
                payload = json.load(f)
            rows = payload if isinstance(payload, list) else payload.get('data', [])
            df = pd.DataFrame(rows)

        if df.empty:
            return jsonify({'success': False, 'error': 'No data to convert'}), 400

        out_path = None
        if target_format == 'parquet':
            # Try to build canonical parquet path when source is *_merged.json
            rel = target_path.relative_to(safe_root)
            parts = rel.parts
            if target_path.name.endswith('_merged.json') and len(parts) >= 3:
                market = parts[0]
                exchange = parts[1]
                core = target_path.name[:-len('_merged.json')]
                tokens = core.split('_')
                if len(tokens) >= 2:
                    tf = tokens[-1]
                    symbol_key = '_'.join(tokens[:-1])
                    out_dir = safe_root / market / exchange / symbol_key / tf
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / 'ohlcv_merged.parquet'
            if out_path is None:
                out_path = target_path.with_suffix('.parquet')

            df.to_parquet(out_path, index=False, compression='snappy')
        else:
            # parquet -> json
            rel = target_path.relative_to(safe_root)
            parts = rel.parts
            if target_path.name == 'ohlcv_merged.parquet' and len(parts) >= 4:
                tf = parts[-2]
                symbol_key = parts[-3]
                out_name = f"{symbol_key}_{tf}_merged.json"
                out_path = target_path.parent / out_name
            else:
                out_path = target_path.with_suffix('.json')

            rows = df.to_dict('records')
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(rows, f, ensure_ascii=False)

        return jsonify({'success': True, 'output': str(out_path)})
    except Exception as e:
        logger.error(f"Convert format error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/local/index')
def api_local_index():
    """本地最终数据索引（以 data_manager_storage 为准）。

    用于“数据管理”模块本地优先展示：有哪些 market/exchange/symbol/timeframe 已经落盘。

    Query params:
      - market: 过滤市场（spot/swap/future/option/margin...）
      - exchange: 过滤交易所（binance/okx/bybit...）
      - timeframe: 过滤时间框架（如 1m/1h/1d）
      - include_range: 1/true 时，尝试从小文件解析 start/end/count
      - max_files: 最大扫描文件数（默认 5000）
      - force: 1/true 时强制重新扫描（否则使用缓存）
    """
    try:
        market = request.args.get('market')
        exchange = request.args.get('exchange')
        timeframe = request.args.get('timeframe')
        include_range = str(request.args.get('include_range', '')).lower() in ('1', 'true', 'yes', 'y')
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y')

        try:
            max_files = int(request.args.get('max_files', 5000))
        except Exception:
            max_files = 5000
        max_files = max(100, min(max_files, 50000))

        cache_key = f"m={market or ''}|e={exchange or ''}|tf={timeframe or ''}|r={int(include_range)}|n={max_files}"
        now = time.time()

        with _local_index_cache_lock:
            cached = _local_index_cache.get('payload')
            cached_at = float(_local_index_cache.get('cached_at') or 0.0)
            cached_key = _local_index_cache.get('cache_key')

            # 15 秒内同参数复用缓存，避免频繁全盘扫描
            if (not force) and cached and cached_key == cache_key and (now - cached_at) < 15:
                return jsonify({
                    'success': True,
                    'cached': True,
                    'cached_at': cached_at,
                    'generated_at': datetime.now().isoformat(),
                    **cached,
                })

        root = Path(os.path.join(os.getcwd(), 'data_manager_storage'))
        payload = _scan_local_storage_index(
            root_dir=root,
            market=market,
            exchange=exchange,
            timeframe=timeframe,
            max_files=max_files,
            include_range=include_range,
        )

        with _local_index_cache_lock:
            _local_index_cache['cache_key'] = cache_key
            _local_index_cache['cached_at'] = now
            _local_index_cache['payload'] = payload

        return jsonify({
            'success': True,
            'cached': False,
            'generated_at': datetime.now().isoformat(),
            **payload,
        })

    except Exception as e:
        logger.error(f"Error in api_local_index: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }), 400


@app.route('/api/local/incremental-fetch', methods=['POST'])
def api_local_incremental_fetch():
    """增量 fetch：检测缺口并只补缺口/尾部新数据，然后合并回本地 merged 文件。

    POST JSON:
      - market, exchange
      - symbols: ["BTC/USDT", ...] 或单个 symbol
      - timeframe: 可选（为空则自动探测本地文件的 timeframe）
      - include_internal: 是否补内部缺口（默认 true）
      - include_tail: 是否补尾部（默认 true）
      - dry_run: true 时只返回缺口不写文件
      - max_gaps: 最大处理缺口数（默认 200）
    """
    try:
        payload = request.json or {}
        market = str(payload.get('market') or 'spot').lower()
        exchange = str(payload.get('exchange') or 'binance').lower()

        symbols = payload.get('symbols') or payload.get('symbol') or []
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).strip() for s in (symbols or []) if str(s).strip()]
        if not symbols:
            return jsonify({'success': False, 'error': 'No symbols provided'}), 400

        timeframe = str(payload.get('timeframe') or '').strip().lower() or None
        end_time = payload.get('end_time', None)
        end_ms = payload.get('end_ms', None)
        include_internal = str(payload.get('include_internal', 'true')).lower() in ('1', 'true', 'yes', 'y')
        include_tail = str(payload.get('include_tail', 'true')).lower() in ('1', 'true', 'yes', 'y')
        dry_run = str(payload.get('dry_run', 'false')).lower() in ('1', 'true', 'yes', 'y')
        try:
            max_gaps = int(payload.get('max_gaps', 200))
        except Exception:
            max_gaps = 200
        max_gaps = max(1, min(max_gaps, 2000))

        results: Dict[str, Any] = {}
        def _to_ms(v: Any) -> Optional[int]:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return int(v if v > 1_000_000_000_000 else v * 1000)
            try:
                return int(pd.Timestamp(v).timestamp() * 1000)
            except Exception:
                return None

        now_ms = _to_ms(end_ms if end_ms is not None else end_time)

        for sym in symbols:
            try:
                found_file, found_tf = _find_local_merged_file(market=market, exchange=exchange, symbol=sym, timeframe=timeframe)
                if not found_file or not found_tf:
                    results[sym] = {'success': False, 'error': 'No local merged file'}
                    continue

                step_ms = _timeframe_to_ms(found_tf)
                if not step_ms:
                    results[sym] = {'success': False, 'error': f'Unsupported timeframe: {found_tf}'}
                    continue

                existing_rows = []
                if found_file.suffix == '.parquet':
                    try:
                        df_old = pd.read_parquet(found_file)
                        existing_rows = df_old.to_dict('records') if not df_old.empty else []
                    except Exception:
                        existing_rows = []
                else:
                    with found_file.open('r', encoding='utf-8') as f:
                        existing_payload = json.load(f)
                    existing_rows = existing_payload if isinstance(existing_payload, list) else existing_payload.get('data', [])
                    if not isinstance(existing_rows, list):
                        existing_rows = []

                ts_sorted = _extract_sorted_timestamps_ms(existing_rows)
                gaps = _detect_ohlcv_gaps(
                    ts_sorted,
                    step_ms,
                    now_ms=now_ms,
                    include_internal=include_internal,
                    include_tail=include_tail,
                    max_gaps=max_gaps,
                )

                if dry_run or not gaps:
                    results[sym] = {
                        'success': True,
                        'dry_run': bool(dry_run),
                        'market': market,
                        'exchange': exchange,
                        'symbol': sym,
                        'timeframe': found_tf,
                        'file_path': str(found_file),
                        'existing_rows': len(existing_rows),
                        'gaps': gaps,
                        'gaps_count': len(gaps),
                        'missing_bars_total': int(sum(int(g.get('missing_bars') or 0) for g in gaps)),
                    }
                    continue

                fetched_total = 0
                # 用 timestamp 作为主键合并去重
                merged: Dict[int, Dict[str, Any]] = {}
                for r in existing_rows:
                    if not isinstance(r, dict):
                        continue
                    ts = r.get('timestamp')
                    if ts is None:
                        continue
                    try:
                        tms = int(float(ts))
                    except Exception:
                        continue
                    if tms < 1_000_000_000_000:
                        tms *= 1000
                    merged[tms] = r

                for g in gaps:
                    start_ms = int(g['start_ms'])
                    end_ms = int(g['end_ms'])
                    new_rows = _fetch_ohlcv_as_rows(exchange, market, sym, found_tf, start_ms, end_ms)
                    for nr in new_rows:
                        if not isinstance(nr, dict):
                            continue
                        ts = nr.get('timestamp')
                        if ts is None:
                            continue
                        tms = int(float(ts))
                        if tms < 1_000_000_000_000:
                            tms *= 1000
                        merged[tms] = {
                            **merged.get(tms, {}),
                            **nr,
                            'timestamp': tms,
                        }
                    fetched_total += len(new_rows)

                merged_rows = [merged[k] for k in sorted(merged.keys())]

                # 原地写回
                if found_file.suffix == '.parquet':
                    df_out = pd.DataFrame(merged_rows)
                    if 'timestamp' in df_out.columns:
                        ts = pd.to_numeric(df_out['timestamp'], errors='coerce')
                        ts = ts.where(ts >= 1_000_000_000_000, ts * 1000)
                        df_out['timestamp'] = ts
                        df_out = df_out.dropna(subset=['timestamp'])
                        df_out['timestamp'] = df_out['timestamp'].astype('int64')
                        df_out = df_out.sort_values('timestamp').drop_duplicates('timestamp', keep='last')
                    tmp_path = found_file.with_suffix('.tmp.parquet')
                    df_out.to_parquet(tmp_path, index=False, compression='snappy')
                    tmp_path.replace(found_file)
                else:
                    tmp_path = found_file.with_suffix(found_file.suffix + '.tmp')
                    with tmp_path.open('w', encoding='utf-8') as f:
                        json.dump(merged_rows, f, ensure_ascii=False)
                    tmp_path.replace(found_file)

                results[sym] = {
                    'success': True,
                    'dry_run': False,
                    'market': market,
                    'exchange': exchange,
                    'symbol': sym,
                    'timeframe': found_tf,
                    'file_path': str(found_file),
                    'existing_rows': len(existing_rows),
                    'new_rows': len(merged_rows),
                    'fetched_rows': int(fetched_total),
                    'gaps_filled': len(gaps),
                }
            except Exception as e:
                results[sym] = {'success': False, 'error': str(e)}

        return jsonify({
            'success': True,
            'market': market,
            'exchange': exchange,
            'results': results,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Error in api_local_incremental_fetch: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 400


@app.route('/api/local/export', methods=['POST'])
def api_local_export():
    """标准化导出：将本地 merged 数据导出为统一 schema。

    POST JSON:
      - market, exchange
      - symbols: [..] 或单个 symbol
      - timeframe: 可选（为空则自动探测）
      - format: csv|jsonl（默认 csv）

    输出 schema（列）：
      ts_ms, datetime, open, high, low, close, volume, market, exchange, symbol, timeframe
    """
    try:
        payload = request.json or {}
        market = str(payload.get('market') or 'spot').lower()
        exchange = str(payload.get('exchange') or 'binance').lower()

        symbols = payload.get('symbols') or payload.get('symbol') or []
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).strip() for s in (symbols or []) if str(s).strip()]
        if not symbols:
            return jsonify({'success': False, 'error': 'No symbols provided'}), 400

        timeframe = str(payload.get('timeframe') or '').strip().lower() or None
        out_format = str(payload.get('format') or 'csv').strip().lower()
        if out_format not in ('csv', 'jsonl'):
            return jsonify({'success': False, 'error': 'format must be csv or jsonl'}), 400

        export_root = Path(os.path.join(os.getcwd(), 'data', 'exports'))
        export_root.mkdir(parents=True, exist_ok=True)
        export_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        filename = f"export_{market}_{exchange}_{export_id}.{out_format}"
        out_path = export_root / filename

        frames = []
        rows_total = 0
        for sym in symbols:
            found_file, found_tf = _find_local_merged_file(market=market, exchange=exchange, symbol=sym, timeframe=timeframe)
            if not found_file or not found_tf:
                continue
            if found_file.suffix == '.parquet':
                df = pd.read_parquet(found_file)
            else:
                with found_file.open('r', encoding='utf-8') as f:
                    existing_payload = json.load(f)
                existing_rows = existing_payload if isinstance(existing_payload, list) else existing_payload.get('data', [])
                if not isinstance(existing_rows, list) or not existing_rows:
                    continue
                df = pd.DataFrame(existing_rows)
            if df.empty or 'timestamp' not in df.columns:
                continue

            ts = pd.to_numeric(df['timestamp'], errors='coerce')
            ts = ts.fillna(0).astype('int64')
            ts = ts.where(ts >= 1_000_000_000_000, ts * 1000)
            df_out = pd.DataFrame({
                'ts_ms': ts,
                'datetime': pd.to_datetime(ts, unit='ms', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S'),
                'open': pd.to_numeric(df.get('open'), errors='coerce'),
                'high': pd.to_numeric(df.get('high'), errors='coerce'),
                'low': pd.to_numeric(df.get('low'), errors='coerce'),
                'close': pd.to_numeric(df.get('close'), errors='coerce'),
                'volume': pd.to_numeric(df.get('volume'), errors='coerce'),
                'market': market,
                'exchange': exchange,
                'symbol': sym,
                'timeframe': found_tf,
            })
            df_out = df_out.dropna(subset=['ts_ms'])
            frames.append(df_out)
            rows_total += int(len(df_out))

        if not frames:
            return jsonify({'success': False, 'error': 'No exportable local data found'}), 400

        final_df = pd.concat(frames, ignore_index=True)
        final_df = final_df.sort_values(['symbol', 'ts_ms'])

        if out_format == 'csv':
            final_df.to_csv(out_path, index=False, encoding='utf-8')
        else:
            # jsonl
            final_df.to_json(out_path, orient='records', lines=True, force_ascii=False)

        return jsonify({
            'success': True,
            'format': out_format,
            'rows': int(rows_total),
            'symbols': symbols,
            'file': filename,
            'download_url': f"/api/local/export/download/{filename}",
        })

    except Exception as e:
        logger.error(f"Error in api_local_export: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 400


@app.route('/api/local/export/download/<path:filename>', methods=['GET'])
def api_local_export_download(filename: str):
    """下载导出的标准化文件（限定在 data/exports 目录内）。"""
    export_root = Path(os.path.join(os.getcwd(), 'data', 'exports'))
    export_root.mkdir(parents=True, exist_ok=True)

    safe_name = os.path.basename(filename)
    if safe_name != filename:
        return jsonify({'success': False, 'error': 'Invalid filename'}), 400

    if not (export_root / safe_name).exists():
        return jsonify({'success': False, 'error': 'File not found'}), 404

    return send_from_directory(str(export_root), safe_name, as_attachment=True)


# ==================== 新增 API 接口 ====================

@app.route('/api/search/symbols', methods=['POST'])
def api_search_symbols():
    """搜索交易对"""
    try:
        data = request.json
        exchange = data.get('exchange', 'binance').lower()
        market = data.get('market', 'spot').lower()
        query = data.get('query', '').upper()
        limit = data.get('limit', 10000)  # Increased to load all available symbols
        fallback_used = False
        fallback_reason = None

        # onchain/social：避免初始化外部连接，直接返回可选项
        if market == 'onchain':
            network = (exchange or '').lower()
            all_symbols = [
                'network_stats',
                'gas_price',
                'block_number',
                'block_latest',
                'multi_network_stats',
                # 带参数格式（在输入框里替换后面的地址/哈希即可）
                'balance:0xYOUR_ADDRESS',
                'balance:0xYOUR_ADDRESS:0xTOKEN_ADDRESS',
                'contract:0xCONTRACT_ADDRESS',
                'tx:0xTRANSACTION_HASH',
                'block:12345678',
            ]

            # 以太坊特有/优先支持的扩展项
            if network in ('ethereum', 'eth'):
                all_symbols.extend([
                    'eth2_staking',
                    'defi_stats',
                    'nft_stats',
                ])
        elif market == 'social':
            all_symbols = [
                'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'TRX', 'AVAX', 'DOT',
                'MATIC', 'LINK', 'LTC', 'BCH', 'ATOM', 'UNI', 'AAVE', 'SUI', 'OP', 'ARB'
            ]
        else:
            all_symbols = []
        
        if not all_symbols:
            # Always fetch all symbols dynamically from exchange
            try:
                fetcher = get_or_create_fetcher(exchange, market)
                # Try multiple methods to get all available symbols
                if hasattr(fetcher, 'get_available_symbols') and callable(fetcher.get_available_symbols):
                    all_symbols = fetcher.get_available_symbols()
                    logger.info(f"Fetched {len(all_symbols)} symbols using get_available_symbols()")
                elif hasattr(fetcher, 'fetch_symbols') and callable(fetcher.fetch_symbols):
                    all_symbols = fetcher.fetch_symbols()
                    logger.info(f"Fetched {len(all_symbols)} symbols using fetch_symbols()")
                elif hasattr(fetcher, 'symbols'):
                    all_symbols = list(fetcher.symbols) if fetcher.symbols else []
                    logger.info(f"Fetched {len(all_symbols)} symbols using symbols property")
                else:
                    # Fallback to predefined list only if no dynamic method available
                    if exchange in PREDEFINED_SYMBOLS and market in PREDEFINED_SYMBOLS[exchange]:
                        all_symbols = PREDEFINED_SYMBOLS[exchange][market]
                        fallback_used = True
                        fallback_reason = 'no_dynamic_method'
                        logger.warning(f"Using fallback predefined symbols: {len(all_symbols)}")
            except Exception as e:
                logger.warning(f"Failed to get symbols from fetcher: {e}, using predefined")
                if exchange in PREDEFINED_SYMBOLS and market in PREDEFINED_SYMBOLS[exchange]:
                    all_symbols = PREDEFINED_SYMBOLS[exchange][market]
                    fallback_used = True
                    fallback_reason = 'fetcher_exception'

        # 动态加载成功但结果为空：也降级到预定义列表（常见于 OKX 网络/地区限制）
        if not all_symbols and exchange in PREDEFINED_SYMBOLS and market in PREDEFINED_SYMBOLS[exchange]:
            all_symbols = PREDEFINED_SYMBOLS[exchange][market]
            fallback_used = True
            fallback_reason = fallback_reason or 'empty_dynamic_result'
            logger.warning(
                f"Dynamic symbols empty, using predefined fallback: exchange={exchange}, market={market}, count={len(all_symbols)}"
            )
        
        # 搜索匹配的交易对
        if query:
            symbols = [s for s in all_symbols if query in s.upper()][:limit]
        else:
            symbols = all_symbols[:limit]
        
        return jsonify({
            'success': True,
            'query': query,
            'symbols': symbols,
            'count': len(symbols),
            'exchange': exchange,
            'market': market,
            'fallback_used': fallback_used,
            'fallback_reason': fallback_reason,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error searching symbols: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/data/fetch-range', methods=['POST'])
def api_fetch_data_range():
    """按时间范围获取数据"""
    try:
        data = request.json
        exchange = data.get('exchange', 'binance')
        market = data.get('market', 'spot')
        symbol = data.get('symbol', 'BTC/USDT')
        timeframe = data.get('timeframe', '1h')
        start_time = data.get('start_time')  # ISO format: '2025-01-01T00:00:00'
        end_time = data.get('end_time')      # ISO format: '2025-01-31T23:59:59'

        # onchain/social 不支持 OHLCV
        if str(market).lower() in ('onchain', 'social'):
            return jsonify({
                'success': False,
                'error': f"Market '{market}' does not support OHLCV",
                'hint': '请切换到 spot/swap/future/option/margin 或使用对应链上/舆情接口'
            }), 400

        def _to_ts_ms(ts_val):
            if ts_val is None:
                return None
            if hasattr(ts_val, 'timestamp') and callable(getattr(ts_val, 'timestamp')):
                return int(ts_val.timestamp() * 1000)
            if isinstance(ts_val, (int, float)):
                return int(ts_val) if ts_val > 1e12 else int(ts_val * 1000)
            try:
                return int(pd.Timestamp(ts_val).timestamp() * 1000)
            except Exception:
                return None

        start_ms = _to_ts_ms(start_time)
        end_ms = _to_ts_ms(end_time)
        start_dt = datetime.utcfromtimestamp(start_ms / 1000) if start_ms else None
        end_dt = datetime.utcfromtimestamp(end_ms / 1000) if end_ms else None
        
        fetcher = get_or_create_fetcher(exchange, market)
        
        if hasattr(fetcher, 'add_symbols'):
            fetcher.add_symbols([symbol])
        
        # 获取指定时间范围的数据
        ohlcv = []
        if hasattr(fetcher, 'fetch_ohlcv_date_range'):
            try:
                data_list = fetcher.fetch_ohlcv_date_range(
                    symbol, timeframe, start_time, end_time
                )
                ohlcv = []
                for item in data_list:
                    try:
                        ts_ms = _to_ts_ms(item[0])
                        if ts_ms is None:
                            continue
                        ohlcv.append({
                            'timestamp': ts_ms,
                            'datetime': _iso_utc_from_ms(ts_ms),
                            'open': float(item[1]),
                            'high': float(item[2]),
                            'low': float(item[3]),
                            'close': float(item[4]),
                            'volume': float(item[5])
                        })
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"fetch_ohlcv_date_range not available, using alternative method: {e}")
                # 备用方案：优先使用 bulk（可覆盖长时间区间）；否则 fetch_ohlcv + 时间过滤
                if hasattr(fetcher, 'fetch_ohlcv'):
                    used_bulk = False
                    data_list = []

                    if start_dt and end_dt and hasattr(fetcher, 'fetch_ohlcv_bulk'):
                        try:
                            data_list = fetcher.fetch_ohlcv_bulk(
                                symbol=symbol,
                                start_date=start_dt,
                                end_date=end_dt,
                                timeframe=timeframe,
                                max_bars_per_request=1000,
                            )
                            used_bulk = True
                        except Exception as bulk_err:
                            logger.warning(f"fetch_ohlcv_bulk failed, fallback to fetch_ohlcv: {bulk_err}")

                    if not used_bulk:
                        data_list = fetcher.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=start_ms,
                            limit=1000,
                        )

                    # 统一解析输出（兼容模型对象/数组/dict）
                    for item in (data_list or []):
                        if hasattr(item, 'timestamp') and hasattr(item, 'open'):
                            ts_ms = _to_ts_ms(getattr(item, 'timestamp', None))
                            if ts_ms is None:
                                continue
                            if not used_bulk:
                                if start_ms and ts_ms < start_ms:
                                    continue
                                if end_ms and ts_ms > end_ms:
                                    continue
                            ohlcv.append({
                                'timestamp': ts_ms,
                                'datetime': _iso_utc_from_ms(ts_ms),
                                'open': float(getattr(item, 'open', 0)),
                                'high': float(getattr(item, 'high', 0)),
                                'low': float(getattr(item, 'low', 0)),
                                'close': float(getattr(item, 'close', 0)),
                                'volume': float(getattr(item, 'volume', 0)),
                            })
                            continue

                        if isinstance(item, (list, tuple)) and len(item) >= 6:
                            ts_ms = _to_ts_ms(item[0])
                            if ts_ms is None:
                                continue
                            if not used_bulk:
                                if start_ms and ts_ms < start_ms:
                                    continue
                                if end_ms and ts_ms > end_ms:
                                    continue
                            ohlcv.append({
                                'timestamp': ts_ms,
                                'datetime': _iso_utc_from_ms(ts_ms),
                                'open': float(item[1]),
                                'high': float(item[2]),
                                'low': float(item[3]),
                                'close': float(item[4]),
                                'volume': float(item[5]),
                            })
                            continue

                        if isinstance(item, dict):
                            ts_ms = _to_ts_ms(item.get('timestamp') or item.get('datetime') or item.get('time'))
                            if ts_ms is None:
                                continue
                            if not used_bulk:
                                if start_ms and ts_ms < start_ms:
                                    continue
                                if end_ms and ts_ms > end_ms:
                                    continue
                            ohlcv.append({
                                'timestamp': ts_ms,
                                'datetime': _iso_utc_from_ms(ts_ms),
                                'open': float(item.get('open', 0)),
                                'high': float(item.get('high', 0)),
                                'low': float(item.get('low', 0)),
                                'close': float(item.get('close', 0)),
                                'volume': float(item.get('volume', 0)),
                            })
        
        return jsonify({
            'success': True,
            'exchange': exchange,
            'market': market,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': end_time,
            'data': ohlcv,
            'count': len(ohlcv),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching data range: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/api/data/batch-fetch', methods=['POST'])
def api_batch_fetch():
    """异步批量获取多个交易对的数据，立即返回 task_id"""
    try:
        data = request.json
        exchange = data.get('exchange', 'binance')
        market = data.get('market', 'spot')
        symbols = data.get('symbols', ['BTC/USDT'])
        timeframe = data.get('timeframe', '1h')
        limit = data.get('limit', 100)
        start_time = data.get('start_time')  # 前端传入的开始时间（毫秒）
        end_time = data.get('end_time')      # 前端传入的结束时间（毫秒）

        # 仅 spot：允许选择数据类型
        data_type = data.get('data_type') or 'ohlcv'
        include = data.get('include')
        auto_save = data.get('auto_save', True)
        storage_format = str(data.get('storage_format') or 'parquet').lower()

        # onchain/social 支持“指标/舆情”批量获取（非 OHLCV）
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols provided'
            }), 400
        
        # 创建获取任务
        task_id = str(uuid.uuid4())
        task = FetchTask(
            task_id,
            market,
            exchange,
            symbols,
            timeframe,
            limit,
            start_time,
            end_time,
            data_type=data_type,
            include=include,
            auto_save=auto_save,
            storage_format=storage_format
        )
        
        # 保存到任务存储
        with task_store_lock:
            task_store[task_id] = task
        
        # 提交到线程池，异步执行
        task_executor.submit(execute_fetch_task, task)
        
        logger.info(f"Created fetch task {task_id} for {len(symbols)} symbols")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'market': market,
            'exchange': exchange,
            'symbols_count': len(symbols),
            'message': 'Fetch task submitted, use task_id to check progress'
        })
    except Exception as e:
        logger.error(f"Error in batch_fetch: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/data/fetch-progress/<task_id>', methods=['GET'])
def api_fetch_progress(task_id):
    """查询获取数据任务进度"""
    try:
        with task_store_lock:
            if task_id not in task_store:
                return jsonify({
                    'success': False,
                    'error': 'Task not found'
                }), 404
            
            task = task_store[task_id]
            task_info = task.to_dict()
        
        return jsonify({
            'success': True,
            'task_info': task_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in fetch_progress: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/data/save', methods=['POST'])
def api_data_save():
    """异步保存获取的数据，立即返回 task_id"""
    try:
        data = request.json
        market = data.get('market', 'spot')
        exchange = data.get('exchange', 'binance')
        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', '1h')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        storage_format = str(data.get('storage_format') or 'parquet').lower()

        # onchain/social 支持“指标/舆情”保存（非 OHLCV）
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols provided'
            }), 400
        
        # 创建保存任务
        task_id = str(uuid.uuid4())
        task = SaveTask(task_id, market, exchange, symbols, timeframe, start_time, end_time)
        task.storage_format = storage_format
        
        # 保存到任务存储
        with task_store_lock:
            task_store[task_id] = task
        
        # 提交到线程池，异步执行
        task_executor.submit(execute_save_task, task)
        
        logger.info(f"Created save task {task_id} for {len(symbols)} symbols")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'market': market,
            'exchange': exchange,
            'symbols_count': len(symbols),
            'message': 'Save task submitted, use task_id to check progress'
        })
    except Exception as e:
        logger.error(f"Error in data_save: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/data/save-progress/<task_id>', methods=['GET'])
def api_save_progress(task_id):
    """查询保存任务进度"""
    try:
        with task_store_lock:
            if task_id not in task_store:
                return jsonify({
                    'success': False,
                    'error': 'Task not found'
                }), 404
            
            task = task_store[task_id]
            task_info = task.to_dict()
        
        return jsonify({
            'success': True,
            'task_info': task_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in save_progress: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/web/merged-results/save', methods=['POST'])
def api_web_merged_results_save():
    """保存前端合并后的结果（用于刷新页面恢复）。

    存储位置：data_manager_storage/web/latest_merged_results.json
    """
    try:
        payload = request.json or {}
        results = payload.get('results')
        meta = payload.get('meta') or {}

        if not isinstance(results, dict):
            return jsonify({'success': False, 'error': 'results must be an object'}), 400

        record = {
            'saved_at': datetime.now().isoformat(),
            'meta': meta,
            'results': results
        }

        ok_latest = _web_store.save('latest_merged_results', record)
        # 同时保留一份带时间戳的历史（可选）
        try:
            _web_store.save_timestamped('merged_results', record, timestamp=True)
        except Exception:
            pass

        return jsonify({'success': True, 'saved': bool(ok_latest)})
    except Exception as e:
        logger.error(f"Error saving merged results: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/web/merged-results/load', methods=['GET'])
def api_web_merged_results_load():
    """加载最近一次保存的合并结果（若不存在则返回 exists=false）。"""
    try:
        record = _web_store.load('latest_merged_results')
        if not record:
            return jsonify({'success': True, 'exists': False})
        return jsonify({'success': True, 'exists': True, 'payload': record})
    except Exception as e:
        logger.error(f"Error loading merged results: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


# ==================== AlphaGen / RL ====================

@app.route('/api/alphagen/train', methods=['POST'])
def api_alphagen_train():
    """提交 AlphaGen MaskablePPO 训练任务（异步）。"""
    try:
        payload = request.json or {}

        panel = str(payload.get('panel') or '').strip()
        if not panel:
            return jsonify({'success': False, 'error': 'panel is required'}), 400

        timesteps = int(payload.get('timesteps') or 2000)
        max_depth = int(payload.get('max_depth') or 3)
        panel_rows = int(payload.get('panel_rows') or 8000)
        n_folds = int(payload.get('n_folds') or 3)
        embargo_bars = int(payload.get('embargo_bars') or 60)
        seed = int(payload.get('seed') or 7)
        out_model = payload.get('out')

        device = str(payload.get('device') or 'cuda').strip()

        reward_mode = str(payload.get('reward_mode') or 'ic')
        use_fold_median_ic = bool(payload.get('use_fold_median_ic') or False)

        # reward / trade / degenerate 参数（可选）
        w_fold_median_ic = float(payload.get('w_fold_median_ic') or 0.5)
        w_degenerate = float(payload.get('w_degenerate') or 0.3)
        min_unique_ratio = float(payload.get('min_unique_ratio') or 0.002)
        min_factor_std = float(payload.get('min_factor_std') or 1e-6)
        trade_z_thr = float(payload.get('trade_z_thr') or 0.8)
        trade_base_fee = float(payload.get('trade_base_fee') or 0.0005)
        trade_impact_coef = float(payload.get('trade_impact_coef') or 0.02)
        trade_dd_thr = float(payload.get('trade_dd_thr') or -0.05)
        trade_dd_penalty = float(payload.get('trade_dd_penalty') or 2.0)
        trade_min_activity = int(payload.get('trade_min_activity') or 5)

        task_id = str(uuid.uuid4())
        params = {
            'panel': panel,
            'timesteps': timesteps,
            'max_depth': max_depth,
            'panel_rows': panel_rows,
            'n_folds': n_folds,
            'embargo_bars': embargo_bars,
            'seed': seed,
            'out': out_model,
            'reward_mode': reward_mode,
        }
        task = AlphaGenTask(task_id, 'train', params)

        with alphagen_task_store_lock:
            alphagen_task_store[task_id] = task

        cmd = [
            sys.executable, '-m', 'alphagen_style.scripts.train_maskable_ppo',
            '--panel', panel,
            '--timesteps', str(timesteps),
            '--max-depth', str(max_depth),
            '--panel-rows', str(panel_rows),
            '--n-folds', str(n_folds),
            '--embargo-bars', str(embargo_bars),
            '--seed', str(seed),
            '--device', str(device),
            '--reward-mode', str(reward_mode),
            '--w-fold-median-ic', str(w_fold_median_ic),
            '--w-degenerate', str(w_degenerate),
            '--min-unique-ratio', str(min_unique_ratio),
            '--min-factor-std', str(min_factor_std),
            '--trade-z-thr', str(trade_z_thr),
            '--trade-base-fee', str(trade_base_fee),
            '--trade-impact-coef', str(trade_impact_coef),
            '--trade-dd-thr', str(trade_dd_thr),
            '--trade-dd-penalty', str(trade_dd_penalty),
            '--trade-min-activity', str(trade_min_activity),
        ]
        if use_fold_median_ic:
            cmd.append('--use-fold-median-ic')
        if out_model:
            cmd += ['--out', str(out_model)]

        alphagen_executor.submit(_run_subprocess_task, task, cmd, cwd=project_root)

        return jsonify({'success': True, 'task_id': task_id})
    except Exception as e:
        logger.error(f"Error in alphagen_train: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/export-topk', methods=['POST'])
def api_alphagen_export_topk():
    """提交导出 Top-K 表达式任务（异步）。"""
    try:
        payload = request.json or {}
        model = str(payload.get('model') or '').strip()
        panel = str(payload.get('panel') or '').strip()
        if not model:
            return jsonify({'success': False, 'error': 'model is required'}), 400
        if not panel:
            return jsonify({'success': False, 'error': 'panel is required'}), 400

        episodes = int(payload.get('episodes') or 200)
        topk = int(payload.get('topk') or 50)
        max_depth = int(payload.get('max_depth') or 2)
        panel_rows = int(payload.get('panel_rows') or 12000)
        n_folds = int(payload.get('n_folds') or 3)
        embargo_bars = int(payload.get('embargo_bars') or 60)
        seed = int(payload.get('seed') or 7)
        deterministic = bool(payload.get('deterministic') or False)

        reward_mode = str(payload.get('reward_mode') or 'ic')
        use_fold_median_ic = bool(payload.get('use_fold_median_ic') or False)

        w_fold_median_ic = float(payload.get('w_fold_median_ic') or 0.5)
        w_degenerate = float(payload.get('w_degenerate') or 0.3)
        min_unique_ratio = float(payload.get('min_unique_ratio') or 0.002)
        min_factor_std = float(payload.get('min_factor_std') or 1e-6)
        trade_z_thr = float(payload.get('trade_z_thr') or 0.8)
        trade_base_fee = float(payload.get('trade_base_fee') or 0.0005)
        trade_impact_coef = float(payload.get('trade_impact_coef') or 0.02)
        trade_dd_thr = float(payload.get('trade_dd_thr') or -0.05)
        trade_dd_penalty = float(payload.get('trade_dd_penalty') or 2.0)
        trade_min_activity = int(payload.get('trade_min_activity') or 5)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = _alphagen_base_dir()
        out_csv = str((base / f'topk_expr_{ts}.csv').resolve())
        out_json = str((base / f'topk_expr_{ts}.json').resolve())

        task_id = str(uuid.uuid4())
        params = {
            'model': model,
            'panel': panel,
            'episodes': episodes,
            'topk': topk,
            'reward_mode': reward_mode,
            'out_csv': out_csv,
            'out_json': out_json,
        }
        task = AlphaGenTask(task_id, 'export_topk', params)
        with alphagen_task_store_lock:
            alphagen_task_store[task_id] = task

        cmd = [
            sys.executable, '-m', 'alphagen_style.scripts.export_topk_expr',
            '--model', model,
            '--panel', panel,
            '--episodes', str(episodes),
            '--topk', str(topk),
            '--max-depth', str(max_depth),
            '--panel-rows', str(panel_rows),
            '--n-folds', str(n_folds),
            '--embargo-bars', str(embargo_bars),
            '--seed', str(seed),
            '--reward-mode', str(reward_mode),
            '--w-fold-median-ic', str(w_fold_median_ic),
            '--w-degenerate', str(w_degenerate),
            '--min-unique-ratio', str(min_unique_ratio),
            '--min-factor-std', str(min_factor_std),
            '--trade-z-thr', str(trade_z_thr),
            '--trade-base-fee', str(trade_base_fee),
            '--trade-impact-coef', str(trade_impact_coef),
            '--trade-dd-thr', str(trade_dd_thr),
            '--trade-dd-penalty', str(trade_dd_penalty),
            '--trade-min-activity', str(trade_min_activity),
            '--out-csv', out_csv,
            '--out-json', out_json,
        ]
        if use_fold_median_ic:
            cmd.append('--use-fold-median-ic')
        if deterministic:
            cmd.append('--deterministic')

        alphagen_executor.submit(_run_subprocess_task, task, cmd, cwd=project_root)

        return jsonify({'success': True, 'task_id': task_id, 'out_csv': out_csv, 'out_json': out_json})
    except Exception as e:
        logger.error(f"Error in alphagen_export_topk: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/build-panel', methods=['POST'])
def api_alphagen_build_panel():
    """Build panel CSV from local spot merged JSON (async).

    payload:
      - exchange: spot exchange (default binance)
      - merged_path: absolute path to *_merged.json (preferred)
        OR symbol + timeframe (server will infer merged file path)
      - symbol: e.g. BTC/USDT
      - timeframe: e.g. 1m
      - horizon: forward horizon in bars
      - max_rows: max rows to load from merged.json
      - swap_funding: optional absolute path to funding history json
      - swap_oi: optional absolute path to open interest history json
    """
    try:
        payload = request.json or {}
        exchange = str(payload.get('exchange') or 'binance').lower()
        merged_path = str(payload.get('merged_path') or '').strip()
        symbol = str(payload.get('symbol') or '').strip()
        timeframe = str(payload.get('timeframe') or '1m').strip()
        horizon = int(payload.get('horizon') or 60)
        max_rows = int(payload.get('max_rows') or 200_000)

        swap_funding = payload.get('swap_funding')
        swap_oi = payload.get('swap_oi')
        swap_funding = str(swap_funding).strip() if swap_funding else None
        swap_oi = str(swap_oi).strip() if swap_oi else None

        # Infer merged_path if not provided
        if not merged_path:
            if not symbol:
                return jsonify({'success': False, 'error': 'merged_path or symbol is required'}), 400
            if '/' not in symbol:
                return jsonify({'success': False, 'error': 'symbol must look like BTC/USDT'}), 400
            base, quote = symbol.upper().replace('_', '/').split('/', 1)
            fname = f"{base}_{quote}_{timeframe}_merged.json"
            p_json = Path(project_root) / 'data_manager_storage' / 'spot' / exchange / fname
            if p_json.exists():
                merged_path = str(p_json.resolve())
            else:
                # parquet canonical location
                symbol_key = f"{base}_{quote}"
                p_parquet = Path(project_root) / 'data_manager_storage' / 'spot' / exchange / symbol_key / timeframe / 'ohlcv_merged.parquet'
                if not p_parquet.exists():
                    return jsonify({'success': False, 'error': f'Cannot find merged file (json/parquet). Tried: {p_json} and {p_parquet}'}), 404
                merged_path = str(p_parquet.resolve())

        mp = Path(merged_path)
        if not mp.exists() or not mp.is_file():
            return jsonify({'success': False, 'error': f'merged_path not found: {merged_path}'}), 404

        # Output
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        panels_dir = _alphagen_base_dir() / 'panels'
        panels_dir.mkdir(parents=True, exist_ok=True)
        out_csv = str((panels_dir / f"alphagen_panel_{ts}.csv").resolve())

        task_id = str(uuid.uuid4())
        params = {
            'exchange': exchange,
            'merged_path': merged_path,
            'symbol': symbol,
            'timeframe': timeframe,
            'horizon': horizon,
            'max_rows': max_rows,
            'swap_funding': swap_funding,
            'swap_oi': swap_oi,
            'out': out_csv,
        }
        task = AlphaGenTask(task_id, 'build_panel', params)
        with alphagen_task_store_lock:
            alphagen_task_store[task_id] = task

        cmd = [
            sys.executable, '-m', 'alphagen_style.scripts.build_spot_panel',
            '--path', str(mp.resolve()),
            '--timeframe', str(timeframe),
            '--horizon', str(horizon),
            '--max-rows', str(max_rows),
            '--out', str(out_csv),
            '--drop-gaps',  # Enable gap filtering by default (safe for data with outages)
            '--max-gap-multiplier', '3.0',  # Allow up to 3x expected interval
        ]
        if swap_funding:
            cmd += ['--swap-funding', str(Path(swap_funding).resolve())]
        if swap_oi:
            cmd += ['--swap-oi', str(Path(swap_oi).resolve())]

        # Run
        alphagen_executor.submit(_run_subprocess_task, task, cmd, cwd=project_root)
        return jsonify({'success': True, 'task_id': task_id, 'out_csv': out_csv})
    except Exception as e:
        logger.error(f"Error in alphagen_build_panel: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/panel-info', methods=['POST'])
def api_alphagen_panel_info():
    """预检 AlphaGen panel 文件（CSV）并返回统计摘要。"""
    try:
        payload = request.json or {}
        panel = str(payload.get('panel') or '').strip()
        if not panel:
            return jsonify({'success': False, 'error': 'panel is required'}), 400

        p = _resolve_alphagen_panel_path(panel)
        if not p.exists() or not p.is_file():
            return jsonify({'success': False, 'error': f'panel not found: {p}'}), 404
        if p.suffix.lower() != '.csv':
            return jsonify({'success': False, 'error': 'panel must be a .csv file'}), 400

        info = _panel_quick_stats(p)
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        logger.error(f"Error in alphagen_panel_info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/multi-candidates', methods=['POST'])
def api_alphagen_multi_candidates():
    """Multi-Asset 模式候选文件匹配与缺失告警。"""
    try:
        payload = request.json or {}
        panel = str(payload.get('panel') or '').strip()
        info = _infer_multi_pattern(panel)
        if info.get('error'):
            return jsonify({'success': False, 'error': info.get('error')}), 400

        pattern = info.get('pattern')
        if not pattern:
            return jsonify({'success': False, 'error': 'pattern not resolved'}), 400

        import glob

        matches = sorted(glob.glob(pattern))
        matches = [str(Path(m).resolve()) for m in matches if os.path.isfile(m)]
        if len(matches) > 200:
            matches = matches[:200]

        warning = None
        if len(matches) == 0:
            warning = '未匹配到候选文件，请检查 panel 路径/模式'
        elif len(matches) < 2:
            warning = '候选文件数量过少（<2），Multi-Asset 可能退化为单资产'

        return jsonify({
            'success': True,
            'pattern': pattern,
            'source': info.get('source'),
            'candidates': matches,
            'warning': warning,
            'count': len(matches),
        })
    except Exception as e:
        logger.error(f"Error in alphagen_multi_candidates: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/job/<task_id>', methods=['GET'])
def api_alphagen_job(task_id: str):
    """查询 AlphaGen 任务状态与日志。"""
    try:
        with alphagen_task_store_lock:
            task = alphagen_task_store.get(task_id)
            if task is None:
                return jsonify({'success': False, 'error': 'Task not found'}), 404
            info = task.to_dict()
        return jsonify({'success': True, 'task_info': info, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Error in alphagen_job: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/cancel/<task_id>', methods=['POST'])
def api_alphagen_cancel(task_id: str):
    """取消 AlphaGen 任务。"""
    try:
        with alphagen_task_store_lock:
            task = alphagen_task_store.get(task_id)
            if task is None:
                return jsonify({'success': False, 'error': 'Task not found'}), 404

            if task.status in ['completed', 'error', 'timeout', 'cancelled']:
                return jsonify({'success': True, 'task_info': task.to_dict()})

            task.cancelled = True
            task.status = 'cancelled'
            task.error = 'cancelled by user'
            if task.proc is not None:
                try:
                    if task.proc.poll() is None:
                        task.proc.terminate()
                except Exception:
                    pass
        return jsonify({'success': True, 'task_info': task.to_dict()})
    except Exception as e:
        logger.error(f"Error in alphagen_cancel: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/evaluate', methods=['POST'])
def api_alphagen_evaluate():
    """同步评估单个 DSL 表达式（返回 reward + components）。"""
    try:
        def _json_safe(obj):
            """Convert nested objects to JSON-safe types.

            In particular, browsers reject non-standard JSON tokens like NaN/Infinity.
            We convert any non-finite numbers to None.
            """
            if obj is None:
                return None
            if isinstance(obj, (str, bool)):
                return obj
            if isinstance(obj, (int,)):
                return int(obj)

            # numpy scalar types
            try:
                import numpy as _np  # local import to avoid hard dependency at module import time

                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                if isinstance(obj, (_np.floating,)):
                    v = float(obj)
                    return v if _np.isfinite(v) else None
            except Exception:
                pass

            # pandas/datetime
            try:
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
            except Exception:
                pass

            if isinstance(obj, float):
                try:
                    import math as _math

                    return obj if _math.isfinite(obj) else None
                except Exception:
                    return None

            if isinstance(obj, dict):
                return {str(k): _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_safe(v) for v in obj]

            # pandas containers (rare here, but keep safe)
            try:
                if isinstance(obj, pd.Series):
                    return _json_safe(obj.to_dict())
                if isinstance(obj, pd.DataFrame):
                    return _json_safe(obj.to_dict(orient='records'))
            except Exception:
                pass

            # fallback: best-effort scalar conversion
            try:
                if hasattr(obj, 'item'):
                    return _json_safe(obj.item())
            except Exception:
                pass
            return str(obj)

        payload = request.json or {}
        expr = str(payload.get('expr') or '').strip()
        panel = str(payload.get('panel') or '').strip()
        if not expr:
            return jsonify({'success': False, 'error': 'expr is required'}), 400
        if not panel:
            return jsonify({'success': False, 'error': 'panel is required'}), 400

        reward_mode = str(payload.get('reward_mode') or 'ic')
        use_fold_median_ic = bool(payload.get('use_fold_median_ic') or False)
        w_fold_median_ic = float(payload.get('w_fold_median_ic') or 0.5)
        w_degenerate = float(payload.get('w_degenerate') or 0.3)
        min_unique_ratio = float(payload.get('min_unique_ratio') or 0.002)
        min_factor_std = float(payload.get('min_factor_std') or 1e-6)
        trade_z_thr = float(payload.get('trade_z_thr') or 0.8)
        trade_base_fee = float(payload.get('trade_base_fee') or 0.0005)
        trade_impact_coef = float(payload.get('trade_impact_coef') or 0.02)
        trade_dd_thr = float(payload.get('trade_dd_thr') or -0.05)
        trade_dd_penalty = float(payload.get('trade_dd_penalty') or 2.0)
        trade_min_activity = int(payload.get('trade_min_activity') or 5)

        n_folds = int(payload.get('n_folds') or 3)
        embargo_bars = int(payload.get('embargo_bars') or 60)
        ic_freq = str(payload.get('ic_freq') or '1D')
        ic_method = str(payload.get('ic_method') or 'spearman')
        panel_rows = payload.get('panel_rows')

        mode = str(payload.get('mode') or 'single')  # 'single' or 'multi'

        if mode == 'multi':
            # Run multi-asset script 
            if '*' not in panel and not panel.endswith('.csv'):
                 # Heuristic: assume panel is a directory or partial path
                 # If user passed "alphagen_panel_*.csv" or regex
                 pass
            
            # Since web evaluation is synchronous and potentially slow for multi-asset,
            # we should limit the number of files or scope for this "preview" evaluation.
            # But the user wants "Directly do multi asset".
            
            # We will use subprocess to run the script to avoid pollution
            # Or import and run.
            from alphagen_style.scripts.evaluate_multi_asset import evaluate_multi_asset
            import glob
            import tempfile
            
            # Generate a temporary path for output
            fd, tmp_out = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            
            try:
                info = _infer_multi_pattern(panel)
                if info.get('error'):
                    return jsonify({'success': False, 'error': info.get('error')}), 400
                search_pattern = info.get('pattern')
                if not search_pattern:
                    return jsonify({'success': False, 'error': 'pattern not resolved'}), 400
                
                # Call logic
                evaluate_multi_asset(
                    file_pattern=search_pattern,
                    expr=expr,
                    target_col=targets[0],
                    output_path=tmp_out
                )
                
                # Read result
                if os.path.exists(tmp_out):
                    with open(tmp_out, 'r') as f:
                        m_res = json.load(f)
                    os.unlink(tmp_out)
                    payload_out = {
                        'success': True,
                        'mode': 'multi',
                        'expr': expr,
                        'panel': panel,
                        'pattern': search_pattern,
                        'result': _json_safe(m_res),
                    }
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    fname = f"alphagen_eval_{ts}.json"
                    try:
                        base = _alphagen_base_dir()
                        p = base / fname
                        with p.open('w', encoding='utf-8') as f:
                            json.dump(_json_safe(payload_out), f, ensure_ascii=False)
                        payload_out['eval_file'] = fname
                        payload_out['download_url'] = f"/api/alphagen/download/{fname}"
                    except Exception:
                        pass
                    return jsonify(payload_out)
                else:
                    return jsonify({'success': False, 'error': 'Multi-asset eval produced no output'}), 500
                    
            except Exception as e:
                logger.error(traceback.format_exc())
                return jsonify({'success': False, 'error': str(e)}), 500

        from alphagen_style.dsl import analyze_expr, eval_expr
        from alphagen_style.evaluation import EvalConfig, RegimeConfig, RewardConfig, TimeSplitConfig, compute_reward, evaluate_factor_panel

        df = pd.read_csv(panel)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.set_index('timestamp')
        df = df.sort_index()
        if panel_rows is not None:
            pr = int(panel_rows)
            if pr > 0:
                df = df.iloc[:pr].copy()

        targets = ['ret_fwd_log']
        if 'ret_fwd_net_perp' in df.columns:
            targets = ['ret_fwd_log', 'ret_fwd_net_perp']

        eval_cfg = EvalConfig(
            time=TimeSplitConfig(n_folds=n_folds, embargo_bars=embargo_bars),
            regime=RegimeConfig(),
            ic_freq=ic_freq,
            ic_method=ic_method,
        )
        reward_cfg = RewardConfig(
            mode=str(reward_mode),
            use_fold_median_ic=bool(use_fold_median_ic),
            w_fold_median_ic=float(w_fold_median_ic),
            w_degenerate=float(w_degenerate),
            min_unique_ratio=float(min_unique_ratio),
            min_factor_std=float(min_factor_std),
            trade_z_thr=float(trade_z_thr),
            trade_base_fee=float(trade_base_fee),
            trade_impact_coef=float(trade_impact_coef),
            trade_dd_thr=float(trade_dd_thr),
            trade_dd_penalty=float(trade_dd_penalty),
            trade_min_activity=int(trade_min_activity),
        )

        info = analyze_expr(expr)
        if info.warmup_bars >= max(len(df) - 5, 0):
            return jsonify({
                'success': True,
                'expr': expr,
                'reward': -10.0,
                'error': 'warmup_too_large',
                'expr_info': {
                    'columns': info.columns,
                    'functions': info.functions,
                    'max_window': info.max_window,
                    'max_shift': info.max_shift,
                    'warmup_bars': info.warmup_bars,
                },
            })

        tmp = df.copy()
        tmp['alpha'] = eval_expr(expr, tmp)
        tmp_eval = tmp.iloc[int(info.warmup_bars):].copy()

        res = evaluate_factor_panel(tmp_eval, factor_col='alpha', target_cols=targets, cfg=eval_cfg, trade_cfg=reward_cfg)
        r = compute_reward(res, cfg=reward_cfg)

        comps = r.get('components') or {}
        if not isinstance(comps, dict):
            comps = {}

        import math as _math

        def _isfinite(x: float) -> bool:
            try:
                return _math.isfinite(float(x))
            except Exception:
                return False

        def _f(x):
            try:
                return float(x)
            except Exception:
                return float('nan')

        # Weighted contributions ("where it kills the score")
        base_v = _f(comps.get('base'))
        penalty_v = _f(comps.get('penalty'))
        terms = [
            ("missing", float(getattr(reward_cfg, 'w_missing', 0.0)), _f(comps.get('missing_pen'))),
            ("low_n", float(getattr(reward_cfg, 'w_low_n', 0.0)), _f(comps.get('low_n_pen'))),
            ("exposure", float(getattr(reward_cfg, 'w_exposure', 0.0)), _f(comps.get('exposure_pen'))),
            ("autocorr", float(getattr(reward_cfg, 'w_autocorr', 0.0)), _f(comps.get('autocorr_pen'))),
            ("turnover", float(getattr(reward_cfg, 'w_turnover', 0.0)), _f(comps.get('turnover_pen'))),
            ("degenerate", float(getattr(reward_cfg, 'w_degenerate', 0.0)), _f(comps.get('degenerate_pen'))),
            ("fold_instability", float(getattr(reward_cfg, 'w_fold_instability', 0.0)), _f(comps.get('fold_instability'))),
            ("regime_instability", float(getattr(reward_cfg, 'w_regime_instability', 0.0)), _f(comps.get('regime_instability'))),
            ("inconsistency", float(getattr(reward_cfg, 'w_inconsistency', 0.0)), _f(comps.get('inconsistency'))),
        ]

        penalty_terms = []
        for name, w, raw in terms:
            weighted = float(w * raw) if (_isfinite(w) and _isfinite(raw)) else float('nan')
            penalty_terms.append({
                'name': name,
                'weight': float(w),
                'raw': float(raw) if _isfinite(raw) else float('nan'),
                'weighted': float(weighted) if _isfinite(weighted) else float('nan'),
            })

        # Sort by absolute weighted impact (largest killers first)
        def _sort_key(item):
            v = item.get('weighted')
            try:
                v = float(v)
            except Exception:
                v = float('nan')
            return -abs(v) if _isfinite(v) else 1e9

        penalty_terms = sorted(penalty_terms, key=_sort_key)

        reward_breakdown = {
            'mode': str(comps.get('mode') or reward_mode),
            'base': float(base_v) if _isfinite(base_v) else float('nan'),
            'penalty': float(penalty_v) if _isfinite(penalty_v) else float('nan'),
            'reward': float(r.get('reward', float('nan'))),
            'base_ic': _f(comps.get('base_ic')),
            'base_trade': _f(comps.get('base_trade')),
            'penalty_terms': penalty_terms,
        }

        out_payload = {
            'success': True,
            'expr': expr,
            'reward': float(r.get('reward', 0.0)),
            'reward_components': r.get('components'),
            'reward_breakdown': reward_breakdown,
            'eval': res,
            'expr_info': {
                'columns': info.columns,
                'functions': info.functions,
                'max_window': info.max_window,
                'max_shift': info.max_shift,
                'warmup_bars': info.warmup_bars,
            }
        }

        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"alphagen_eval_{ts}.json"
            base = _alphagen_base_dir()
            p = base / fname
            with p.open('w', encoding='utf-8') as f:
                json.dump(_json_safe(out_payload), f, ensure_ascii=False)
            out_payload['eval_file'] = fname
            out_payload['download_url'] = f"/api/alphagen/download/{fname}"
        except Exception:
            pass

        return jsonify(_json_safe(out_payload))
    except Exception as e:
        logger.error(f"Error in alphagen_evaluate: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/exports', methods=['GET'])
def api_alphagen_exports():
    """列出 alphagen 导出目录下的文件。"""
    try:
        base = _alphagen_base_dir()
        files = []
        for p in sorted(base.glob('*')):
            if p.is_file():
                files.append({
                    'name': p.name,
                    'size': p.stat().st_size,
                    'modified_at': _iso_utc_from_s(p.stat().st_mtime),
                    'download_url': f"/api/alphagen/download/{p.name}",
                })
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        logger.error(f"Error in alphagen_exports: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/spot-merged-files', methods=['GET'])
def api_alphagen_spot_merged_files():
    """List locally available spot OHLCV sources for building a panel (json/parquet)."""
    try:
        exchange = str(request.args.get('exchange') or 'binance').lower()
        files = _list_spot_merged_files(exchange=exchange)
        return jsonify({'success': True, 'exchange': exchange, 'files': files})
    except Exception as e:
        logger.error(f"Error in alphagen_spot_merged_files: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/infer-swap-aux', methods=['GET'])
def api_alphagen_infer_swap_aux():
    """Infer local swap funding/OI paths for a given symbol."""
    try:
        exchange = str(request.args.get('exchange') or 'binance').lower()
        symbol = str(request.args.get('symbol') or '').strip()
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol is required'}), 400
        paths = _infer_swap_aux_paths(exchange=exchange, symbol=symbol)
        return jsonify({'success': True, 'exchange': exchange, 'symbol': symbol, 'paths': paths})
    except Exception as e:
        logger.error(f"Error in alphagen_infer_swap_aux: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/preview', methods=['GET'])
def api_alphagen_preview():
    """预览导出 TopK 文件（默认前 20 条）。

    query:
      - file: 导出文件名（限定在 data_manager_storage/web/alphagen 下）
      - limit: 返回条数
    """
    try:
        filename = str(request.args.get('file') or '').strip()
        if not filename:
            return jsonify({'success': False, 'error': 'file is required'}), 400
        limit = int(request.args.get('limit') or 20)
        limit = max(1, min(limit, 200))

        base = _alphagen_base_dir()
        p = _safe_join(base, filename)
        if not p.exists() or not p.is_file():
            return jsonify({'success': False, 'error': 'File not found'}), 404

        suffix = p.suffix.lower()
        rows: List[Dict[str, Any]] = []
        if suffix == '.csv':
            df = pd.read_csv(str(p))
            if 'reward' in df.columns:
                try:
                    df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
                    df = df.sort_values('reward', ascending=False)
                except Exception:
                    pass
            df = df.head(limit)
            rows = df.to_dict(orient='records')
        elif suffix == '.json':
            with open(p, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, list):
                # out_details: list[dict]
                for item in payload[:limit]:
                    if not isinstance(item, dict):
                        continue
                    rows.append({
                        'reward': item.get('reward'),
                        'expr': item.get('expr'),
                        'primary_target': (item.get('reward_components') or {}).get('primary_target') if isinstance(item.get('reward_components'), dict) else None,
                        'ic': (item.get('reward_components') or {}).get('ic') if isinstance(item.get('reward_components'), dict) else None,
                        'ic_ir': (item.get('reward_components') or {}).get('ic_ir') if isinstance(item.get('reward_components'), dict) else None,
                        'penalty': (item.get('reward_components') or {}).get('penalty') if isinstance(item.get('reward_components'), dict) else None,
                    })
            else:
                return jsonify({'success': False, 'error': 'Unsupported json schema'}), 400
        else:
            return jsonify({'success': False, 'error': f'Unsupported file type: {suffix}'}), 400

        # 清洗为 JSON-safe
        cleaned: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            out = {}
            for k, v in r.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    out[k] = v.isoformat()
                elif pd.isna(v) if hasattr(pd, 'isna') else False:
                    out[k] = None
                else:
                    out[k] = v
            cleaned.append(out)

        return jsonify({
            'success': True,
            'file': p.name,
            'limit': limit,
            'rows': cleaned,
        })
    except Exception as e:
        logger.error(f"Error in alphagen_preview: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/alphagen/download/<path:filename>', methods=['GET'])
def api_alphagen_download(filename: str):
    """下载 alphagen 导出文件（限制在 data_manager_storage/web/alphagen 下）。"""
    try:
        base = _alphagen_base_dir()
        p = _safe_join(base, filename)
        if not p.exists() or not p.is_file():
            return jsonify({'success': False, 'error': 'File not found'}), 404
        return send_from_directory(str(base), p.name, as_attachment=True)
    except Exception as e:
        logger.error(f"Error in alphagen_download: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/timeframes', methods=['GET'])
def api_timeframes():
    """获取支持的时间框架"""
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w', '1M']
    return jsonify({
        'success': True,
        'timeframes': timeframes,
        'timestamp': datetime.now().isoformat()
    })


# ==================== ML / Research ====================


def _ml_config_to_dict() -> Dict[str, Any]:
    from machine_learning.config import Config

    cfg = Config()
    return {
        'data_path': str(cfg.data_path),
        'time_col': cfg.time_col,
        'horizon_steps': cfg.horizon_steps,
        'rolling_vol_window': cfg.rolling_vol_window,
        'target_threshold_k': cfg.target_threshold_k,
        'seq_len': cfg.seq_len,
        'batch_size': cfg.batch_size,
        'epochs': cfg.epochs,
        'lr': cfg.lr,
        'pca_components': cfg.pca_components,
        'max_rows': cfg.max_rows,
        'max_runtime_seconds': cfg.max_runtime_seconds,
        'experiments': [
            {
                'name': e.name,
                'model_type': e.model_type,
                'models': e.models,
                'deep_model': e.deep_model,
                'use_pca': e.use_pca,
                'use_ic_weight': e.use_ic_weight,
            }
            for e in cfg.experiments
        ],
    }


def _run_ml_task(task: MLTask) -> None:
    try:
        task.status = 'running'
        task.add_log('ML pipeline started')

        from machine_learning.run_pipeline import run_with_overrides

        def _progress(info: Dict[str, Any]):
            try:
                idx = info.get('index', 0)
                total = info.get('total', 0)
                exp = info.get('exp')
                stage = info.get('stage')
                if total:
                    task.progress = int(idx / total * 100)
                if exp and stage:
                    task.add_log(f"{stage}: {exp} ({idx}/{total})")
            except Exception:
                pass

        def _cancel_cb():
            return bool(task.cancelled)

        payload = dict(task.params or {})
        payload["_cancel_cb"] = _cancel_cb

        summary = run_with_overrides(payload, progress_cb=_progress)
        task.result = summary
        task.progress = 100
        task.status = 'completed'
        task.add_log('ML pipeline completed')
    except Exception as e:
        task.status = 'error'
        task.error = str(e)
        task.add_log(f'Error: {e}')
        logger.error(f"ML task failed: {e}", exc_info=True)


@app.route('/api/ml/config', methods=['GET'])
def api_ml_config():
    try:
        return jsonify({'success': True, 'config': _ml_config_to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/run', methods=['POST'])
def api_ml_run():
    try:
        payload = request.json or {}
        task_id = str(uuid.uuid4())
        task = MLTask(task_id, payload)

        with ml_task_store_lock:
            ml_task_store[task_id] = task

        ml_executor.submit(_run_ml_task, task)

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'ML task submitted'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/task/<task_id>', methods=['GET'])
def api_ml_task(task_id: str):
    try:
        with ml_task_store_lock:
            task = ml_task_store.get(task_id)

        if not task:
            return jsonify({'success': False, 'error': 'Task not found'}), 404

        return jsonify({'success': True, 'task': task.to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/cancel/<task_id>', methods=['POST'])
def api_ml_cancel(task_id: str):
    try:
        with ml_task_store_lock:
            task = ml_task_store.get(task_id)
            if not task:
                return jsonify({'success': False, 'error': 'Task not found'}), 404
            task.cancelled = True
            task.status = 'error'
            task.error = 'Cancelled by user'
            task.add_log('Cancelled by user')

        return jsonify({'success': True, 'task': task.to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/results', methods=['GET'])
def api_ml_results():
    try:
        from machine_learning.config import Config

        cfg = Config()
        summary_path = Path(project_root) / cfg.out_dir / 'summary.json'
        if not summary_path.exists():
            return jsonify({'success': True, 'available': False, 'summary': {}})

        with summary_path.open('r', encoding='utf-8') as f:
            summary = json.load(f)

        return jsonify({'success': True, 'available': True, 'summary': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/market-info/<market>', methods=['GET'])
def api_market_info(market):
    """获取市场信息"""
    market_info = {
        'spot': {
            'name': '现货交易',
            'description': '现货市场交易',
            'supported_exchanges': ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'huobi', 'upbit', 'kraken', 'coinbase'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        },
        'swap': {
            'name': '永续合约',
            'description': '永续合约交易',
            'supported_exchanges': ['binance', 'okx', 'bybit', 'kucoin'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        },
        'future': {
            'name': '期货',
            'description': '期货交易',
            'supported_exchanges': ['binance', 'okx', 'bybit'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        },
        'option': {
            'name': '期权',
            'description': '期权交易',
            'supported_exchanges': ['binance', 'okx', 'bybit'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        },
        'margin': {
            'name': '杠杆交易',
            'description': '杠杆交易',
            'supported_exchanges': ['binance', 'okx'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        },
        'onchain': {
            'name': '链上数据',
            'description': '链上交易分析',
            # 这里的 exchange 字段表示链网络
            'supported_exchanges': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche'],
            'timeframes': ['1h', '1d', '1w']
        },
        'social': {
            'name': '社交数据',
            'description': '社交媒体舆情',
            # 这里的 exchange 字段表示社交平台
            'supported_exchanges': ['twitter', 'reddit', 'telegram'],
            'timeframes': ['1h', '1d', '1w']
        }
    }
    
    if market not in market_info:
        return jsonify({
            'success': False,
            'error': f'Market {market} not found'
        }), 404
    
    return jsonify({
        'success': True,
        'market': market,
        'info': market_info[market],
        'timestamp': datetime.now().isoformat()
    })



@app.route('/api/system/trading_pairs', methods=['GET'])
def get_available_trading_pairs():
    """获取本地存储中所有可用的交易对"""
    storage_root = Path(project_root) / 'data_manager_storage'
    pairs: set[str] = set()
    try:
        from crypto_data_system.storage.data_inventory import DataInventory
        inv = DataInventory(str(storage_root))
        df = inv.scan(consolidate=True)
        if df is not None and hasattr(df, 'empty') and not df.empty:
            # Prefer normalized display symbol (BTC/USDT)
            for s in df.get('symbol', []).tolist():
                if not s or str(s).lower() == 'unknown':
                    continue
                pairs.add(str(s))
    except Exception as e:
        logger.warning(f"trading_pairs inventory scan failed: {e}")

    # Fallback: legacy folder scan (kept for compatibility)
    if not pairs:
        # Scan Spot
        spot_dir = storage_root / 'spot' / 'binance'
        if spot_dir.exists():
            for item in spot_dir.iterdir():
                if item.is_dir() and '_' in item.name:
                    pair_name = item.name.replace('_', '/')
                    pairs.add(pair_name)

    return jsonify({
        'success': True,
        'count': len(pairs),
        'pairs': sorted(list(pairs))
    })


@app.route('/api/symbols/classify', methods=['POST'])
def api_symbols_classify():
    """交易对三级分类（标的资产 + 交易类型 + 计价资产类型）"""
    try:
        payload = request.json or {}
        symbols = payload.get('symbols') or []
        if not isinstance(symbols, list) or not symbols:
            return jsonify({'success': False, 'error': 'symbols is required'}), 400

        exchange = str(payload.get('exchange') or 'binance').lower()
        market = str(payload.get('market') or '').lower().strip() or None

        items = []
        for s in symbols:
            if not s:
                continue
            info = classify_symbol(str(s), exchange=exchange, market_type=market)
            items.append(info)

        return jsonify({
            'success': True,
            'count': len(items),
            'exchange': exchange,
            'market': market or '',
            'items': items,
        })
    except Exception as e:
        logger.error(f"Error in symbols_classify: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': 'API 端点未找到'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': str(error)
    }), 500


@app.route('/api/local/repair-gap', methods=['POST'])
def api_local_repair_gap():
    """修复特定数据补录"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        file_path = data.get('file_path')
        timeframe = data.get('timeframe')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        if not all([file_path, timeframe, start_time, end_time]):
             return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
             
        source_exchange = data.get('source_exchange')

        # Parse path
        safe_root = Path(os.path.join(os.getcwd(), 'data_manager_storage')).resolve()
        target_path = Path(file_path).resolve()
        
        if not str(target_path).startswith(str(safe_root)):
             return jsonify({'success': False, 'error': 'Access denied'}), 403
             
        if not target_path.exists():
             return jsonify({'success': False, 'error': 'File not found'}), 404
             
        # Infer context
        rel_path = target_path.relative_to(safe_root)
        parts = rel_path.parts # (market, exchange, ..., filename)
        
        market = None
        exchange = None
        symbol = None
        
        # Case A: Parquet file in structured directory (e.g. spot/binance/BTC_USDT/1m/ohlcv_merged.parquet)
        if target_path.suffix == '.parquet' and parts[-1] == 'ohlcv_merged.parquet':
            if len(parts) < 5:
                return jsonify({'success': False, 'error': 'Cannot infer context from parquet path'}), 400
            
            # parts: ('spot', 'binance', 'BTC_USDT', '1m', 'ohlcv_merged.parquet')
            market = parts[0]
            exchange = parts[1]
            symbol_dir = parts[-3]
            tf_dir = parts[-2]
            
            if tf_dir != timeframe:
                return jsonify({'success': False, 'error': f'Directory timeframe {tf_dir} != requested {timeframe}'}), 400
                
            symbol_key = symbol_dir
            
        # Case B: JSON merged file (e.g. spot/binance/BTC_USDT_1m_merged.json)
        elif target_path.suffix == '.json':
            if len(parts) < 3:
                return jsonify({'success': False, 'error': 'Cannot infer market/exchange from path'}), 400
            
            market = parts[0]
            exchange = parts[1]
            filename = parts[-1]
            suffix = f"_{timeframe}_merged.json"
            
            if not filename.endswith(suffix):
                return jsonify({'success': False, 'error': f'Filename {filename} does not match timeframe {timeframe}'}), 400
                
            symbol_key = filename[:-len(suffix)]
            
        else:
            return jsonify({'success': False, 'error': f'Unsupported file type or naming: {parts[-1]}'}), 400

        # Symbol reconstruction
        symbol = symbol_key.replace('_', '/')
        if symbol.count('/') >= 2:
             p = symbol.rpartition('/')
             symbol = f"{p[0]}:{p[2]}"

        # Resolve start/end times
        def _to_ms(t):
             if isinstance(t, (int, float)): return int(t if t > 1e12 else t*1000)
             try: return int(pd.Timestamp(t).timestamp() * 1000)
             except: return None
             
        start_ms = _to_ms(start_time)
        end_ms = _to_ms(end_time)
        
        if not start_ms or not end_ms:
             return jsonify({'success': False, 'error': 'Invalid time format'}), 400
             
        # Fetch data (优先 binance，其次原交易所，再 fallback 列表)
        input_proxy = data.get('proxy')

        from crypto_data_system.utils.date_utils import calculate_timeframe_seconds
        tf_sec = calculate_timeframe_seconds(timeframe)
        tf_ms = int(tf_sec * 1000) if tf_sec else None
        expected_count = None
        if tf_ms and end_ms > start_ms:
            expected_count = int((end_ms - start_ms) // tf_ms) + 1

        def _calc_coverage(rows):
            if expected_count is None or not rows:
                return None
            ts_set = set()
            for r in rows:
                t = r.get('timestamp')
                if t is not None:
                    try:
                        ts_set.add(int(t))
                    except Exception:
                        pass
            return len(ts_set) / expected_count if expected_count > 0 else None

        def _fetch_rows_for_exchange(ex_name: str):
            ex_name = str(ex_name).strip().lower()
            if not ex_name:
                return []

            if input_proxy:
                from crypto_data_system import create_fetcher
                temp_config = {'proxy_url': input_proxy}
                try:
                    temp_fetcher = create_fetcher(exchange=ex_name, market_type=market, config=temp_config)
                    try:
                        if hasattr(temp_fetcher, 'add_symbols'):
                            temp_fetcher.add_symbols([symbol])
                    except Exception:
                        pass

                    start_dt = datetime.utcfromtimestamp(int(start_ms) / 1000)
                    end_dt = datetime.utcfromtimestamp(int(end_ms) / 1000)
                    new_rows = []
                    if hasattr(temp_fetcher, 'fetch_ohlcv_bulk') and callable(getattr(temp_fetcher, 'fetch_ohlcv_bulk')):
                        try:
                            df = temp_fetcher.fetch_ohlcv_bulk(
                                symbol=symbol,
                                start_date=start_dt,
                                end_date=end_dt,
                                timeframe=timeframe,
                                max_bars_per_request=1000,
                            )
                            if df is not None and not df.empty:
                                df = df.copy()
                                if 'timestamp' not in df.columns:
                                    if isinstance(df.index, pd.DatetimeIndex):
                                        df['timestamp'] = (df.index.astype('int64') // 1_000_000).astype('int64')
                                    elif 'datetime' in df.columns:
                                        dt = pd.to_datetime(df['datetime'], errors='coerce')
                                        df['timestamp'] = (dt.astype('int64') // 1_000_000).astype('int64')
                                    elif 'date' in df.columns:
                                        dt = pd.to_datetime(df['date'], errors='coerce')
                                        df['timestamp'] = (dt.astype('int64') // 1_000_000).astype('int64')
                                    else:
                                        raise KeyError('timestamp')
                                else:
                                    ts_col = df['timestamp']
                                    if pd.api.types.is_datetime64_any_dtype(ts_col):
                                        df['timestamp'] = (pd.to_datetime(ts_col, errors='coerce').astype('int64') // 1_000_000).astype('int64')
                                    else:
                                        ts_num = pd.to_numeric(ts_col, errors='coerce')
                                        if ts_num.notna().any():
                                            mx = float(ts_num.max())
                                            if mx > 1_000_000_000_000_000:
                                                df['timestamp'] = (ts_num // 1_000_000).astype('int64')
                                            elif mx < 1_000_000_000_000:
                                                df['timestamp'] = (ts_num * 1000).astype('int64')
                                            else:
                                                df['timestamp'] = ts_num.astype('int64')
                                        else:
                                            dt = pd.to_datetime(ts_col, errors='coerce')
                                            df['timestamp'] = (dt.astype('int64') // 1_000_000).astype('int64')

                                df = df.dropna(subset=['timestamp'])
                                df = df[(df['timestamp'] >= start_ms) & (df['timestamp'] <= end_ms)].copy()

                                cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                                for c in cols:
                                    if c not in df.columns:
                                        df[c] = 0.0

                                new_rows = df[cols].to_dict('records')
                        except Exception as e:
                            logger.warning(f"Temp fetcher bulk failed: {e}")
                            new_rows = []

                    if not new_rows and hasattr(temp_fetcher, 'fetch_ohlcv'):
                        try:
                            raw = temp_fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=start_ms, limit=1000)
                            out_rows = []
                            for item in (raw or []):
                                if isinstance(item, (list, tuple)) and len(item) >= 6:
                                    tms = int(item[0])
                                    if tms < 1_000_000_000_000:
                                        tms *= 1000
                                    if tms < start_ms or tms > end_ms:
                                        continue
                                    out_rows.append({
                                        'timestamp': tms,
                                        'open': float(item[1]),
                                        'high': float(item[2]),
                                        'low': float(item[3]),
                                        'close': float(item[4]),
                                        'volume': float(item[5]),
                                    })
                            new_rows = out_rows
                        except Exception as e:
                            logger.warning(f"Temp fetcher fetch_ohlcv failed: {e}")
                            new_rows = []

                    try:
                        if hasattr(temp_fetcher, 'close'):
                            temp_fetcher.close()
                    except Exception:
                        pass

                    return new_rows
                except Exception as e:
                    logger.warning(f"Temp fetcher failed for {ex_name}: {e}")
                    return []

            return _fetch_ohlcv_as_rows(ex_name, market, symbol, timeframe, start_ms, end_ms)

        if source_exchange and str(source_exchange).strip():
            candidates = [str(source_exchange).strip().lower()]
        else:
            fallback_ex = ['okx', 'bybit', 'kucoin', 'gate', 'huobi']
            candidates = ['binance']
            if exchange and str(exchange).strip().lower() not in candidates:
                candidates.append(str(exchange).strip().lower())
            for ex in fallback_ex:
                if ex not in candidates:
                    candidates.append(ex)

        best_rows = []
        best_cov = -1.0
        best_ex = None

        for ex_name in candidates:
            rows = _fetch_rows_for_exchange(ex_name)
            if not rows:
                continue
            cov = _calc_coverage(rows)
            if cov is None:
                best_rows, best_ex, best_cov = rows, ex_name, -1.0
                break
            if cov > best_cov:
                best_rows, best_ex, best_cov = rows, ex_name, cov
            if cov >= 0.95:
                break

        new_rows = best_rows

        if not new_rows:
            return jsonify({'success': False, 'error': f'No data fetched for {symbol} ({start_time} - {end_time})'}), 400

        if expected_count is not None and best_cov >= 0 and best_cov < 0.5:
            return jsonify({
                'success': False,
                'error': f'Coverage too low ({best_cov:.2%}) using {best_ex}',
                'exchange_tried': candidates,
                'coverage': best_cov,
                'expected_count': expected_count,
                'fetched_rows': len(new_rows)
            }), 400

        # Merge Logic
        if target_path.suffix == '.parquet':
            try:
                df_old = pd.read_parquet(target_path)
            except Exception:
                df_old = pd.DataFrame()
            
            # Convert new rows to DataFrame
            df_new = pd.DataFrame(new_rows)
            # 确保列对齐
            if 'timestamp' in df_new.columns:
                 # timestamp 统一用 int ms
                 df_new['timestamp'] = df_new['timestamp'].astype('int64')

            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            
            if 'timestamp' in df_merged.columns:
                # Normalize/sanitize timestamp
                ts = df_merged['timestamp']
                if pd.api.types.is_datetime64_any_dtype(ts):
                    ts = (pd.to_datetime(ts, errors='coerce').astype('int64') // 1_000_000)
                else:
                    ts = pd.to_numeric(ts, errors='coerce')
                    if ts.max() < 1_000_000_000_000:
                        ts = ts * 1000
                    else:
                        ts = ts.where(ts >= 1_000_000_000_000, ts * 1000)

                df_merged = df_merged.copy()
                df_merged['timestamp'] = ts
                df_merged = df_merged.dropna(subset=['timestamp'])
                df_merged = df_merged[df_merged['timestamp'] > 0]
                df_merged['timestamp'] = df_merged['timestamp'].astype('int64')

                df_merged = df_merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Atomic write
            tmp_path = target_path.with_suffix('.tmp.parquet')
            df_merged.to_parquet(tmp_path)
            tmp_path.replace(target_path)
            
            total_rows = len(df_merged)
            
        else: # JSON
            with target_path.open('r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
            existing_rows = existing_data if isinstance(existing_data, list) else existing_data.get('data', [])
            
            merged = {}
            for r in existing_rows:
                ts = r.get('timestamp')
                if ts: merged[int(ts)] = r
                
            for r in new_rows:
                ts = r.get('timestamp')
                if ts: merged[int(ts)] = r
                
            merged_list = [merged[k] for k in sorted(merged.keys())]
            
            tmp_path = target_path.with_suffix(target_path.suffix + '.tmp')
            with tmp_path.open('w', encoding='utf-8') as f:
                json.dump(merged_list, f, ensure_ascii=False)
            tmp_path.replace(target_path)
            
            total_rows = len(merged_list)
        
        return jsonify({
            'success': True,
            'fetched_rows': len(new_rows),
            'total_rows': total_rows,
            'exchange_used': best_ex,
            'coverage': best_cov if expected_count is not None else None,
            'expected_count': expected_count
        })
        
    except Exception as e:
        logger.error(f"Repair gap error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 主函数 ====================

if __name__ == '__main__':
    import logging
    
    # 启用详细日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Crypto Data System Web App...")
    logger.info(f"Open browser at: http://localhost:5000")
    
    # 运行 Flask 应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
