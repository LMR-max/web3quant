# crypto_data_system/storage/data_inventory.py

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class DataInventory:
    """
    本地数据库存管理工具
    
    用于扫描本地存储的数据文件，提取元数据，检查数据连续性等。
    """
    
    def __init__(self, root_dir: str = None):
        if root_dir:
            self.root_dir = Path(root_dir)
        else:
            self.root_dir = Path(os.path.join(os.getcwd(), "data_manager_storage"))

    @staticmethod
    def _normalize_symbol_display(symbol: str) -> str:
        """Normalize symbols for UI/API consumption.

        Storage often uses underscore format (e.g. BTC_USDT). Most APIs/UI expect BTC/USDT.
        """
        s = str(symbol or '').strip()
        if not s:
            return 'unknown'
        s = s.upper()
        if '/' in s:
            return s
        if '_' in s:
            parts = [p for p in s.split('_') if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        return s
            
    def scan(self, consolidate: bool = True) -> pd.DataFrame:
        """
        扫描整个存储目录并生成报告
        
        参数:
            consolidate: 是否合并同一数据源的不同文件格式（例如优先显示 Parquet）
        """
        records = []
        if not self.root_dir.exists():
            return pd.DataFrame()

        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if not (file.endswith('.parquet') or file.endswith('merged.json')):
                    continue
                    
                path = Path(root) / file
                rel_path = path.relative_to(self.root_dir)
                parts = rel_path.parts
                
                info = {
                    'file_path': str(path),
                    'file_name': file,
                    'file_size_mb': round(path.stat().st_size / (1024 * 1024), 2),
                    'last_modified': datetime.fromtimestamp(path.stat().st_mtime),
                    'type': 'unknown',
                    'exchange': 'unknown',
                    'symbol': 'unknown',
                    'symbol_key': 'unknown',
                    'timeframe': 'unknown',
                    'format': 'parquet' if file.endswith('.parquet') else 'json'
                }
                
                if len(parts) >= 2:
                    info['type'] = parts[0]
                    info['exchange'] = parts[1]
                    
                if file == "ohlcv_merged.parquet" and len(parts) >= 4:
                    info['timeframe'] = parts[-2]
                    info['symbol_key'] = parts[-3]
                    info['symbol'] = self._normalize_symbol_display(parts[-3])
                    # 处理 Swap/Future 的 contract_type 子目录
                    if len(parts) > 4:
                        if info['type'] in ['swap', 'future', 'option']:
                           # 此时 parts[2] 可能是 contract_type (e.g. linear)
                           # 结构: future/binance/linear/BTC_USDT/1h/...
                           pass

                if file.endswith("merged.json") and "_" in file:
                    try:
                        name_parts = file.replace("_merged.json", "").split("_")
                        if len(name_parts) >= 2:
                            info['timeframe'] = name_parts[-1]
                            sym_key = "_".join(name_parts[:-1])
                            info['symbol_key'] = sym_key
                            info['symbol'] = self._normalize_symbol_display(sym_key)
                    except Exception:
                        pass

                # Final normalization / defaults
                if info.get('symbol') == 'unknown' and info.get('symbol_key') != 'unknown':
                    info['symbol'] = self._normalize_symbol_display(info.get('symbol_key'))
                if info.get('symbol_key') == 'unknown' and info.get('symbol') != 'unknown':
                    # Keep best-effort key for debugging
                    info['symbol_key'] = str(info.get('symbol')).replace('/', '_')

                # Inspect content
                stats = self._inspect_file(path, info['format'])
                info.update(stats)
                records.append(info)
                
        df = pd.DataFrame(records)
        if df.empty:
            return df
            
        if consolidate:
            # 按 (type, exchange, symbol, timeframe) 分组，优先保留 Parquet
            try:
                # 确保有这就列
                cols = ['type', 'exchange', 'symbol', 'timeframe', 'format']
                if all(c in df.columns for c in cols):
                    # 排序：让 parquet 排在 json 前面 ('parquet' > 'json' is True)
                    df = df.sort_values(['exchange', 'symbol', 'format'], ascending=[True, True, False])
                    # 去重，保留第一个
                    df = df.drop_duplicates(subset=['type', 'exchange', 'symbol', 'timeframe'], keep='first')
            except Exception:
                pass
            
        return df

    def _inspect_file(self, path: Path, fmt: str) -> Dict[str, Any]:
        stats = {
            'count': 0,
            'start_time': None,
            'end_time': None,
            'gaps_count': 0
        }
        
        try:
            if fmt == 'parquet':
                try:
                    # 读取 Timestamp 列
                    # 尝试用 pyarrow 直接读 metadata 会更快，但为了兼容性用 pandas
                    df = pd.read_parquet(path, columns=['timestamp'])
                    if not df.empty:
                        stats['count'] = len(df)
                        ts_min = df['timestamp'].min()
                        ts_max = df['timestamp'].max()
                        stats['start_time'] = pd.to_datetime(ts_min, unit='ms')
                        stats['end_time'] = pd.to_datetime(ts_max, unit='ms')
                except Exception:
                    pass
                    
            elif fmt == 'json':
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        stats['count'] = len(data)
                        first = data[0]
                        last = data[-1]
                        
                        # 尝试解析 timestamp
                        t1 = first.get('timestamp')
                        t2 = last.get('timestamp')
                        
                        if t1 is not None:
                            stats['start_time'] = pd.to_datetime(int(t1), unit='ms')
                        if t2 is not None:
                            stats['end_time'] = pd.to_datetime(int(t2), unit='ms')
                            
        except Exception as e:
            # print(f"Error: {e}")
            pass
            
        return stats

    def check_continuity(self, symbol: str, timeframe: str, exchange: str = 'binance', type_: str = 'spot') -> Dict[str, Any]:
        """
        检查特定数据的连续性（寻找缺失的时间段）
        """
        # 简单实现：通过scan找到路径
        df_index = self.scan(consolidate=False)
        if df_index.empty:
            return {'error': 'No data found'}
            
        target = df_index[
            (df_index['symbol'] == symbol) & 
            (df_index['timeframe'] == timeframe) & 
            (df_index['exchange'] == exchange) &
            (df_index['type'] == type_) & 
            (df_index['format'] == 'parquet')
        ]
        
        if target.empty:
             return {'error': 'Parquet file not found via index'}
             
        path = Path(target.iloc[0]['file_path'])
        
        try:
            df = pd.read_parquet(path)
            if df.empty or 'timestamp' not in df.columns:
                return {'error': 'Empty or invalid data'}
                
            df = df.sort_values('timestamp')
            
            # 转换时间
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            diffs = timestamps.diff().dropna()
            
            # 理论间隔
            from crypto_data_system.utils.date_utils import calculate_timeframe_seconds
            tf_sec = calculate_timeframe_seconds(timeframe)
            if not tf_sec:
                return {'error': f'Invalid timeframe {timeframe}'}

            expected_delta = pd.Timedelta(seconds=tf_sec)
            
            # 寻找间隔大于预期的位置
            # 允许 tiny error?
            gaps = diffs[diffs > expected_delta * 1.05] 
            
            return {
                'total_bars': len(df),
                'expected_interval': str(expected_delta),
                'gaps_found': len(gaps),
                'gap_locations': [t.isoformat() for t in timestamps[gaps.index].tolist()[:10]] # First 10
            }
            
        except Exception as e:
            return {'error': str(e)}

    def delete_dataset(self, symbol: str, timeframe: str, exchange: str, type_: str) -> bool:
        """删除指定数据集（包括 Parquet 和 JSON）"""
        # 待实现
        return False

def get_inventory_summary(root_dir: str = None) -> pd.DataFrame:
    """快捷函数：获取库存摘要"""
    inv = DataInventory(root_dir)
    return inv.scan()

if __name__ == "__main__":
    # 简单的测试运行
    df = get_inventory_summary()
    if not df.empty:
        print(f"Found {len(df)} files.")
        print(df[['type', 'exchange', 'symbol', 'timeframe', 'count', 'start_time', 'end_time']].to_string())
    else:
        print("No data files found.")
