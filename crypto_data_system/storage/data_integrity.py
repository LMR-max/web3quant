
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from crypto_data_system.utils.date_utils import calculate_timeframe_seconds

logger = logging.getLogger(__name__)

class DataIntegrityVerifier:
    """
    Data Integrity Verification Module
    
    Provides methods to verify the completeness and continuity of local data files (Parquet/JSON).
    """

    @staticmethod
    def verify_file(file_path: Union[str, Path], timeframe: str) -> Dict[str, Any]:
        """
        Verify a single data file for continuity gaps.

        Args:
            file_path: Path to the .parquet or .json file.
            timeframe: Timeframe string (e.g., '1m', '1h').

        Returns:
            Dict containing integrity report:
                - status: 'ok' or 'warning' or 'error'
                - total_rows: Number of rows
                - start_time: Start timestamp (ISO)
                - end_time: End timestamp (ISO)
                - coverage_pct: Percentage of expected data present (0-100)
                - gaps_count: Number of missing intervals
                - gaps: List of gaps [{start, end, duration_str}]
        """
        path = Path(file_path)
        if not path.exists():
            return {'status': 'error', 'message': f'File not found: {path}'}

        try:
            # 1. Load Data
            df = DataIntegrityVerifier._load_data(path)
            if df.empty:
                return {'status': 'error', 'message': 'File is empty'}
            
            if 'timestamp' not in df.columns:
                return {'status': 'error', 'message': 'Missing timestamp column'}

            # Normalize timestamp to int milliseconds and drop invalid values.
            ts = df['timestamp']
            try:
                if pd.api.types.is_datetime64_any_dtype(ts):
                    ts_ms = (pd.to_datetime(ts, errors='coerce').astype('int64') // 1_000_000)
                else:
                    ts_num = pd.to_numeric(ts, errors='coerce')
                    # If values look like seconds, convert to ms.
                    # Typical ms timestamps are >= 1e12.
                    if ts_num.max() < 1_000_000_000_000:
                        ts_ms = (ts_num * 1000)
                    else:
                        # Mixed: treat <1e12 as seconds
                        ts_ms = ts_num.where(ts_num >= 1_000_000_000_000, ts_num * 1000)
            except Exception:
                return {'status': 'error', 'message': 'Failed to normalize timestamp column'}

            df = df.copy()
            df['timestamp'] = ts_ms
            df = df.dropna(subset=['timestamp'])
            # Remove non-positive timestamps (e.g. 0) that would push range to 1970.
            df = df[df['timestamp'] > 0]

            # If there are obvious outliers far in the past (e.g. 1970) alongside modern ms timestamps,
            # trim them so expected_rows/coverage reflect the real data range.
            try:
                ts_min = float(df['timestamp'].min())
                ts_max = float(df['timestamp'].max())
                if ts_min < 946684800000 and ts_max > 1_200_000_000_000:
                    df = df[df['timestamp'] >= 946684800000]
            except Exception:
                pass

            # Ensure sorted and unique (reset index to avoid non-consecutive labels)
            # NOTE: if we don't reset index, later gap detection that relies on idx-1
            # can fail with KeyError (e.g., idx label 100 -> idx-1 label 99 not present).
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            
            # 2. Analyze Timestamps
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            start_time = timestamps.iloc[0]
            end_time = timestamps.iloc[-1]
            total_rows = len(df)
            
            # 3. Calculate Expected
            tf_seconds = calculate_timeframe_seconds(timeframe)
            if not tf_seconds:
                 return {'status': 'error', 'message': f'Invalid timeframe: {timeframe}'}

            expected_delta = pd.Timedelta(seconds=tf_seconds)
            total_duration_sec = (end_time - start_time).total_seconds()
            
            # Theoretical count (inclusive of start and end)
            expected_rows = int(total_duration_sec / tf_seconds) + 1
            
            if expected_rows == 0:
                coverage = 0.0
            else:
                coverage = min(100.0, (total_rows / expected_rows) * 100.0)

            # 4. Detect Gaps
            # Calculate difference between consecutive timestamps
            if len(timestamps) < 2:
                return {
                    'status': 'ok',
                    'file_name': path.name,
                    'total_rows': total_rows,
                    'expected_rows': expected_rows,
                    'coverage_pct': round(coverage, 2),
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'gaps_count': 0,
                    'head_gaps': []
                }

            diffs = timestamps.diff()

            # Allow a small tolerance (e.g., 5% deviation or minimal ms drift)
            # A gap is defined as diff > expected_delta * 1.05
            gap_mask = diffs > (expected_delta * 1.05)

            # Use positional indices to avoid KeyError on non-consecutive index labels
            gap_positions = [i for i, v in enumerate(gap_mask.tolist()) if bool(v)]

            gaps_report = []
            for pos in gap_positions:
                if pos <= 0:
                    continue
                # gap starts after the previous candle finishes
                prev_ts = timestamps.iloc[pos - 1]
                curr_ts = timestamps.iloc[pos]
                
                # Gap duration
                gap_duration = curr_ts - prev_ts - expected_delta
                
                # Just reporting the hole: (prev_ts + 1 bar) -> (curr_ts)
                hole_start = prev_ts + expected_delta
                hole_end = curr_ts # Start of next existing bar
                
                gaps_report.append({
                    'start': hole_start.isoformat(),
                    'end': hole_end.isoformat(),
                    'duration_hours': round(gap_duration.total_seconds() / 3600, 2),
                    'duration_str': str(gap_duration)
                })
            
            status = 'ok'
            if len(gaps_report) > 0:
                status = 'warning'
            
            return {
                'status': status,
                'file_name': path.name,
                'total_rows': int(total_rows),
                'expected_rows': expected_rows,
                'coverage_pct': round(coverage, 2),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'gaps_count': len(gaps_report),
                'head_gaps': gaps_report[:50]  # Return top 50 gaps to avoid huge JSON
            }

        except Exception as e:
            logger.error(f"Integrity check failed: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    @staticmethod
    def _load_data(path: Path) -> pd.DataFrame:
        """Helper to load generic data file to DataFrame with 'timestamp' column"""
        if path.suffix == '.parquet':
            return pd.read_parquet(path, columns=['timestamp'])
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Handle list of lists or list of dicts
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                if isinstance(item, dict):
                    # List of Dicts
                    # If memory is an issue for huge JSONs, need partial loading, but assume it fits
                    return pd.DataFrame(data, columns=['timestamp'])
                elif isinstance(item, list):
                    # OHLCV list [ts, o, h, l, c, v]
                    # Assume index 0 is timestamp
                    timestamps = [x[0] for x in data]
                    return pd.DataFrame({'timestamp': timestamps})
        return pd.DataFrame()
