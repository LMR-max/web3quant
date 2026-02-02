"""
日期时间处理工具模块
提供日期范围处理、时间转换、市场时间判断等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import time
import pytz
from dateutil.relativedelta import relativedelta

# ==================== 配置 ====================

class TimeZone(Enum):
    """时区枚举"""
    UTC = "UTC"
    NY = "America/New_York"  # 纽约
    LONDON = "Europe/London"  # 伦敦
    TOKYO = "Asia/Tokyo"  # 东京
    SHANGHAI = "Asia/Shanghai"  # 上海
    HK = "Asia/Hong_Kong"  # 香港


class MarketHours(Enum):
    """市场交易时间"""
    CRYPTO_24_7 = "crypto_24_7"  # 加密货币市场24/7
    NYSE = "nyse"  # 纽约证券交易所
    LONDON = "london"  # 伦敦证券交易所
    TOKYO = "tokyo"  # 东京证券交易所


@dataclass
class MarketSchedule:
    """市场交易时间表"""
    market: str
    timezone: str
    open_time: str  # "09:30"
    close_time: str  # "16:00"
    days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # 0=周一, 6=周日
    holidays: List[str] = field(default_factory=list)  # 节假日列表


# ==================== 主要工具类 ====================

@dataclass
class DateRange:
    """日期范围"""
    start: datetime
    end: datetime
    timezone: str = "UTC"
    
    def __post_init__(self):
        """后初始化，确保时间类型正确"""
        if isinstance(self.start, str):
            self.start = parse_date_string(self.start)
        if isinstance(self.end, str):
            self.end = parse_date_string(self.end)
        
        # 设置时区
        self.start = self.start.astimezone(pytz.timezone(self.timezone))
        self.end = self.end.astimezone(pytz.timezone(self.timezone))
    
    @property
    def days(self) -> int:
        """总天数"""
        return (self.end - self.start).days
    
    @property
    def hours(self) -> int:
        """总小时数"""
        return int((self.end - self.start).total_seconds() / 3600)
    
    def contains(self, date: datetime) -> bool:
        """检查日期是否在范围内"""
        date = date.astimezone(pytz.timezone(self.timezone))
        return self.start <= date <= self.end
    
    def split_by_days(self, days: int = 30) -> List['DateRange']:
        """按天数分割日期范围"""
        ranges = []
        current_start = self.start
        
        while current_start < self.end:
            current_end = min(current_start + timedelta(days=days), self.end)
            ranges.append(DateRange(current_start, current_end, self.timezone))
            current_start = current_end
        
        return ranges
    
    def split_by_months(self, months: int = 1) -> List['DateRange']:
        """按月数分割日期范围"""
        ranges = []
        current_start = self.start
        
        while current_start < self.end:
            current_end = min(current_start + relativedelta(months=months), self.end)
            ranges.append(DateRange(current_start, current_end, self.timezone))
            current_start = current_end
        
        return ranges
    
    def to_pandas_range(self, freq: str = 'D') -> pd.DatetimeIndex:
        """转换为pandas日期范围"""
        return pd.date_range(start=self.start, end=self.end, freq=freq)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
            'timezone': self.timezone,
            'days': self.days,
            'hours': self.hours
        }
    
    def __str__(self) -> str:
        return f"{self.start.strftime('%Y-%m-%d %H:%M:%S')} to {self.end.strftime('%Y-%m-%d %H:%M:%S')} ({self.timezone})"


class DateTimeUtils:
    """日期时间工具类"""
    
    @staticmethod
    def now(timezone: str = "UTC") -> datetime:
        """获取当前时间"""
        tz = pytz.timezone(timezone)
        return datetime.now(tz)
    
    @staticmethod
    def today(timezone: str = "UTC") -> datetime:
        """获取今天日期"""
        tz = pytz.timezone(timezone)
        return datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    
    @staticmethod
    def yesterday(timezone: str = "UTC") -> datetime:
        """获取昨天日期"""
        today = DateTimeUtils.today(timezone)
        return today - timedelta(days=1)
    
    @staticmethod
    def format_date(date: datetime, fmt: str = "%Y-%m-%d") -> str:
        """格式化日期"""
        return date.strftime(fmt)
    
    @staticmethod
    def parse_date(date_str: str, fmt: str = None) -> datetime:
        """解析日期字符串"""
        if fmt:
            return datetime.strptime(date_str, fmt)
        
        # 尝试常见格式
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y%m%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"无法解析日期字符串: {date_str}")
    
    @staticmethod
    def convert_timezone(date: datetime, from_tz: str, to_tz: str) -> datetime:
        """转换时区"""
        from_tz_obj = pytz.timezone(from_tz)
        to_tz_obj = pytz.timezone(to_tz)
        
        if date.tzinfo is None:
            date = from_tz_obj.localize(date)
        
        return date.astimezone(to_tz_obj)
    
    @staticmethod
    def get_market_hours(market: str = "crypto_24_7") -> Dict:
        """获取市场交易时间"""
        market_hours = {
            "crypto_24_7": {
                "open": "00:00",
                "close": "23:59",
                "timezone": "UTC",
                "days": [0, 1, 2, 3, 4, 5, 6],  # 每天
                "description": "Cryptocurrency markets (24/7)"
            },
            "nyse": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "America/New_York",
                "days": [0, 1, 2, 3, 4],  # 周一到周五
                "description": "New York Stock Exchange"
            },
            "london": {
                "open": "08:00",
                "close": "16:30",
                "timezone": "Europe/London",
                "days": [0, 1, 2, 3, 4],  # 周一到周五
                "description": "London Stock Exchange"
            },
            "tokyo": {
                "open": "09:00",
                "close": "15:00",
                "timezone": "Asia/Tokyo",
                "days": [0, 1, 2, 3, 4],  # 周一到周五
                "description": "Tokyo Stock Exchange"
            }
        }
        
        return market_hours.get(market, market_hours["crypto_24_7"])
    
    @staticmethod
    def is_market_open(market: str = "crypto_24_7", date: datetime = None) -> bool:
        """检查市场是否开放"""
        if market == "crypto_24_7":
            return True
        
        if date is None:
            date = datetime.now(pytz.UTC)
        
        market_info = DateTimeUtils.get_market_hours(market)
        
        # 转换到市场时区
        market_tz = pytz.timezone(market_info["timezone"])
        market_time = date.astimezone(market_tz)
        
        # 检查星期几
        weekday = market_time.weekday()  # 0=周一, 6=周日
        if weekday not in market_info["days"]:
            return False
        
        # 检查时间
        market_open = datetime.strptime(market_info["open"], "%H:%M").time()
        market_close = datetime.strptime(market_info["close"], "%H:%M").time()
        current_time = market_time.time()
        
        return market_open <= current_time <= market_close
    
    @staticmethod
    def next_market_open(market: str = "nyse", date: datetime = None) -> datetime:
        """获取下一个市场开放时间"""
        if date is None:
            date = datetime.now(pytz.UTC)
        
        market_info = DateTimeUtils.get_market_hours(market)
        market_tz = pytz.timezone(market_info["timezone"])
        market_time = date.astimezone(market_tz)
        
        # 获取开放时间
        open_time = datetime.strptime(market_info["open"], "%H:%M").time()
        
        # 如果今天开放且时间早于开放时间，则返回今天开放时间
        if market_time.weekday() in market_info["days"]:
            today_open = datetime.combine(market_time.date(), open_time)
            today_open = market_tz.localize(today_open)
            
            if market_time < today_open:
                return today_open
        
        # 否则找下一个开放日
        days_ahead = 1
        while True:
            next_day = market_time + timedelta(days=days_ahead)
            if next_day.weekday() in market_info["days"]:
                next_open = datetime.combine(next_day.date(), open_time)
                next_open = market_tz.localize(next_open)
                return next_open
            days_ahead += 1


# ==================== 核心工具函数 ====================

def split_date_range(start_date: Union[str, datetime], 
                    end_date: Union[str, datetime], 
                    timeframe: str,
                    max_bars: int = 1000) -> List[Tuple[datetime, datetime]]:
    """
    分割日期范围以适应API限制
    
    参数:
        start_date: 开始日期
        end_date: 结束日期
        timeframe: 时间间隔 (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
        max_bars: 每次请求最大K线数量
    
    返回:
        日期范围列表 [(start1, end1), (start2, end2), ...]
    """
    # 解析日期
    if isinstance(start_date, str):
        start_date = parse_date_string(start_date)
    if isinstance(end_date, str):
        end_date = parse_date_string(end_date)
    
    # 计算每个时间间隔的秒数
    timeframe_seconds = calculate_timeframe_seconds(timeframe)
    
    # 计算最大时间范围（秒）
    max_seconds = max_bars * timeframe_seconds
    
    # 分割日期范围
    chunks = []
    current = start_date
    
    while current < end_date:
        chunk_end = min(current + timedelta(seconds=max_seconds), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end
    
    return chunks


def calculate_timeframe_seconds(timeframe: str) -> int:
    """将时间间隔转换为秒数"""
    timeframe_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800,
        '1M': 2592000,  # 30天
        '1y': 31536000  # 365天
    }
    
    # 支持数字前缀，如 "2h", "3d"
    match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
    if match:
        num = int(match.group(1))
        unit = match.group(2).lower()
        
        # 转换单位
        unit_map = {
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'M': 2592000,
            'y': 31536000
        }
        
        if unit in unit_map:
            return num * unit_map[unit]
    
    # 如果不是数字前缀，直接查找
    if timeframe in timeframe_map:
        return timeframe_map[timeframe]
    
    # 默认返回1小时
    return 3600


def parse_date_string(date_str: str) -> datetime:
    """解析日期字符串，支持多种格式"""
    date_str = str(date_str).strip()
    
    # 常见格式列表
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # 如果都不行，尝试用pandas解析
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except Exception as e:
        raise ValueError(f"无法解析日期字符串: {date_str}. 错误: {e}")


def format_timestamp(timestamp: Union[datetime, int, float], 
                    format_str: str = "%Y-%m-%d %H:%M:%S",
                    timezone: str = "UTC") -> str:
    """格式化时间戳"""
    if isinstance(timestamp, (int, float)):
        # 假设是Unix时间戳（秒或毫秒）
        if timestamp > 1e12:  # 毫秒
            dt = datetime.fromtimestamp(timestamp / 1000, tz=pytz.UTC)
        else:  # 秒
            dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    else:
        dt = timestamp
    
    # 转换时区
    if timezone != "UTC" and dt.tzinfo is not None:
        tz = pytz.timezone(timezone)
        dt = dt.astimezone(tz)
    
    return dt.strftime(format_str)


def get_working_days(start_date: datetime, end_date: datetime, 
                    holidays: List[datetime] = None) -> List[datetime]:
    """获取工作日列表（周一至周五，排除节假日）"""
    if holidays is None:
        holidays = []
    
    days = pd.date_range(start=start_date, end=end_date, freq='B')  # B=工作日
    
    # 排除节假日
    holidays_set = set(pd.to_datetime(holidays).date)
    working_days = [day for day in days if day.date() not in holidays_set]
    
    return working_days


def add_business_days(start_date: datetime, days: int, 
                     holidays: List[datetime] = None) -> datetime:
    """添加工作日"""
    if holidays is None:
        holidays = []
    
    current_date = start_date
    added_days = 0
    
    while added_days < days:
        current_date += timedelta(days=1)
        
        # 检查是否是工作日（周一至周五）且不是节假日
        if current_date.weekday() < 5 and current_date.date() not in holidays:
            added_days += 1
    
    return current_date


def get_timeframe_label(timeframe: str) -> str:
    """获取时间间隔的友好标签"""
    labels = {
        '1m': '1分钟',
        '5m': '5分钟',
        '15m': '15分钟',
        '30m': '30分钟',
        '1h': '1小时',
        '4h': '4小时',
        '1d': '日线',
        '1w': '周线',
        '1M': '月线',
        '1y': '年线'
    }
    
    return labels.get(timeframe, timeframe)


def get_period_dates(period: str, end_date: datetime = None) -> Tuple[datetime, datetime]:
    """获取常用时间段日期范围"""
    if end_date is None:
        end_date = datetime.now(pytz.UTC)
    
    period_map = {
        'today': (end_date.replace(hour=0, minute=0, second=0, microsecond=0), end_date),
        'yesterday': (
            end_date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1),
            end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        ),
        'this_week': (
            end_date - timedelta(days=end_date.weekday()),
            end_date
        ),
        'last_week': (
            end_date - timedelta(days=end_date.weekday() + 7),
            end_date - timedelta(days=end_date.weekday() + 1)
        ),
        'this_month': (
            end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            end_date
        ),
        'last_month': (
            (end_date.replace(day=1) - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(seconds=1)
        ),
        'last_7_days': (end_date - timedelta(days=7), end_date),
        'last_30_days': (end_date - timedelta(days=30), end_date),
        'last_90_days': (end_date - timedelta(days=90), end_date),
        'last_365_days': (end_date - timedelta(days=365), end_date),
        'ytd': (end_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0), end_date),
        'all': (datetime(2009, 1, 3, tzinfo=pytz.UTC), end_date)  # 比特币创世块时间
    }
    
    if period in period_map:
        return period_map[period]
    else:
        # 尝试解析为天数
        match = re.match(r'last_(\d+)_days', period)
        if match:
            days = int(match.group(1))
            return (end_date - timedelta(days=days), end_date)
        
        # 默认返回最近30天
        return (end_date - timedelta(days=30), end_date)


def calculate_actual_bars(start_date: datetime, end_date: datetime, 
                         timeframe: str) -> int:
    """计算实际K线数量"""
    seconds_per_bar = calculate_timeframe_seconds(timeframe)
    total_seconds = (end_date - start_date).total_seconds()
    
    return int(total_seconds / seconds_per_bar)


def align_to_timeframe(timestamp: datetime, timeframe: str, 
                      align_type: str = "floor") -> datetime:
    """将时间对齐到时间间隔"""
    seconds = calculate_timeframe_seconds(timeframe)
    
    # 转换为Unix时间戳（秒）
    unix_seconds = int(timestamp.timestamp())
    
    if align_type == "floor":
        aligned_seconds = (unix_seconds // seconds) * seconds
    elif align_type == "ceil":
        aligned_seconds = ((unix_seconds + seconds - 1) // seconds) * seconds
    elif align_type == "round":
        aligned_seconds = round(unix_seconds / seconds) * seconds
    else:
        raise ValueError(f"不支持的align_type: {align_type}")
    
    return datetime.fromtimestamp(aligned_seconds, tz=timestamp.tzinfo)


def generate_time_series(start_date: datetime, end_date: datetime, 
                        timeframe: str, align: bool = True) -> List[datetime]:
    """生成时间序列"""
    if align:
        start_date = align_to_timeframe(start_date, timeframe, "floor")
        end_date = align_to_timeframe(end_date, timeframe, "ceil")
    
    seconds = calculate_timeframe_seconds(timeframe)
    timestamps = []
    
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(seconds=seconds)
    
    return timestamps


# ==================== 时间序列分析工具 ====================

class TimeSeriesAnalyzer:
    """时间序列分析工具"""
    
    def __init__(self, dates: List[datetime], values: List[float]):
        self.dates = dates
        self.values = values
        self.df = pd.DataFrame({'date': dates, 'value': values})
        self.df.set_index('date', inplace=True)
    
    def resample(self, timeframe: str, method: str = 'mean') -> pd.DataFrame:
        """重采样时间序列"""
        if method == 'mean':
            return self.df.resample(timeframe).mean()
        elif method == 'sum':
            return self.df.resample(timeframe).sum()
        elif method == 'last':
            return self.df.resample(timeframe).last()
        elif method == 'first':
            return self.df.resample(timeframe).first()
        else:
            raise ValueError(f"不支持的重采样方法: {method}")
    
    def get_period_return(self, period: str = 'D') -> pd.Series:
        """获取周期收益率"""
        if period == 'D':
            return self.df['value'].pct_change()
        else:
            resampled = self.resample(period, 'last')
            return resampled['value'].pct_change()
    
    def get_rolling_statistics(self, window: int, 
                              statistics: List[str] = None) -> pd.DataFrame:
        """获取滚动统计量"""
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max']
        
        result = pd.DataFrame(index=self.df.index)
        
        for stat in statistics:
            if stat == 'mean':
                result[f'rolling_mean_{window}'] = self.df['value'].rolling(window=window).mean()
            elif stat == 'std':
                result[f'rolling_std_{window}'] = self.df['value'].rolling(window=window).std()
            elif stat == 'min':
                result[f'rolling_min_{window}'] = self.df['value'].rolling(window=window).min()
            elif stat == 'max':
                result[f'rolling_max_{window}'] = self.df['value'].rolling(window=window).max()
            elif stat == 'median':
                result[f'rolling_median_{window}'] = self.df['value'].rolling(window=window).median()
        
        return result
    
    def detect_anomalies(self, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """检测异常值"""
        df = self.df.copy()
        
        if method == 'zscore':
            df['zscore'] = (df['value'] - df['value'].mean()) / df['value'].std()
            df['is_anomaly'] = df['zscore'].abs() > threshold
        
        elif method == 'iqr':
            Q1 = df['value'].quantile(0.25)
            Q3 = df['value'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df['is_anomaly'] = (df['value'] < lower_bound) | (df['value'] > upper_bound)
        
        return df


# ==================== 测试函数 ====================

def test_date_utils():
    """测试日期工具"""
    print("=" * 60)
    print("日期工具模块测试")
    print("=" * 60)
    
    # 测试日期解析
    print("\n1. 测试日期解析:")
    date_str = "2023-12-01"
    parsed_date = parse_date_string(date_str)
    print(f"解析 '{date_str}': {parsed_date}")
    
    # 测试时间间隔转换
    print("\n2. 测试时间间隔转换:")
    timeframe = "4h"
    seconds = calculate_timeframe_seconds(timeframe)
    print(f"{timeframe} = {seconds} 秒 ({seconds/3600} 小时)")
    
    # 测试日期范围分割
    print("\n3. 测试日期范围分割:")
    start = "2023-01-01"
    end = "2023-12-31"
    chunks = split_date_range(start, end, "1h", max_bars=1000)
    print(f"从 {start} 到 {end} 分割为 {len(chunks)} 个分块")
    print(f"第一个分块: {chunks[0][0]} 到 {chunks[0][1]}")
    
    # 测试DateRange类
    print("\n4. 测试DateRange类:")
    date_range = DateRange(start, end)
    print(f"日期范围: {date_range}")
    print(f"天数: {date_range.days}")
    print(f"小时数: {date_range.hours}")
    
    # 测试市场时间判断
    print("\n5. 测试市场时间判断:")
    is_open = DateTimeUtils.is_market_open("crypto_24_7")
    print(f"加密货币市场现在是否开放: {is_open}")
    
    # 测试常用时间段
    print("\n6. 测试常用时间段:")
    start_date, end_date = get_period_dates("last_30_days")
    print(f"最近30天: {start_date.date()} 到 {end_date.date()}")
    
    # 测试时间对齐
    print("\n7. 测试时间对齐:")
    now = datetime.now()
    aligned = align_to_timeframe(now, "1h", "floor")
    print(f"当前时间: {now}")
    print(f"对齐到小时: {aligned}")
    
    # 测试时间序列生成
    print("\n8. 测试时间序列生成:")
    times = generate_time_series(
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        "4h"
    )
    print(f"生成 {len(times)} 个时间点")
    print(f"前3个: {times[:3]}")
    
    print("\n✅ 日期工具模块测试完成")


# ==================== 便捷函数 ====================

def get_current_time_string(timezone: str = "UTC", 
                           format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间字符串"""
    return format_timestamp(datetime.now(), format_str, timezone)


def get_today_string(timezone: str = "UTC") -> str:
    """获取今天日期字符串"""
    return format_timestamp(datetime.now(), "%Y-%m-%d", timezone)


def get_yesterday_string(timezone: str = "UTC") -> str:
    """获取昨天日期字符串"""
    yesterday = datetime.now() - timedelta(days=1)
    return format_timestamp(yesterday, "%Y-%m-%d", timezone)


def get_timestamp_ms() -> int:
    """获取当前时间戳（毫秒）"""
    return int(time.time() * 1000)


def get_timestamp_seconds() -> int:
    """获取当前时间戳（秒）"""
    return int(time.time())


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    test_date_utils()