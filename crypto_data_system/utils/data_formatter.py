"""
数据格式化模块
提供数据清洗、转换、标准化和格式化功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime, timedelta
from scipy import stats

# ==================== 配置 ====================

class ColumnType(Enum):
    """列类型枚举"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    RATIO = "ratio"


class DataFormat(Enum):
    """数据格式枚举"""
    DATAFRAME = "dataframe"
    SERIES = "series"
    DICT = "dict"
    LIST = "list"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


@dataclass
class FormatConfig:
    """格式化配置"""
    # 缺失值处理
    missing_value_strategy: str = "fill"  # fill, drop, ignore
    fill_value: Any = None  # 当strategy=fill时的填充值
    fill_method: str = "ffill"  # ffill, bfill, mean, median, mode
    
    # 异常值处理
    outlier_strategy: str = "cap"  # cap, drop, ignore
    outlier_threshold: float = 3.0  # 标准差倍数
    
    # 数据类型转换
    auto_infer_types: bool = True
    convert_to_numeric: bool = True
    datetime_format: str = None  # 自动推断如果为None
    
    # 标准化
    normalize_numeric: bool = False
    normalize_method: str = "zscore"  # zscore, minmax, robust
    
    # 列处理
    rename_columns: bool = True
    column_case: str = "snake"  # snake, camel, pascal, lower, upper
    drop_duplicates: bool = True
    sort_columns: bool = False
    
    # 索引处理
    set_datetime_index: bool = True
    datetime_index_column: str = "timestamp"
    
    # 性能优化
    downcast_numeric: bool = True
    use_category: bool = True
    optimize_memory: bool = True


# ==================== 核心工具类 ====================

class DataFormatter:
    """数据格式化器"""
    
    def __init__(self, config: FormatConfig = None):
        self.config = config or FormatConfig()
        self.column_types: Dict[str, ColumnType] = {}
        self.metadata: Dict[str, Any] = {}
    
    def normalize_column_names(self, df: pd.DataFrame, 
                              case: str = "snake") -> pd.DataFrame:
        """
        规范化列名 - 类方法版本
        """
        return normalize_column_names(df, case)
    
    def format_dataframe(self, df: pd.DataFrame, 
                        config: FormatConfig = None) -> pd.DataFrame:
        """
        格式化DataFrame
        
        参数:
            df: 输入的DataFrame
            config: 格式化配置
            
        返回:
            格式化后的DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 使用提供的配置或实例配置
        current_config = config or self.config
        
        # 复制DataFrame避免修改原始数据
        df_formatted = df.copy()
        
        # 记录原始信息
        self.metadata['original_shape'] = df.shape
        self.metadata['original_columns'] = list(df.columns)
        
        try:
            # 1. 处理列名
            if current_config.rename_columns:
                df_formatted = self.normalize_column_names(df_formatted, 
                                                          case=current_config.column_case)
            
            # 2. 检测列类型
            self._detect_column_types(df_formatted)
            
            # 3. 转换数据类型
            df_formatted = self.convert_dtypes(df_formatted, current_config)
            
            # 4. 处理缺失值
            df_formatted = self.handle_missing_values(df_formatted, current_config)
            
            # 5. 处理异常值
            df_formatted = self.handle_outliers(df_formatted, current_config)
            
            # 6. 标准化数值列
            if current_config.normalize_numeric:
                df_formatted = self.normalize_numeric_columns(df_formatted, 
                                                            current_config.normalize_method)
            
            # 7. 设置索引
            if current_config.set_datetime_index:
                df_formatted = self.set_datetime_index(df_formatted, 
                                                      current_config.datetime_index_column)
            
            # 8. 删除重复行
            if current_config.drop_duplicates:
                df_formatted = self.drop_duplicates(df_formatted)
            
            # 9. 排序列
            if current_config.sort_columns:
                df_formatted = df_formatted.reindex(sorted(df_formatted.columns), axis=1)
            
            # 10. 内存优化
            if current_config.optimize_memory:
                df_formatted = self.optimize_memory(df_formatted)
            
            # 记录最终信息
            self.metadata['formatted_shape'] = df_formatted.shape
            self.metadata['formatted_columns'] = list(df_formatted.columns)
            
            return df_formatted
            
        except Exception as e:
            print(f"格式化DataFrame失败: {e}")
            return df
    
    def _detect_column_types(self, df: pd.DataFrame) -> None:
        """检测列类型"""
        self.column_types.clear()
        
        for column in df.columns:
            col_data = df[column]
            
            # 跳过全空列
            if col_data.isnull().all():
                self.column_types[column] = ColumnType.CATEGORICAL
                continue
            
            # 检测类型
            if pd.api.types.is_numeric_dtype(col_data):
                # 检查是否为百分比
                if col_data.dtype == 'object':
                    if col_data.astype(str).str.contains('%').any():
                        self.column_types[column] = ColumnType.PERCENTAGE
                    else:
                        self.column_types[column] = ColumnType.NUMERIC
                else:
                    self.column_types[column] = ColumnType.NUMERIC
            
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                self.column_types[column] = ColumnType.DATETIME
            
            elif pd.api.types.is_bool_dtype(col_data):
                self.column_types[column] = ColumnType.BOOLEAN
            
            elif pd.api.types.is_string_dtype(col_data):
                # 检查是否为分类数据
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.1:  # 唯一值比例小于10%
                    self.column_types[column] = ColumnType.CATEGORICAL
                else:
                    self.column_types[column] = ColumnType.TEXT
            
            else:
                # 默认为分类
                self.column_types[column] = ColumnType.CATEGORICAL
    
    def convert_dtypes(self, df: pd.DataFrame, 
                      config: FormatConfig = None) -> pd.DataFrame:
        """转换数据类型"""
        config = config or self.config
        df_converted = df.copy()
        
        for column in df.columns:
            col_type = self.column_types.get(column, ColumnType.CATEGORICAL)
            
            try:
                if col_type == ColumnType.NUMERIC and config.convert_to_numeric:
                    # 转换为数值类型
                    df_converted[column] = pd.to_numeric(df_converted[column], 
                                                        errors='coerce')
                
                elif col_type == ColumnType.DATETIME:
                    # 转换为日期时间
                    if config.datetime_format:
                        df_converted[column] = pd.to_datetime(df_converted[column], 
                                                            format=config.datetime_format,
                                                            errors='coerce')
                    else:
                        df_converted[column] = pd.to_datetime(df_converted[column], 
                                                            errors='coerce')
                
                elif col_type == ColumnType.BOOLEAN:
                    # 转换为布尔型
                    df_converted[column] = df_converted[column].astype(bool)
                
                elif col_type == ColumnType.PERCENTAGE:
                    # 转换百分比
                    df_converted[column] = df_converted[column].astype(str).str.rstrip('%')
                    df_converted[column] = pd.to_numeric(df_converted[column], 
                                                        errors='coerce') / 100
                
                elif col_type == ColumnType.CATEGORICAL and config.use_category:
                    # 转换为分类类型
                    df_converted[column] = df_converted[column].astype('category')
            
            except Exception as e:
                print(f"转换列 {column} 失败: {e}")
                continue
        
        return df_converted
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             config: FormatConfig = None) -> pd.DataFrame:
        """处理缺失值"""
        config = config or self.config
        df_cleaned = df.copy()
        
        if config.missing_value_strategy == "ignore":
            return df_cleaned
        
        for column in df.columns:
            col_type = self.column_types.get(column, ColumnType.CATEGORICAL)
            missing_count = df_cleaned[column].isnull().sum()
            
            if missing_count == 0:
                continue
            
            if config.missing_value_strategy == "drop":
                # 删除包含缺失值的行（如果缺失值太多，可能删除整列）
                if missing_count / len(df) < 0.3:  # 缺失值少于30%
                    df_cleaned = df_cleaned.dropna(subset=[column])
                else:
                    print(f"列 {column} 缺失值过多 ({missing_count}/{len(df)}), 考虑填充")
            
            elif config.missing_value_strategy == "fill":
                fill_value = config.fill_value
                
                if fill_value is not None:
                    # 使用指定的填充值
                    df_cleaned[column] = df_cleaned[column].fillna(fill_value)
                else:
                    # 根据列类型选择填充方法
                    if col_type == ColumnType.NUMERIC:
                        if config.fill_method == "mean":
                            fill_val = df_cleaned[column].mean()
                        elif config.fill_method == "median":
                            fill_val = df_cleaned[column].median()
                        elif config.fill_method == "mode":
                            fill_val = df_cleaned[column].mode().iloc[0] if not df_cleaned[column].mode().empty else 0
                        elif config.fill_method == "ffill":
                            df_cleaned[column] = df_cleaned[column].ffill()
                            continue
                        elif config.fill_method == "bfill":
                            df_cleaned[column] = df_cleaned[column].bfill()
                            continue
                        else:
                            fill_val = 0
                        
                        df_cleaned[column] = df_cleaned[column].fillna(fill_val)
                    
                    elif col_type in [ColumnType.CATEGORICAL, ColumnType.TEXT]:
                        if config.fill_method == "mode":
                            fill_val = df_cleaned[column].mode().iloc[0] if not df_cleaned[column].mode().empty else "Unknown"
                        elif config.fill_method == "ffill":
                            df_cleaned[column] = df_cleaned[column].ffill()
                            continue
                        elif config.fill_method == "bfill":
                            df_cleaned[column] = df_cleaned[column].bfill()
                            continue
                        else:
                            fill_val = "Unknown"
                        
                        df_cleaned[column] = df_cleaned[column].fillna(fill_val)
                    
                    elif col_type == ColumnType.DATETIME:
                        if config.fill_method == "ffill":
                            df_cleaned[column] = df_cleaned[column].ffill()
                            continue
                        elif config.fill_method == "bfill":
                            df_cleaned[column] = df_cleaned[column].bfill()
                            continue
                        else:
                            # 使用前一个有效值或后一个有效值
                            df_cleaned[column] = df_cleaned[column].ffill().bfill()
        
        return df_cleaned
    
    def handle_outliers(self, df: pd.DataFrame, 
                       config: FormatConfig = None) -> pd.DataFrame:
        """处理异常值"""
        config = config or self.config
        df_cleaned = df.copy()
        
        if config.outlier_strategy == "ignore":
            return df_cleaned
        
        for column in df.columns:
            col_type = self.column_types.get(column, ColumnType.CATEGORICAL)
            
            if col_type != ColumnType.NUMERIC:
                continue
            
            # 计算z-score
            z_scores = np.abs(stats.zscore(df_cleaned[column].dropna()))
            
            if len(z_scores) == 0:
                continue
            
            # 找出异常值
            outlier_mask = z_scores > config.outlier_threshold
            
            if not outlier_mask.any():
                continue
            
            outlier_count = outlier_mask.sum()
            outlier_ratio = outlier_count / len(z_scores)
            
            print(f"列 {column} 发现 {outlier_count} 个异常值 ({outlier_ratio:.1%})")
            
            if config.outlier_strategy == "drop":
                # 删除异常值
                valid_indices = np.where(~outlier_mask)[0]
                df_cleaned[column].iloc[valid_indices] = df_cleaned[column].iloc[valid_indices]
            
            elif config.outlier_strategy == "cap":
                # 使用上下限截断异常值
                lower_bound = df_cleaned[column].quantile(0.01)
                upper_bound = df_cleaned[column].quantile(0.99)
                
                df_cleaned[column] = np.clip(df_cleaned[column], lower_bound, upper_bound)
        
        return df_cleaned
    
    def normalize_numeric_columns(self, df: pd.DataFrame, 
                                 method: str = "zscore") -> pd.DataFrame:
        """标准化数值列"""
        df_normalized = df.copy()
        
        for column in df.columns:
            col_type = self.column_types.get(column, ColumnType.NUMERIC)
            
            if col_type != ColumnType.NUMERIC:
                continue
            
            col_data = df_normalized[column].copy()
            
            if method == "zscore":
                # Z-score标准化
                mean = col_data.mean()
                std = col_data.std()
                
                if std != 0:
                    df_normalized[column] = (col_data - mean) / std
            
            elif method == "minmax":
                # 最小-最大标准化
                min_val = col_data.min()
                max_val = col_data.max()
                
                if max_val != min_val:
                    df_normalized[column] = (col_data - min_val) / (max_val - min_val)
            
            elif method == "robust":
                # 鲁棒标准化（使用中位数和IQR）
                median = col_data.median()
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                if iqr != 0:
                    df_normalized[column] = (col_data - median) / iqr
        
        return df_normalized
    
    def set_datetime_index(self, df: pd.DataFrame, 
                          datetime_column: str = "timestamp") -> pd.DataFrame:
        """设置日期时间索引"""
        if datetime_column in df.columns:
            try:
                # 确保列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
                    df[datetime_column] = pd.to_datetime(df[datetime_column], 
                                                        errors='coerce')
                
                # 设置索引并排序
                df_indexed = df.set_index(datetime_column)
                df_indexed = df_indexed.sort_index()
                
                return df_indexed
            except Exception as e:
                print(f"设置datetime索引失败: {e}")
        
        return df
    
    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除重复行"""
        if df.index.name and pd.api.types.is_datetime64_any_dtype(df.index):
            # 对于时间序列数据，基于索引删除重复
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='first')]
        else:
            # 对于普通数据，基于所有列删除重复
            df = df.drop_duplicates()
        
        return df
    
    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化内存使用"""
        df_optimized = df.copy()
        
        # 优化数值类型
        if self.config.downcast_numeric:
            for column in df_optimized.select_dtypes(include=['int64', 'int32']).columns:
                df_optimized[column] = pd.to_numeric(df_optimized[column], 
                                                    downcast='integer')
            
            for column in df_optimized.select_dtypes(include=['float64', 'float32']).columns:
                df_optimized[column] = pd.to_numeric(df_optimized[column], 
                                                    downcast='float')
        
        # 优化分类类型
        if self.config.use_category:
            for column in df_optimized.select_dtypes(include=['object']).columns:
                unique_ratio = df_optimized[column].nunique() / len(df_optimized[column])
                if unique_ratio < 0.5:  # 唯一值比例小于50%
                    df_optimized[column] = df_optimized[column].astype('category')
        
        return df_optimized
    
    def get_column_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """获取列摘要信息"""
        summary_data = []
        
        for column in df.columns:
            col_type = self.column_types.get(column, ColumnType.CATEGORICAL)
            col_data = df[column]
            
            # 基本统计
            missing = col_data.isnull().sum()
            missing_pct = missing / len(df) * 100
            unique = col_data.nunique()
            unique_pct = unique / len(df) * 100
            
            col_summary = {
                'column': column,
                'type': col_type.value,
                'dtype': str(col_data.dtype),
                'total': len(df),
                'missing': missing,
                'missing_pct': round(missing_pct, 2),
                'unique': unique,
                'unique_pct': round(unique_pct, 2)
            }
            
            # 数值列额外统计
            if col_type == ColumnType.NUMERIC:
                col_summary.update({
                    'mean': round(col_data.mean(), 4),
                    'std': round(col_data.std(), 4),
                    'min': round(col_data.min(), 4),
                    '25%': round(col_data.quantile(0.25), 4),
                    'median': round(col_data.median(), 4),
                    '75%': round(col_data.quantile(0.75), 4),
                    'max': round(col_data.max(), 4)
                })
            
            # 分类列额外统计
            elif col_type in [ColumnType.CATEGORICAL, ColumnType.TEXT]:
                if unique > 0:
                    top_value = col_data.mode().iloc[0] if not col_data.mode().empty else None
                    top_count = (col_data == top_value).sum() if top_value is not None else 0
                    top_pct = top_count / len(df) * 100
                    
                    col_summary.update({
                        'top_value': str(top_value),
                        'top_count': top_count,
                        'top_pct': round(top_pct, 2)
                    })
            
            summary_data.append(col_summary)
        
        return pd.DataFrame(summary_data)


# ==================== 工具函数 ====================

def normalize_column_names(df: pd.DataFrame, 
                          case: str = "snake") -> pd.DataFrame:
    """
    规范化列名
    
    参数:
        df: DataFrame
        case: 命名规范 (snake, camel, pascal, lower, upper)
    
    返回:
        列名规范化后的DataFrame
    """
    df_renamed = df.copy()
    new_columns = {}
    
    for column in df.columns:
        if not isinstance(column, str):
            column = str(column)
        
        # 去除空白字符
        column_clean = column.strip()
        
        # 替换特殊字符为下划线
        column_clean = re.sub(r'[^\w\s]', '_', column_clean)
        
        # 替换多个空格或下划线为单个下划线
        column_clean = re.sub(r'[\s_]+', '_', column_clean)
        
        # 根据命名规范转换
        if case == "snake":
            # snake_case: all_lowercase_with_underscores
            column_clean = column_clean.lower()
        
        elif case == "camel":
            # camelCase: firstWordLowercaseThenCapitalized
            parts = column_clean.split('_')
            if len(parts) > 1:
                column_clean = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
            else:
                column_clean = column_clean.lower()
        
        elif case == "pascal":
            # PascalCase: AllWordsCapitalized
            parts = column_clean.split('_')
            column_clean = ''.join(p.capitalize() for p in parts)
        
        elif case == "lower":
            # lowercase: all lowercase
            column_clean = column_clean.lower()
        
        elif case == "upper":
            # UPPERCASE: ALL UPPERCASE
            column_clean = column_clean.upper()
        
        # 移除首尾下划线
        column_clean = column_clean.strip('_')
        
        # 如果列名为空，使用默认名
        if not column_clean:
            column_clean = f"column_{len(new_columns)}"
        
        # 确保列名唯一
        if column_clean in new_columns.values():
            counter = 1
            original_name = column_clean
            while column_clean in new_columns.values():
                column_clean = f"{original_name}_{counter}"
                counter += 1
        
        new_columns[column] = column_clean
    
    df_renamed.columns = [new_columns[col] for col in df.columns]
    return df_renamed


def convert_dtypes(df: pd.DataFrame, 
                  numeric_columns: List[str] = None,
                  datetime_columns: List[str] = None,
                  categorical_columns: List[str] = None) -> pd.DataFrame:
    """快速转换数据类型"""
    df_converted = df.copy()
    
    # 转换数值列
    if numeric_columns:
        for col in numeric_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
    
    # 转换日期时间列
    if datetime_columns:
        for col in datetime_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
    
    # 转换分类列
    if categorical_columns:
        for col in categorical_columns:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype('category')
    
    return df_converted


def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = "fill",
                         fill_value: Any = None,
                         fill_method: str = "ffill",
                         drop_threshold: float = 0.3) -> pd.DataFrame:
    """快速处理缺失值"""
    if strategy == "ignore":
        return df
    
    df_cleaned = df.copy()
    
    if strategy == "drop":
        # 删除缺失值过多的列
        cols_to_drop = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > drop_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)
            print(f"删除缺失值过多的列: {cols_to_drop}")
        
        # 删除剩余缺失值的行
        df_cleaned = df_cleaned.dropna()
    
    elif strategy == "fill":
        if fill_value is not None:
            df_cleaned = df_cleaned.fillna(fill_value)
        elif fill_method == "ffill":
            df_cleaned = df_cleaned.ffill().bfill()  # 前后填充
        elif fill_method == "mean":
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
                df_cleaned[numeric_cols].mean()
            )
        elif fill_method == "median":
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
                df_cleaned[numeric_cols].median()
            )
        elif fill_method == "mode":
            for col in df_cleaned.columns:
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val.iloc[0])
    
    return df_cleaned


def format_dataframe(df: pd.DataFrame, 
                    config: Dict = None) -> pd.DataFrame:
    """快速格式化DataFrame（简化版）"""
    if config is None:
        config = {}
    
    df_formatted = df.copy()
    
    # 重命名列
    if config.get('rename_columns', True):
        case = config.get('column_case', 'snake')
        df_formatted = normalize_column_names(df_formatted, case)
    
    # 处理缺失值
    missing_strategy = config.get('missing_strategy', 'fill')
    df_formatted = handle_missing_values(df_formatted, strategy=missing_strategy)
    
    # 设置索引
    datetime_col = config.get('datetime_column')
    if datetime_col and datetime_col in df_formatted.columns:
        try:
            df_formatted[datetime_col] = pd.to_datetime(df_formatted[datetime_col])
            df_formatted = df_formatted.set_index(datetime_col).sort_index()
        except:
            pass
    
    # 删除重复
    if config.get('drop_duplicates', True):
        df_formatted = df_formatted.drop_duplicates()
    
    return df_formatted


# ==================== 特定数据处理函数 ====================

class CryptoDataFormatter:
    """加密货币数据专用格式化器"""
    
    @staticmethod
    def format_ohlcv(df: pd.DataFrame, 
                    timeframe: str = None) -> pd.DataFrame:
        """格式化OHLCV数据"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df_formatted = df.copy()
        
        # 标准OHLCV列名映射
        column_mapping = {
            'open': ['open', 'o', 'Open'],
            'high': ['high', 'h', 'High'],
            'low': ['low', 'l', 'Low'],
            'close': ['close', 'c', 'Close', 'last'],
            'volume': ['volume', 'v', 'Volume', 'vol'],
            'timestamp': ['timestamp', 'time', 'date', 'Date']
        }
        
        # 重命名列
        new_columns = {}
        for target_col, possible_names in column_mapping.items():
            for col in df_formatted.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    new_columns[col] = target_col
                    break
        
        if new_columns:
            df_formatted = df_formatted.rename(columns=new_columns)
        
        # 确保有必要的列
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_formatted.columns]
        
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 添加缺失的列
        if 'volume' not in df_formatted.columns:
            df_formatted['volume'] = 0.0
        
        if 'timestamp' in df_formatted.columns:
            df_formatted['timestamp'] = pd.to_datetime(df_formatted['timestamp'], unit='ms')
            df_formatted = df_formatted.set_index('timestamp').sort_index()
        
        # 添加时间间隔信息
        if timeframe:
            df_formatted['timeframe'] = timeframe
        
        # 添加技术指标列占位符
        tech_cols = ['returns', 'log_returns', 'sma_20', 'sma_50', 'rsi', 'atr']
        for col in tech_cols:
            if col not in df_formatted.columns:
                df_formatted[col] = np.nan
        
        return df_formatted
    
    @staticmethod
    def format_orderbook(orderbook: Dict, symbol: str = None) -> pd.DataFrame:
        """格式化订单簿数据"""
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return pd.DataFrame()
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # 创建DataFrame
        bids_df = pd.DataFrame(bids, columns=['bid_price', 'bid_amount'])
        asks_df = pd.DataFrame(asks, columns=['ask_price', 'ask_amount'])
        
        # 合并买卖盘
        max_len = max(len(bids), len(asks))
        
        if bids:
            bids_df.index = range(len(bids))
        if asks:
            asks_df.index = range(len(asks))
        
        orderbook_df = pd.concat([bids_df, asks_df], axis=1)
        
        # 添加元数据
        orderbook_df['timestamp'] = pd.Timestamp.now()
        if symbol:
            orderbook_df['symbol'] = symbol
        
        # 计算指标
        if len(bids) > 0 and len(asks) > 0:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            
            orderbook_df['best_bid'] = best_bid
            orderbook_df['best_ask'] = best_ask
            orderbook_df['spread'] = spread
            orderbook_df['spread_pct'] = spread_pct
        
        return orderbook_df
    
    @staticmethod
    def format_funding_rates(funding_rates: List[Dict], symbol: str = None) -> pd.DataFrame:
        """格式化资金费率数据"""
        if not funding_rates:
            return pd.DataFrame()
        
        df = pd.DataFrame(funding_rates)
        
        # 重命名列
        column_mapping = {
            'fundingRate': 'funding_rate',
            'fundingTime': 'funding_time',
            'symbol': 'symbol'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 转换数据类型
        if 'funding_rate' in df.columns:
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
        
        if 'funding_time' in df.columns:
            df['funding_time'] = pd.to_datetime(df['funding_time'], unit='ms')
        
        if symbol and 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # 计算累计资金费率
        if 'funding_rate' in df.columns:
            df['cumulative_rate'] = df['funding_rate'].cumsum()
        
        # 设置索引
        if 'funding_time' in df.columns:
            df = df.set_index('funding_time').sort_index()
        
        return df
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if df.empty or 'close' not in df.columns:
            return df
        
        df_indicators = df.copy()
        
        # 价格变动
        df_indicators['returns'] = df_indicators['close'].pct_change(fill_method=None)
        df_indicators['log_returns'] = np.log(df_indicators['close'] / df_indicators['close'].shift(1))
        
        # 移动平均线
        df_indicators['sma_20'] = df_indicators['close'].rolling(window=20).mean()
        df_indicators['sma_50'] = df_indicators['close'].rolling(window=50).mean()
        df_indicators['sma_200'] = df_indicators['close'].rolling(window=200).mean()
        
        df_indicators['ema_12'] = df_indicators['close'].ewm(span=12, adjust=False).mean()
        df_indicators['ema_26'] = df_indicators['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df_indicators['macd'] = df_indicators['ema_12'] - df_indicators['ema_26']
        df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
        df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
        
        # RSI
        delta = df_indicators['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df_indicators['bb_middle'] = df_indicators['close'].rolling(window=20).mean()
        bb_std = df_indicators['close'].rolling(window=20).std()
        df_indicators['bb_upper'] = df_indicators['bb_middle'] + (bb_std * 2)
        df_indicators['bb_lower'] = df_indicators['bb_middle'] - (bb_std * 2)
        
        # ATR (平均真实范围)
        high_low = df_indicators['high'] - df_indicators['low']
        high_close = np.abs(df_indicators['high'] - df_indicators['close'].shift())
        low_close = np.abs(df_indicators['low'] - df_indicators['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df_indicators['atr'] = true_range.rolling(window=14).mean()
        
        # 成交量指标
        if 'volume' in df_indicators.columns:
            df_indicators['volume_sma'] = df_indicators['volume'].rolling(window=20).mean()
            df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_sma']
        
        return df_indicators


# ==================== 测试函数 ====================

def test_data_formatter():
    """测试数据格式化功能"""
    print("=" * 60)
    print("数据格式化模块测试")
    print("=" * 60)
    
    # 创建测试数据
    print("\n1. 创建测试数据...")
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100),
        'Some Text': ['text_' + str(i) for i in range(100)],
        'Category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # 添加一些缺失值
    df.loc[10:15, 'Close'] = np.nan
    df.loc[20:25, 'Volume'] = np.nan
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 测试DataFormatter
    print("\n2. 测试DataFormatter类:")
    formatter = DataFormatter()
    df_formatted = formatter.format_dataframe(df)
    
    print(f"格式化后形状: {df_formatted.shape}")
    print(f"格式化后列名: {list(df_formatted.columns)}")
    print(f"格式化后索引: {df_formatted.index.name}")
    
    # 测试列摘要
    print("\n3. 测试列摘要:")
    summary = formatter.get_column_summary(df_formatted)
    print(f"列摘要形状: {summary.shape}")
    print(f"列类型分布:")
    print(summary['type'].value_counts())
    
    # 测试规范化列名
    print("\n4. 测试规范化列名:")
    df_snake = normalize_column_names(df, case='snake')
    print(f"snake_case列名: {list(df_snake.columns)}")
    
    # 测试加密货币数据格式化
    print("\n5. 测试加密货币数据格式化:")
    crypto_formatter = CryptoDataFormatter()
    
    # 测试OHLCV格式化
    ohlcv_df = crypto_formatter.format_ohlcv(df, timeframe='1d')
    print(f"OHLCV数据形状: {ohlcv_df.shape}")
    print(f"OHLCV列名: {list(ohlcv_df.columns)}")
    
    # 测试技术指标计算
    print("\n6. 测试技术指标计算:")
    if not ohlcv_df.empty:
        indicators_df = crypto_formatter.calculate_technical_indicators(ohlcv_df)
        indicator_cols = [col for col in indicators_df.columns if col not in ohlcv_df.columns]
        print(f"计算了 {len(indicator_cols)} 个技术指标:")
        print(f"新列: {indicator_cols}")
    
    # 测试订单簿格式化
    print("\n7. 测试订单簿格式化:")
    mock_orderbook = {
        'bids': [[100.0, 1.5], [99.5, 2.0], [99.0, 3.0]],
        'asks': [[101.0, 1.0], [101.5, 1.5], [102.0, 2.0]]
    }
    orderbook_df = crypto_formatter.format_orderbook(mock_orderbook, symbol='BTC/USDT')
    print(f"订单簿数据形状: {orderbook_df.shape}")
    print(f"订单簿列名: {list(orderbook_df.columns)}")
    
    # 测试快速格式化函数
    print("\n8. 测试快速格式化函数:")
    quick_formatted = format_dataframe(df, {
        'rename_columns': True,
        'missing_strategy': 'fill',
        'datetime_column': 'timestamp'
    })
    print(f"快速格式化形状: {quick_formatted.shape}")
    
    print("\n✅ 数据格式化模块测试完成")


if __name__ == "__main__":
    test_data_formatter()