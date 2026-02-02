"""
日志配置模块
提供统一的日志配置和管理功能
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
import pandas as pd
from pathlib import Path
from typing import Any

# ==================== 配置 ====================

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """日志配置"""
    name: str = "crypto_data_system"
    level: LogLevel = LogLevel.INFO
    log_dir: str = "./logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5  # 保留的备份文件数
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # 输出配置
    console_output: bool = True
    file_output: bool = True
    json_output: bool = False  # 是否输出JSON格式日志
    
    # 文件配置
    log_filename: str = "crypto_data.log"
    error_filename: str = "crypto_data_error.log"
    
    # 高级配置
    enable_rotation: bool = True
    rotation_when: str = "midnight"  # midnight, D, H, M
    rotation_interval: int = 1
    encoding: str = "utf-8"
    
    def __post_init__(self):
        """后初始化处理"""
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志文件路径
        self.log_filepath = os.path.join(self.log_dir, self.log_filename)
        self.error_filepath = os.path.join(self.log_dir, self.error_filename)


class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.default_fields = kwargs
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.threadName,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加自定义字段
        if hasattr(record, 'custom_fields'):
            log_data.update(record.custom_fields)
        
        # 添加默认字段
        log_data.update(self.default_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色控制台输出格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[41m',   # 红底白字
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录，添加颜色"""
        # 添加颜色
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        message = super().format(record)
        
        if self.COLORS['RESET'] in message:
            # 如果已经有颜色标记，不重复添加
            return message
        
        return f"{level_color}{message}{self.COLORS['RESET']}"


# ==================== 日志管理器 ====================

class LogManager:
    """日志管理器"""
    
    _instances: Dict[str, 'LogManager'] = {}
    
    def __new__(cls, name: str = "crypto_data_system", config: LogConfig = None):
        """单例模式"""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(self, name: str = "crypto_data_system", config: LogConfig = None):
        if hasattr(self, '_initialized'):
            return
        
        self.name = name
        self.config = config or LogConfig(name=name)
        self.loggers: Dict[str, logging.Logger] = {}
        
        # 初始化根日志记录器
        self._setup_root_logger()
        
        self._initialized = True
    
    def _setup_root_logger(self):
        """设置根日志记录器"""
        # 获取或创建根logger
        root_logger = logging.getLogger(self.name)
        root_logger.setLevel(self.config.level.value)
        
        # 清除现有处理器（避免重复）
        root_logger.handlers.clear()
        
        # 添加处理器
        if self.config.console_output:
            self._add_console_handler(root_logger)
        
        if self.config.file_output:
            self._add_file_handler(root_logger)
        
        if self.config.json_output:
            self._add_json_handler(root_logger)
        
        # 添加错误处理器
        self._add_error_handler(root_logger)
    
    def _add_console_handler(self, logger: logging.Logger):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.config.json_output:
            formatter = JSONFormatter(app_name=self.name)
        else:
            formatter = ColoredFormatter(
                fmt=self.config.format,
                datefmt=self.config.date_format
            )
        
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.config.level.value)
        logger.addHandler(console_handler)
    
    def _add_file_handler(self, logger: logging.Logger):
        """添加文件处理器"""
        if self.config.enable_rotation:
            # 使用RotatingFileHandler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.config.log_filepath,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding=self.config.encoding
            )
        else:
            # 使用普通FileHandler
            file_handler = logging.FileHandler(
                filename=self.config.log_filepath,
                encoding=self.config.encoding
            )
        
        formatter = logging.Formatter(
            fmt=self.config.format,
            datefmt=self.config.date_format
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.config.level.value)
        logger.addHandler(file_handler)
    
    def _add_json_handler(self, logger: logging.Logger):
        """添加JSON处理器"""
        json_filename = self.config.log_filepath.replace('.log', '.json.log')
        
        json_handler = logging.FileHandler(
            filename=json_filename,
            encoding=self.config.encoding
        )
        
        formatter = JSONFormatter(app_name=self.name)
        json_handler.setFormatter(formatter)
        json_handler.setLevel(self.config.level.value)
        logger.addHandler(json_handler)
    
    def _add_error_handler(self, logger: logging.Logger):
        """添加错误处理器（专门记录ERROR及以上级别）"""
        error_handler = logging.FileHandler(
            filename=self.config.error_filepath,
            encoding=self.config.encoding
        )
        
        formatter = logging.Formatter(
            fmt=self.config.format,
            datefmt=self.config.date_format
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """获取指定名称的日志记录器"""
        if not name:
            name = self.name
        
        if name not in self.loggers:
            # 创建子logger
            logger = logging.getLogger(f"{self.name}.{name}")
            logger.setLevel(self.config.level.value)
            
            # 不重复添加处理器，继承根logger的处理器
            logger.propagate = True
            
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def update_config(self, **kwargs):
        """更新日志配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 重新设置日志器
        self._setup_root_logger()
    
    def log_to_file(self, message: str, level: str = "INFO", 
                   filename: str = None, **kwargs):
        """直接记录日志到指定文件"""
        if not filename:
            filename = self.config.log_filepath
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        try:
            with open(filename, 'a', encoding=self.config.encoding) as f:
                if self.config.json_output:
                    f.write(json.dumps(log_entry) + '\n')
                else:
                    log_line = f"{log_entry['timestamp']} - {level} - {message}\n"
                    f.write(log_line)
        except Exception as e:
            print(f"日志写入失败: {e}")
    
    def get_log_files(self) -> Dict[str, List[str]]:
        """获取所有日志文件"""
        log_dir = Path(self.config.log_dir)
        
        if not log_dir.exists():
            return {}
        
        files = {}
        for file_path in log_dir.glob("*.log*"):
            file_name = file_path.name
            file_type = "unknown"
            
            if "error" in file_name:
                file_type = "error"
            elif "json" in file_name:
                file_type = "json"
            else:
                file_type = "standard"
            
            if file_type not in files:
                files[file_type] = []
            
            files[file_type].append(str(file_path))
        
        return files
    
    def clear_logs(self, keep_recent: int = 5):
        """清理日志文件（保留最近的几个）"""
        log_files = self.get_log_files()
        
        for file_type, file_list in log_files.items():
            if len(file_list) > keep_recent:
                # 按修改时间排序，删除旧的
                sorted_files = sorted(file_list, key=os.path.getmtime)
                files_to_delete = sorted_files[:-keep_recent]
                
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        self.get_logger().info(f"删除旧日志文件: {file_path}")
                    except Exception as e:
                        self.get_logger().error(f"删除日志文件失败 {file_path}: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        log_files = self.get_log_files()
        
        stats = {
            'total_files': sum(len(files) for files in log_files.values()),
            'file_types': {k: len(v) for k, v in log_files.items()},
            'log_dir': self.config.log_dir,
            'log_config': {
                'level': self.config.level.name,
                'max_file_size': self.config.max_file_size,
                'backup_count': self.config.backup_count
            }
        }
        
        # 计算总文件大小
        total_size = 0
        for file_list in log_files.values():
            for file_path in file_list:
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        stats['total_size_bytes'] = total_size
        stats['total_size_mb'] = total_size / (1024 * 1024)
        
        return stats


# ==================== 装饰器和上下文管理器 ====================

def log_execution_time(logger: logging.Logger = None, level: str = "INFO"):
    """
    记录函数执行时间的装饰器
    
    参数:
        logger: 日志记录器，如果为None则使用默认logger
        level: 日志级别
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取logger
            if logger is None:
                log = get_logger(func.__module__)
            else:
                log = logger
            
            start_time = time.time()
            
            try:
                log.log(getattr(logging, level.upper()), 
                       f"开始执行: {func.__name__}")
                
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                log.log(getattr(logging, level.upper()), 
                       f"完成执行: {func.__name__}, 耗时: {execution_time:.2f}秒")
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                log.error(f"执行失败: {func.__name__}, 耗时: {execution_time:.2f}秒, 错误: {e}")
                raise
        
        return wrapper
    
    return decorator


def log_errors(logger: logging.Logger = None, level: str = "ERROR", 
              reraise: bool = True):
    """
    捕获并记录异常的装饰器
    
    参数:
        logger: 日志记录器
        level: 日志级别
        reraise: 是否重新抛出异常
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取logger
            if logger is None:
                log = get_logger(func.__module__)
            else:
                log = logger
            
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                log_level = getattr(logging, level.upper())
                
                # 记录异常信息
                error_msg = f"函数 {func.__name__} 执行异常: {str(e)}"
                log.log(log_level, error_msg)
                log.log(log_level, traceback.format_exc())
                
                if reraise:
                    raise
                else:
                    return None
        
        return wrapper
    
    return decorator


class LoggingContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, level: int = logging.INFO, 
                 message: str = "", **kwargs):
        self.logger = logger
        self.level = level
        self.message = message
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        
        if self.message:
            self.logger.log(self.level, f"开始: {self.message}", extra=self.kwargs)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            # 没有异常
            if self.message:
                self.logger.log(self.level, 
                               f"完成: {self.message}, 耗时: {execution_time:.2f}秒",
                               extra=self.kwargs)
        else:
            # 有异常
            self.logger.error(f"异常: {self.message}, 耗时: {execution_time:.2f}秒, "
                             f"错误: {exc_val}", extra=self.kwargs)


# ==================== 便捷函数 ====================

# 全局日志管理器实例
_default_log_manager: Optional[LogManager] = None

def setup_logger(config: LogConfig = None) -> LogManager:
    """设置全局日志记录器"""
    global _default_log_manager
    
    if _default_log_manager is None:
        _default_log_manager = LogManager(config=config)
    
    return _default_log_manager

def get_logger(name: str = None) -> logging.Logger:
    """获取日志记录器"""
    global _default_log_manager
    
    if _default_log_manager is None:
        # 使用默认配置初始化
        _default_log_manager = setup_logger()
    
    return _default_log_manager.get_logger(name)

def log_with_fields(message: str, level: str = "INFO", 
                   logger_name: str = None, **fields):
    """记录带自定义字段的日志"""
    logger = get_logger(logger_name)
    
    # 添加自定义字段
    extra = {'custom_fields': fields}
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message, extra=extra)

def setup_file_logging(filename: str, level: str = "INFO", 
                      format_str: str = None, max_size: int = 10485760):
    """快速设置文件日志记录"""
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger()
    
    # 创建文件处理器
    handler = RotatingFileHandler(
        filename=filename,
        maxBytes=max_size,
        backupCount=5,
        encoding='utf-8'
    )
    
    # 设置格式
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    
    # 设置级别
    handler.setLevel(getattr(logging, level.upper()))
    
    # 添加到logger
    logger.addHandler(handler)
    
    return handler

def log_dataframe_info(df: Any, logger: logging.Logger = None, 
                      df_name: str = "DataFrame"):
    """记录DataFrame的基本信息"""
    if logger is None:
        logger = get_logger("dataframe")
    
    logger.info(f"{df_name} 基本信息:")
    logger.info(f"  形状: {df.shape}")
    logger.info(f"  列: {list(df.columns)}")
    logger.info(f"  索引类型: {type(df.index).__name__}")
    
    if len(df) > 0:
        logger.info(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
    
    # 数据类型信息
    dtypes = df.dtypes.value_counts().to_dict()
    logger.info(f"  数据类型分布: {dtypes}")
    
    # 缺失值信息
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"  缺失值总计: {missing.sum()}")
        for col, count in missing[missing > 0].items():
            logger.warning(f"    {col}: {count} 个缺失值")

def log_dict_as_table(data: Dict, title: str = "", logger: logging.Logger = None,
                     level: str = "INFO"):
    """以表格形式记录字典数据"""
    if logger is None:
        logger = get_logger()
    
    if not data:
        return
    
    log_func = getattr(logger, level.lower(), logger.info)
    
    if title:
        log_func(title)
    
    # 计算列宽
    max_key_len = max(len(str(k)) for k in data.keys())
    max_val_len = max(len(str(v)) for v in data.values())
    
    # 表头
    header = f"| {'键':^{max_key_len}} | {'值':^{max_val_len}} |"
    separator = f"+{'-' * (max_key_len + 2)}+{'-' * (max_val_len + 2)}+"
    
    log_func(separator)
    log_func(header)
    log_func(separator)
    
    # 数据行
    for key, value in data.items():
        row = f"| {str(key):<{max_key_len}} | {str(value):<{max_val_len}} |"
        log_func(row)
    
    log_func(separator)

# ==================== 测试函数 ====================

def test_logger():
    """测试日志功能"""
    print("=" * 60)
    print("日志模块测试")
    print("=" * 60)
    
    # 创建测试日志配置
    config = LogConfig(
        name="test_logger",
        level=LogLevel.DEBUG,
        log_dir="./test_logs",
        console_output=True,
        file_output=True,
        json_output=False
    )
    
    # 设置日志器
    setup_logger(config)
    
    # 获取日志器
    logger = get_logger("test")
    
    # 测试不同级别的日志
    print("\n1. 测试日志级别:")
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")
    
    try:
        raise ValueError("测试异常")
    except ValueError as e:
        logger.exception("捕获到异常: %s", e)
    
    # 测试带字段的日志
    print("\n2. 测试带字段的日志:")
    log_with_fields(
        "带字段的日志",
        level="INFO",
        user_id=123,
        action="login",
        ip="192.168.1.1"
    )
    
    # 测试执行时间装饰器
    print("\n3. 测试执行时间装饰器:")
    
    @log_execution_time(level="INFO")
    def slow_function():
        time.sleep(0.1)
        return "完成"
    
    result = slow_function()
    print(f"函数结果: {result}")
    
    # 测试错误装饰器
    print("\n4. 测试错误装饰器:")
    
    @log_errors(reraise=False)
    def error_function():
        raise RuntimeError("测试错误")
    
    error_function()
    print("错误已被捕获和记录")
    
    # 测试上下文管理器
    print("\n5. 测试上下文管理器:")
    with LoggingContext(logger, logging.INFO, "测试上下文"):
        time.sleep(0.05)
        print("在上下文中执行操作")
    
    # 测试DataFrame日志
    print("\n6. 测试DataFrame日志:")
    try:
        import pandas as pd
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [5.1, 6.2, 7.3, 8.4],
            'c': ['x', 'y', 'z', None]
        })
        log_dataframe_info(df, df_name="测试DataFrame")
    except ImportError:
        print("pandas未安装，跳过DataFrame测试")
    
    # 获取日志统计
    print("\n7. 获取日志统计:")
    log_manager = setup_logger()
    stats = log_manager.get_log_stats()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # 清理测试日志
    print("\n8. 清理测试日志...")
    import shutil
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")
    
    print("\n✅ 日志模块测试完成")


# ==================== 必要的导入 ====================

import time
from functools import wraps

try:
    import pandas as pd
except ImportError:
    pd = None

# 初始化默认日志管理器
setup_logger()

if __name__ == "__main__":
    test_logger()