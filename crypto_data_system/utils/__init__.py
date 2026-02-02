"""
工具模块包
"""

import os
import sys

# 获取当前文件所在目录（utils目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（utils的父目录）
parent_dir = os.path.dirname(current_dir)

# 添加项目根目录到Python路径，确保模块导入正常
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# ==================== 初始化 __all__ ====================
__all__ = []

# ==================== 导入所有工具模块 ====================

# 缓存管理
try:
    from .cache import CacheManager, CacheConfig, CacheStrategy, cached, cache_result, test_cache
    __all__.extend(['CacheManager', 'CacheConfig', 'CacheStrategy', 'cached', 'cache_result', 'test_cache'])
except ImportError as e:
    print(f"⚠️  utils.cache 模块导入失败: {e}")

# 日期处理工具
try:
    from .date_utils import (
        DateRange, DateTimeUtils, TimeSeriesAnalyzer, split_date_range, format_timestamp, 
        parse_date_string, calculate_timeframe_seconds, get_period_dates,
        get_current_time_string, get_timestamp_ms, get_timestamp_seconds, test_date_utils
    )
    __all__.extend([
        'DateRange', 'DateTimeUtils', 'TimeSeriesAnalyzer', 'split_date_range', 'format_timestamp',
        'parse_date_string', 'calculate_timeframe_seconds', 'get_period_dates',
        'get_current_time_string', 'get_timestamp_ms', 'get_timestamp_seconds', 'test_date_utils'
    ])
except ImportError as e:
    print(f"⚠️  utils.date_utils 模块导入失败: {e}")

# 日志配置
try:
    from .logger import (
        LogManager, LogConfig, LogLevel, 
        setup_logger, get_logger, log_execution_time, log_errors,
        log_with_fields, log_dataframe_info, log_dict_as_table, test_logger
    )
    __all__.extend([
        'LogManager', 'LogConfig', 'LogLevel', 
        'setup_logger', 'get_logger', 'log_execution_time', 'log_errors',
        'log_with_fields', 'log_dataframe_info', 'log_dict_as_table', 'test_logger'
    ])
except ImportError as e:
    print(f"⚠️  utils.logger 模块导入失败: {e}")

# 数据格式化
try:
    from .data_formatter import (
        DataFormatter, CryptoDataFormatter, FormatConfig, ColumnType,
        format_dataframe, normalize_column_names, 
        convert_dtypes, handle_missing_values, test_data_formatter
    )
    __all__.extend([
        'DataFormatter', 'CryptoDataFormatter', 'FormatConfig', 'ColumnType',
        'format_dataframe', 'normalize_column_names',
        'convert_dtypes', 'handle_missing_values', 'test_data_formatter'
    ])
except ImportError as e:
    print(f"⚠️  utils.data_formatter 模块导入失败: {e}")

# ==================== 版本信息 ====================

__version__ = "1.0.0"
__author__ = "Data Tools Team"
__description__ = "数据处理工具模块集合"

# ==================== 工具函数 ====================

def print_available_modules():
    """打印可用的工具模块"""
    print("=" * 60)
    print(f"工具模块 v{__version__}")
    print("=" * 60)
    
    modules_info = [
        ("cache", "缓存管理工具"),
        ("date_utils", "日期时间处理工具"),
        ("logger", "日志配置工具"),
        ("data_formatter", "数据格式化工具"),
    ]
    
    for module_name, description in modules_info:
        if module_name in sys.modules or f"utils.{module_name}" in sys.modules:
            status = "✅ 已加载"
        else:
            status = "❌ 未加载"
        
        print(f"{status} {module_name:<15} - {description}")
    
    if __all__:
        print(f"\n可用的导入: {', '.join(sorted(__all__))}")
    else:
        print(f"\n可用的导入: 无")

def get_module_info(module_name: str):
    """获取指定模块的信息"""
    try:
        module = sys.modules.get(f"utils.{module_name}")
        if module:
            return {
                'name': module.__name__,
                'file': module.__file__,
                'doc': module.__doc__,
                'functions': [name for name in dir(module) if not name.startswith('_')]
            }
    except:
        pass
    return None

def reload_modules():
    """重新加载所有工具模块"""
    import importlib
    
    modules_to_reload = [
        'cache',
        'date_utils', 
        'logger',
        'data_formatter',
    ]
    
    for module_name in modules_to_reload:
        try:
            module = sys.modules.get(f"utils.{module_name}")
            if module:
                importlib.reload(module)
                print(f"✅ 重新加载: {module_name}")
        except Exception as e:
            print(f"⚠️  重新加载 {module_name} 失败: {e}")

# ==================== 便捷导入别名 ====================

# 为常用工具提供更短的别名
try:
    from .cache import CacheManager
    cache_manager = CacheManager if 'CacheManager' in globals() else None
except:
    cache_manager = None

try:
    from .logger import get_logger
    logger = get_logger if 'get_logger' in globals() else None
except:
    logger = None

try:
    from .date_utils import DateRange
    date_range = DateRange if 'DateRange' in globals() else None
except:
    date_range = None

# ==================== 初始化检查 ====================

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
    ]
    
    optional_packages = [
        ('pytz', 'pytz'),
        ('scipy', 'scipy'),
        ('dateutil', 'dateutil'),
    ]
    
    print("检查依赖包...")
    
    missing_required = []
    missing_optional = []
    
    for package, _ in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package, _ in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print("❌ 以下必需依赖包未安装:")
        for package in missing_required:
            print(f"    pip install {package}")
    
    if missing_optional:
        print("⚠️  以下可选依赖包未安装:")
        for package in missing_optional:
            print(f"    pip install {package}")
    
    if not missing_required:
        print("✅ 所有必需依赖包已安装")
    
    return len(missing_required) == 0

# ==================== 测试函数 ====================

def test_all_utils():
    """测试所有工具模块"""
    print("=" * 60)
    print("工具模块测试")
    print("=" * 60)
    
    # 检查依赖
    check_dependencies()
    
    # 打印可用模块
    print_available_modules()
    
    # 测试缓存模块
    if 'CacheManager' in globals():
        try:
            print("\n测试缓存模块...")
            from .cache import test_cache as test_cache_func
            test_cache_func()
        except Exception as e:
            print(f"\n❌ 缓存模块测试失败: {e}")
    else:
        print(f"\n⚠️  缓存模块未导入，跳过测试")
    
    # 测试日期工具模块
    if 'DateRange' in globals():
        try:
            print("\n测试日期工具模块...")
            from .date_utils import test_date_utils as test_date_func
            test_date_func()
        except Exception as e:
            print(f"\n❌ 日期工具模块测试失败: {e}")
    else:
        print(f"\n⚠️  日期工具模块未导入，跳过测试")
    
    print("\n✅ 工具模块测试完成")

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 当直接运行此文件时，运行测试
    test_all_utils()
else:
    # 当作为模块导入时，初始化日志
    try:
        if 'setup_logger' in globals():
            setup_logger()
    except:
        pass