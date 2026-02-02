"""
缓存管理模块
提供数据缓存功能，支持内存缓存和文件缓存
"""

import os
import json
import pickle
import hashlib
import time
import gzip
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
from functools import wraps

# ==================== 缓存配置 ====================

@dataclass
class CacheConfig:
    """缓存配置"""
    cache_dir: str = "./data/cache"
    memory_cache_size: int = 1000  # 内存缓存最大条目数
    default_ttl: int = 3600  # 默认缓存时间（秒）
    compress: bool = True  # 是否压缩存储
    compress_level: int = 3  # 压缩级别（1-9）
    file_format: str = "pickle"  # pickle, json, parquet
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    auto_cleanup: bool = True
    cleanup_interval: int = 3600  # 清理间隔（秒）
    
    def get_cache_file_extension(self) -> str:
        """获取缓存文件扩展名"""
        extensions = {
            'pickle': '.pkl',
            'json': '.json',
            'parquet': '.parquet',
            'feather': '.feather'
        }
        return extensions.get(self.file_format, '.pkl')
    
    def get_compressed_extension(self) -> str:
        """获取压缩文件扩展名"""
        return '.gz' if self.compress else ''


class CacheStrategy(Enum):
    """缓存策略"""
    MEMORY_ONLY = "memory_only"      # 仅内存缓存
    DISK_ONLY = "disk_only"          # 仅磁盘缓存
    MEMORY_AND_DISK = "memory_disk"  # 内存+磁盘
    LRU = "lru"                      # LRU缓存


class CacheEntry:
    """缓存条目"""
    
    def __init__(self, key: str, data: Any, ttl: int = None, 
                 created_at: datetime = None):
        self.key = key
        self.data = data
        self.ttl = ttl or 3600
        self.created_at = created_at or datetime.now()
        self.accessed_at = self.created_at
        self.access_count = 0
        self.size = self._calculate_size(data)
    
    def _calculate_size(self, data: Any) -> int:
        """估算数据大小（字节）"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame大小估算
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, pd.Series):
                return data.memory_usage(deep=True)
            elif isinstance(data, (dict, list)):
                return len(str(data).encode('utf-8'))
            elif isinstance(data, (int, float)):
                return 8
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            else:
                return 100  # 默认值
        except:
            return 100
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl
    
    def remaining_time(self) -> float:
        """剩余时间（秒）"""
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return max(0, self.ttl - elapsed)
    
    def access(self) -> None:
        """访问条目"""
        self.accessed_at = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'key': self.key,
            'ttl': self.ttl,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'size': self.size
        }


# ==================== 缓存管理器 ====================

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = self.config.cache_dir
        
        # 创建缓存目录
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        # 清理子目录
        self.sub_dirs = {
            'spot': os.path.join(self.disk_cache_dir, 'spot'),
            'swap': os.path.join(self.disk_cache_dir, 'swap'),
            'margin': os.path.join(self.disk_cache_dir, 'margin'),
            'future': os.path.join(self.disk_cache_dir, 'future'),
            'option': os.path.join(self.disk_cache_dir, 'option'),
            'onchain': os.path.join(self.disk_cache_dir, 'onchain'),
            'social': os.path.join(self.disk_cache_dir, 'social')
        }

        # 兼容历史/模块内部使用的子目录别名（统一落到标准市场目录）
        # 例如：option_fetcher 曾用 binance_option；onchain_fetcher 曾用 ethereum
        self.sub_dir_aliases = {
            'binance_option': 'option',
            'options': 'option',
            'ethereum': 'onchain',
            'eth': 'onchain',
        }
        
        for sub_dir in self.sub_dirs.values():
            os.makedirs(sub_dir, exist_ok=True)
        
        # 清理线程（如果需要）
        if self.config.auto_cleanup:
            self.last_cleanup = datetime.now()
    
    # ========== 缓存键生成 ==========
    
    def generate_key(self, prefix: str, params: Dict = None, 
                     data_type: str = None) -> str:
        """生成缓存键"""
        key_parts = [prefix]
        
        if data_type:
            key_parts.append(data_type)
        
        if params:
            # 对参数字典进行排序，确保相同参数生成相同key
            sorted_params = json.dumps(params, sort_keys=True, default=str)
            param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:12]
            key_parts.append(param_hash)
        
        # 清理非法字符
        key = "_".join(key_parts)
        key = "".join(c for c in key if c.isalnum() or c in ('_', '-', '.'))
        
        return key
    
    def get_cache_filepath(self, key: str, sub_dir: str = None) -> str:
        """获取缓存文件路径"""
        cache_dir = self.disk_cache_dir
        if sub_dir:
            # 归一化
            sub_dir_norm = str(sub_dir).strip().lower().replace('\\', '/')
            sub_dir_norm = self.sub_dir_aliases.get(sub_dir_norm, sub_dir_norm)

            # 优先走预定义市场目录
            if sub_dir_norm in self.sub_dirs:
                cache_dir = self.sub_dirs[sub_dir_norm]
            else:
                # 支持自定义/嵌套子目录（不会再把文件散落在根目录）
                parts = [p for p in sub_dir_norm.split('/') if p]
                if parts:
                    cache_dir = os.path.join(self.disk_cache_dir, *parts)
                    os.makedirs(cache_dir, exist_ok=True)
        
        filename = key + self.config.get_cache_file_extension()
        if self.config.compress:
            filename += self.config.get_compressed_extension()
        
        return os.path.join(cache_dir, filename)
    
    # ========== 内存缓存操作 ==========
    
    def memory_set(self, key: str, data: Any, ttl: int = None) -> None:
        """设置内存缓存"""
        if not self.config.enable_memory_cache:
            return
        
        # 检查缓存大小，如果超过限制，清理最久未使用的
        if len(self.memory_cache) >= self.config.memory_cache_size:
            self._cleanup_memory_cache()
        
        entry = CacheEntry(key, data, ttl)
        self.memory_cache[key] = entry
    
    def memory_get(self, key: str) -> Optional[Any]:
        """获取内存缓存"""
        if not self.config.enable_memory_cache:
            return None
        
        entry = self.memory_cache.get(key)
        
        if entry:
            if entry.is_expired():
                # 过期，删除
                del self.memory_cache[key]
                return None
            
            entry.access()
            return entry.data
        
        return None
    
    def memory_delete(self, key: str) -> bool:
        """删除内存缓存"""
        if key in self.memory_cache:
            del self.memory_cache[key]
            return True
        return False
    
    def _cleanup_memory_cache(self) -> None:
        """清理内存缓存（LRU策略）"""
        if not self.memory_cache:
            return
        
        # 按访问时间排序，删除最久未访问的
        entries = sorted(
            self.memory_cache.values(),
            key=lambda x: x.accessed_at
        )
        
        # 删除前20%的最久未访问条目
        delete_count = max(1, len(entries) // 5)
        
        for entry in entries[:delete_count]:
            if entry.key in self.memory_cache:
                del self.memory_cache[entry.key]
    
    # ========== 磁盘缓存操作 ==========
    
    def disk_set(self, key: str, data: Any, sub_dir: str = None) -> bool:
        """设置磁盘缓存"""
        if not self.config.enable_disk_cache:
            return False
        
        filepath = self.get_cache_filepath(key, sub_dir)
        
        try:
            # 根据格式保存数据
            if self.config.file_format == 'pickle':
                cache_data = {
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                    'ttl': self.config.default_ttl
                }
                
                data_bytes = pickle.dumps(cache_data)
                
                if self.config.compress:
                    data_bytes = gzip.compress(data_bytes, compresslevel=self.config.compress_level)
                
                with open(filepath, 'wb') as f:
                    f.write(data_bytes)
            
            elif self.config.file_format == 'json':
                cache_data = {
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                    'ttl': self.config.default_ttl
                }
                
                # JSON只支持可序列化的数据
                json_data = json.dumps(cache_data, default=self._json_default)
                
                if self.config.compress:
                    json_data = gzip.compress(json_data.encode('utf-8'), 
                                             compresslevel=self.config.compress_level)
                    with open(filepath, 'wb') as f:
                        f.write(json_data)
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(json_data)
            
            elif self.config.file_format == 'parquet':
                # 只支持DataFrame
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(filepath, compression='snappy' if self.config.compress else None)
                else:
                    raise ValueError("Parquet格式只支持DataFrame")
            
            return True
            
        except Exception as e:
            print(f"磁盘缓存保存失败: {e}")
            # 删除可能损坏的文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def _json_default(self, obj):
        """JSON序列化的默认处理函数"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"无法序列化类型: {type(obj)}")
    
    def disk_get(self, key: str, sub_dir: str = None) -> Optional[Any]:
        """获取磁盘缓存"""
        if not self.config.enable_disk_cache:
            return None
        
        filepath = self.get_cache_filepath(key, sub_dir)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            # 检查文件年龄
            file_age = time.time() - os.path.getmtime(filepath)
            if file_age > self.config.default_ttl:
                os.remove(filepath)  # 删除过期文件
                return None
            
            # 根据格式读取数据
            if self.config.file_format == 'pickle':
                with open(filepath, 'rb') as f:
                    data_bytes = f.read()
                
                if self.config.compress:
                    data_bytes = gzip.decompress(data_bytes)
                
                cache_data = pickle.loads(data_bytes)
                
                # 检查缓存是否过期
                cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
                cache_age = (datetime.now() - cache_timestamp).total_seconds()
                
                if cache_age > cache_data.get('ttl', self.config.default_ttl):
                    os.remove(filepath)
                    return None
                
                return cache_data['data']
            
            elif self.config.file_format == 'json':
                if self.config.compress:
                    with open(filepath, 'rb') as f:
                        json_bytes = f.read()
                    json_str = gzip.decompress(json_bytes).decode('utf-8')
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json_str = f.read()
                
                cache_data = json.loads(json_str)
                
                # 检查缓存是否过期
                cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
                cache_age = (datetime.now() - cache_timestamp).total_seconds()
                
                if cache_age > cache_data.get('ttl', self.config.default_ttl):
                    os.remove(filepath)
                    return None
                
                # JSON数据可能需要额外处理
                return cache_data['data']
            
            elif self.config.file_format == 'parquet':
                return pd.read_parquet(filepath)
            
            return None
            
        except Exception as e:
            print(f"磁盘缓存读取失败: {e}")
            # 删除可能损坏的文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return None
    
    def disk_delete(self, key: str, sub_dir: str = None) -> bool:
        """删除磁盘缓存"""
        filepath = self.get_cache_filepath(key, sub_dir)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    # ========== 统一缓存操作 ==========
    
    def set(self, key: str, data: Any, ttl: int = None, 
            sub_dir: str = None, strategy: CacheStrategy = CacheStrategy.MEMORY_AND_DISK) -> None:
        """设置缓存"""
        
        # 内存缓存
        if strategy in [CacheStrategy.MEMORY_ONLY, CacheStrategy.MEMORY_AND_DISK, CacheStrategy.LRU]:
            self.memory_set(key, data, ttl)
        
        # 磁盘缓存
        if strategy in [CacheStrategy.DISK_ONLY, CacheStrategy.MEMORY_AND_DISK]:
            self.disk_set(key, data, sub_dir)
    
    def get(self, key: str, sub_dir: str = None, 
            strategy: CacheStrategy = CacheStrategy.MEMORY_AND_DISK) -> Optional[Any]:
        """获取缓存"""
        
        # 先尝试内存缓存
        if strategy in [CacheStrategy.MEMORY_ONLY, CacheStrategy.MEMORY_AND_DISK, CacheStrategy.LRU]:
            memory_data = self.memory_get(key)
            if memory_data is not None:
                return memory_data
        
        # 再尝试磁盘缓存
        if strategy in [CacheStrategy.DISK_ONLY, CacheStrategy.MEMORY_AND_DISK]:
            disk_data = self.disk_get(key, sub_dir)
            if disk_data is not None:
                # 如果启用了内存缓存，将磁盘数据加载到内存
                if strategy == CacheStrategy.MEMORY_AND_DISK and self.config.enable_memory_cache:
                    self.memory_set(key, disk_data)
                return disk_data
        
        return None
    
    def delete(self, key: str, sub_dir: str = None) -> bool:
        """删除缓存"""
        memory_deleted = self.memory_delete(key)
        disk_deleted = self.disk_delete(key, sub_dir)
        return memory_deleted or disk_deleted
    
    def clear(self, sub_dir: str = None, clear_memory: bool = True, 
              clear_disk: bool = True) -> None:
        """清除缓存"""
        if clear_memory:
            self.memory_cache.clear()
        
        if clear_disk and sub_dir:
            cache_dir = self.sub_dirs.get(sub_dir, self.disk_cache_dir)
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    filepath = os.path.join(cache_dir, filename)
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        print(f"删除文件失败 {filepath}: {e}")
    
    # ========== 缓存统计和管理 ==========
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        memory_stats = {
            'entries': len(self.memory_cache),
            'total_size': sum(entry.size for entry in self.memory_cache.values()),
            'avg_size': np.mean([entry.size for entry in self.memory_cache.values()]) if self.memory_cache else 0
        }
        
        disk_stats = {}
        for sub_dir_name, sub_dir_path in self.sub_dirs.items():
            if os.path.exists(sub_dir_path):
                files = os.listdir(sub_dir_path)
                disk_stats[sub_dir_name] = {
                    'files': len(files),
                    'total_size': sum(os.path.getsize(os.path.join(sub_dir_path, f)) for f in files)
                }
        
        return {
            'memory_cache': memory_stats,
            'disk_cache': disk_stats,
            'config': asdict(self.config)
        }
    
    def cleanup_expired(self) -> Dict[str, int]:
        """清理过期缓存"""
        cleaned_counts = {
            'memory': 0,
            'disk': 0
        }
        
        # 清理内存缓存
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            cleaned_counts['memory'] += 1
        
        # 清理磁盘缓存
        for sub_dir_name, sub_dir_path in self.sub_dirs.items():
            if os.path.exists(sub_dir_path):
                for filename in os.listdir(sub_dir_path):
                    filepath = os.path.join(sub_dir_path, filename)
                    
                    # 检查文件年龄
                    file_age = time.time() - os.path.getmtime(filepath)
                    if file_age > self.config.default_ttl:
                        try:
                            os.remove(filepath)
                            cleaned_counts['disk'] += 1
                        except Exception as e:
                            print(f"删除过期文件失败 {filepath}: {e}")
        
        self.last_cleanup = datetime.now()
        return cleaned_counts
    
    def get_cache_info(self, key: str) -> Optional[Dict]:
        """获取缓存条目信息"""
        entry = self.memory_cache.get(key)
        if entry:
            info = entry.to_dict()
            info['location'] = 'memory'
            return info
        
        # 检查磁盘
        for sub_dir_name, sub_dir_path in self.sub_dirs.items():
            filepath = self.get_cache_filepath(key, sub_dir_name)
            if os.path.exists(filepath):
                file_stats = os.stat(filepath)
                return {
                    'key': key,
                    'location': 'disk',
                    'sub_dir': sub_dir_name,
                    'filepath': filepath,
                    'size': file_stats.st_size,
                    'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                }
        
        return None

# ==================== 装饰器 ====================

def cached(cache_manager: CacheManager = None, ttl: int = 3600, 
           key_prefix: str = "", sub_dir: str = None,
           strategy: CacheStrategy = CacheStrategy.MEMORY_AND_DISK):
    """
    缓存装饰器
    
    参数:
        cache_manager: 缓存管理器实例
        ttl: 缓存时间（秒）
        key_prefix: 缓存键前缀
        sub_dir: 子目录
        strategy: 缓存策略
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 使用提供的缓存管理器或创建默认的
            cm = cache_manager or CacheManager()
            
            # 生成缓存键
            # 基于函数名、参数等生成
            cache_key_parts = [key_prefix or func.__name__]
            
            # 添加参数信息
            if args:
                cache_key_parts.append(str(hash(str(args))))
            if kwargs:
                sorted_kwargs = json.dumps(kwargs, sort_keys=True, default=str)
                cache_key_parts.append(hashlib.md5(sorted_kwargs.encode()).hexdigest()[:8])
            
            cache_key = "_".join(cache_key_parts)
            
            # 尝试从缓存获取
            cached_result = cm.get(cache_key, sub_dir=sub_dir, strategy=strategy)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            if result is not None:
                cm.set(cache_key, result, ttl=ttl, sub_dir=sub_dir, strategy=strategy)
            
            return result
        
        return wrapper
    
    return decorator


def cache_result(cache_key: str = None, ttl: int = 3600, 
                 sub_dir: str = None, strategy: CacheStrategy = CacheStrategy.MEMORY_AND_DISK):
    """
    缓存结果的装饰器（指定缓存键）
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 检查是否有cache_manager属性
            if hasattr(self, 'cache_manager'):
                cm = self.cache_manager
            else:
                cm = CacheManager()
            
            # 确定缓存键
            if cache_key:
                key = cache_key
            elif hasattr(self, 'generate_cache_key'):
                key = self.generate_cache_key(func.__name__, *args, **kwargs)
            else:
                # 默认生成方式
                key_parts = [func.__module__, func.__name__]
                if args:
                    key_parts.append(str(args))
                if kwargs:
                    key_parts.append(str(sorted(kwargs.items())))
                key = hashlib.md5("_".join(key_parts).encode()).hexdigest()[:16]
            
            # 尝试从缓存获取
            cached_result = cm.get(key, sub_dir=sub_dir, strategy=strategy)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(self, *args, **kwargs)
            
            # 缓存结果
            if result is not None:
                cm.set(key, result, ttl=ttl, sub_dir=sub_dir, strategy=strategy)
            
            return result
        
        return wrapper
    
    return decorator


# ==================== 测试函数 ====================

def test_cache():
    """测试缓存功能"""
    print("=" * 60)
    print("缓存模块测试")
    print("=" * 60)
    
    # 创建缓存管理器
    config = CacheConfig(
        cache_dir="./test_cache",
        memory_cache_size=10,
        default_ttl=60,  # 60秒
        enable_memory_cache=True,
        enable_disk_cache=True
    )
    
    cache_manager = CacheManager(config)
    
    # 测试内存缓存
    print("\n1. 测试内存缓存:")
    cache_manager.memory_set("test_key1", {"data": "test_value"}, ttl=30)
    result = cache_manager.memory_get("test_key1")
    print(f"内存缓存获取: {result}")
    
    # 测试磁盘缓存
    print("\n2. 测试磁盘缓存:")
    test_data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })
    cache_manager.disk_set("test_df", test_data, sub_dir="spot")
    loaded_data = cache_manager.disk_get("test_df", sub_dir="spot")
    print(f"磁盘缓存获取DataFrame: 形状 {loaded_data.shape if loaded_data is not None else 'None'}")
    
    # 测试统一接口
    print("\n3. 测试统一接口:")
    cache_manager.set("unified_key", {"unified": "data"}, ttl=60, sub_dir="swap")
    unified_result = cache_manager.get("unified_key", sub_dir="swap")
    print(f"统一接口获取: {unified_result}")
    
    # 测试装饰器
    print("\n4. 测试缓存装饰器:")
    
    @cached(cache_manager=cache_manager, ttl=10, key_prefix="test_func")
    def expensive_function(x, y):
        print(f"执行expensive_function({x}, {y})")
        return x * y
    
    # 第一次调用应该执行函数
    result1 = expensive_function(3, 4)
    print(f"第一次调用结果: {result1}")
    
    # 第二次调用应该从缓存获取
    result2 = expensive_function(3, 4)
    print(f"第二次调用结果: {result2}")
    
    # 测试缓存统计
    print("\n5. 测试缓存统计:")
    stats = cache_manager.get_stats()
    print(f"内存缓存条目数: {stats['memory_cache']['entries']}")
    print(f"磁盘缓存文件数: {len(stats['disk_cache'])}")
    
    # 清理测试缓存
    print("\n6. 清理测试缓存...")
    import shutil
    if os.path.exists("./test_cache"):
        shutil.rmtree("./test_cache")
    
    print("✅ 缓存模块测试完成")


if __name__ == "__main__":
    test_cache()