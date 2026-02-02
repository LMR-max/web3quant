"""
通用数据管理器（文件存储实现）

提供一个可替换的 DataManager 基类及文件存储实现 FileDataManager。
可选地与 `utils.cache.CacheManager` 配合使用。
"""
from __future__ import annotations

import os
import json
import pickle
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None


class BaseDataManager(ABC):
    """抽象数据管理器接口"""

    @abstractmethod
    def save(self, key: str, data: Any) -> bool:
        """保存数据"""
        raise NotImplementedError()

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """加载数据"""
        raise NotImplementedError()

    @abstractmethod
    def list_keys(self) -> List[str]:
        """列出所有键"""
        raise NotImplementedError()

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除数据"""
        raise NotImplementedError()

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        raise NotImplementedError()

    def save_dict(self, key: str, data: Dict[str, Any]) -> bool:
        """保存字典数据"""
        return self.save(f"{key}_dict", data)

    def load_dict(self, key: str) -> Optional[Dict[str, Any]]:
        """加载字典数据"""
        data = self.load(f"{key}_dict")
        return data if isinstance(data, dict) else None

    def save_list(self, key: str, data: List[Any]) -> bool:
        """保存列表数据"""
        return self.save(f"{key}_list", data)

    def load_list(self, key: str) -> Optional[List[Any]]:
        """加载列表数据"""
        data = self.load(f"{key}_list")
        return data if isinstance(data, list) else None

    def save_dataframe(self, key: str, df: Any) -> bool:
        """保存 DataFrame 数据"""
        if pd is None:
            return False
        if not isinstance(df, pd.DataFrame):
            return False
        # 转换为 dict 并保存
        return self.save(f"{key}_df", df.to_dict(orient='records'))

    def load_dataframe(self, key: str) -> Optional[Any]:
        """加载 DataFrame 数据"""
        if pd is None:
            return None
        data = self.load(f"{key}_df")
        if isinstance(data, list):
            return pd.DataFrame(data)
        return None

    def save_timestamped(self, key: str, data: Any, timestamp: bool = True) -> bool:
        """保存带时间戳的数据"""
        if timestamp:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.save(f"{key}_{ts}", data)
        return self.save(key, data)

    def delete_all(self, prefix: str = "") -> int:
        """删除指定前缀的所有数据"""
        count = 0
        for key in self.list_keys():
            if not prefix or key.startswith(prefix):
                if self.delete(key):
                    count += 1
        return count


import shutil
import tempfile

class FileDataManager(BaseDataManager):
    """基于文件系统的简单数据管理器。

    存储结构：root_dir / sub_dir / <key>.<ext>
    支持 JSON 序列化（默认）和 pickle 回退。
    使用原子写入防止文件损坏。
    """

    def __init__(self, root_dir: str = None, sub_dir: str = "data", file_format: str = "json", cache_manager: Any = None):
        self.root_dir = Path(root_dir or os.path.join(os.getcwd(), "data_manager_storage"))
        self.sub_dir = sub_dir
        self.file_format = file_format.lower()
        self.cache_manager = cache_manager

        self._dir = self.root_dir / self.sub_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # 清理 key，避免路径穿越
        safe_key = "".join(c for c in key if c.isalnum() or c in ('_', '-', '.'))
        ext = ".json" if self.file_format == "json" else ".pkl"
        return self._dir / (safe_key + ext)

    def save(self, key: str, data: Any) -> bool:
        path = self._path_for(key)
        
        # 使用临时文件进行原子写入
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self._dir, text=(self.file_format == "json"))
        try:
            with os.fdopen(tmp_fd, 'w' if self.file_format == "json" else 'wb') as f:
                if self.file_format == "json":
                    json.dump(data, f, default=self._json_default)
                else:
                    pickle.dump(data, f)
            
            # 原子移动
            shutil.move(tmp_path, path)

            # 更新 cache manager（如果有）
            if self.cache_manager and hasattr(self.cache_manager, 'memory_set'):
                try:
                    self.cache_manager.memory_set(key, data)
                except Exception:
                    pass

            return True

        except Exception as e:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
            # 回退：如果默认格式为 json，尝试 pickle 保存 (非原子，但在异常路径下允许)
            try:
                if self.file_format == "json":
                    with (path.with_suffix('.pkl')).open('wb') as f:
                        pickle.dump(data, f)
                    return True
            except Exception:
                pass
            return False

    def load(self, key: str) -> Optional[Any]:
        path = self._path_for(key)

        # 先检查内存缓存
        if self.cache_manager and hasattr(self.cache_manager, 'memory_get'):
            try:
                cached = self.cache_manager.memory_get(key)
                if cached is not None:
                    return cached
            except Exception:
                pass

        if path.exists():
            try:
                if path.suffix == '.json':
                    with path.open('r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    with path.open('rb') as f:
                        return pickle.load(f)
            except Exception:
                return None

        # 尝试 pickle 后缀的文件
        pkl = path.with_suffix('.pkl')
        if pkl.exists():
            try:
                with pkl.open('rb') as f:
                    return pickle.load(f)
            except Exception:
                return None

        return None

    def list_keys(self) -> List[str]:
        keys: List[str] = []
        for p in self._dir.iterdir():
            if p.is_file():
                name = p.stem
                keys.append(name)
        return keys

    def delete(self, key: str) -> bool:
        path = self._path_for(key)
        removed = False
        if path.exists():
            path.unlink()
            removed = True

        pkl = path.with_suffix('.pkl')
        if pkl.exists():
            pkl.unlink()
            removed = True

        if removed and self.cache_manager and hasattr(self.cache_manager, 'memory_delete'):
            try:
                self.cache_manager.memory_delete(key)
            except Exception:
                pass

        return removed

    def exists(self, key: str) -> bool:
        path = self._path_for(key)
        if path.exists():
            return True
        if path.with_suffix('.pkl').exists():
            return True
        return False

    @staticmethod
    def _json_default(obj):
        # 简单的 JSON 序列化回退
        try:
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            if pd and isinstance(obj, (pd.Timestamp, pd.Series, pd.DataFrame)):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                else:
                    return obj.isoformat()
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)
        except Exception:
            return str(obj)

    def save_csv(self, key: str, data: List[Dict[str, Any]]) -> bool:
        """保存为 CSV 文件（需要列表包含字典）"""
        if not data or not isinstance(data, list):
            return False
        
        path = self._path_for(key).with_suffix('.csv')
        try:
            fieldnames = set()
            for row in data:
                if isinstance(row, dict):
                    fieldnames.update(row.keys())
            
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                writer.writeheader()
                writer.writerows(data)
            return True
        except Exception:
            return False

    def load_csv(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """从 CSV 文件加载数据"""
        path = self._path_for(key).with_suffix('.csv')
        if not path.exists():
            return None
        
        try:
            with path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception:
            return None


__all__ = ["BaseDataManager", "FileDataManager"]
