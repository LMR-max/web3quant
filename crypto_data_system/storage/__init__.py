"""
存储管理模块

提供统一的数据持久化接口和文件管理功能。
支持 JSON、Pickle、CSV 等多种格式存储。
"""

from .data_manager import BaseDataManager, FileDataManager

__all__ = [
    'BaseDataManager',
    'FileDataManager',
]
