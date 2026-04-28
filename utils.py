"""
工具函数：日志、路径、可选依赖探测、简单持久化（预留）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional


def now_str() -> str:
    """获取当前时间字符串（用于日志）"""

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在"""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_import(module_name: str) -> Optional[Any]:
    """
    安全导入可选依赖：
    - 成功返回模块对象
    - 失败返回 None
    """

    try:
        return __import__(module_name)
    except Exception:
        return None


def write_json(path: str | Path, data: Any) -> None:
    """写JSON（预留：项目管理/历史项目）"""

    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path, default: Any = None) -> Any:
    """读JSON（预留）"""

    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_dict_dataclass(obj: Any) -> dict:
    """dataclass 转 dict"""

    try:
        return asdict(obj)
    except Exception:
        return dict(obj)


def open_in_explorer(path: str | Path) -> None:
    """在 Windows 资源管理器打开目录"""

    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    os.startfile(str(p))  # noqa: S606,S607（Windows专用）

