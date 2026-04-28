"""
数据模型（轻量级）

说明：
- V1.0 采用 pandas DataFrame + Python dataclass 组织结果
- 后续可升级 sqlite 持久化（已在 utils.py 中预留接口）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DetectParams:
    """识别参数（与GUI参数区对应）"""

    scale_denominator: int = 100  # 比例尺 1:100 的分母
    dpi: int = 300  # 图像/图纸采样DPI（用于像素-米换算）
    area_unit: str = "㎡"
    house_category: str = "住宅"  # 住宅/商铺/厂房
    output_formats: Tuple[str, ...] = ("Excel", "PDF", "PNG")
    auto_number: bool = True
    auto_ocr: bool = True

    # OpenCV识别阈值（可后续在GUI中扩展高级参数）
    min_contour_area_px: int = 800  # 过滤过小噪声轮廓
    approx_epsilon_ratio: float = 0.01  # 多边形逼近强度

    # 识别增强参数（V2.0）
    use_canny: bool = True
    canny_th1: int = 50
    canny_th2: int = 150
    morph_kernel: int = 3  # 形态学核大小
    morph_close_iter: int = 2
    morph_open_iter: int = 1
    enhance_contrast: bool = True
    sharpen: bool = True
    denoise: bool = True
    binarize_mode: str = "adaptive"  # adaptive/otsu

    # 轮廓筛选（V2.0）
    min_bbox_side_px: int = 10
    max_aspect_ratio: float = 15.0  # 过细长的线状目标过滤（如标注线）
    simplify_max_points: int = 80  # 过密点简化上限（用于人工校正更好操作）


@dataclass
class HouseFeature:
    """单个房屋/目标轮廓的识别结果"""

    uid: str  # 系统内部唯一编号（可用uuid或自增）
    number: str  # 房号（自动编号或OCR识别）
    building_no: Optional[str]  # 楼栋号（OCR/属性表/人工）
    contour_area_px: float  # 像素面积
    area_m2: float  # 实际面积（㎡）
    bbox_xywh: Tuple[int, int, int, int]  # 外接矩形 (x,y,w,h)
    centroid_xy: Tuple[float, float]  # 质心像素坐标
    contour_xy: List[Tuple[int, int]] | None = None  # 轮廓点（多边形），用于制图/人工校正
    ocr_text: Optional[str] = None
    source_path: Optional[str] = None

