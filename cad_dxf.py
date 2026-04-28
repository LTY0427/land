"""
DXF/CAD 图纸识别（V2.0）

能力（依赖 ezdxf，可选安装）：
- 读取墙体线段/多段线（LINE/LWPOLYLINE/POLYLINE）
- 过滤标注/尺寸线（DIMENSION、TEXT/MTEXT 等）
- 尝试闭合轮廓，输出房间边界候选多边形

说明：
- DXF 解析在工程上很容易因图层规范不同而差异巨大，V2.0 先提供“可用的通用基线”
- 后续可按项目补充：图层白名单、线宽/颜色过滤、块参照 explode、拓扑闭合等
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .models import DetectParams, HouseFeature


LogFn = Callable[[str], None]


@dataclass
class DxfParseResult:
    features: List[HouseFeature]
    extent: Tuple[float, float, float, float]  # minx,miny,maxx,maxy


def load_dxf_detect_rooms(path: str, params: DetectParams, logger: Optional[LogFn] = None) -> DxfParseResult:
    """
    从 DXF 中提取房间边界（候选轮廓）

    输出：
    - HouseFeature.contour_xy 使用“DXF坐标”映射到像素坐标并不合适，
      V2.0 暂以“DXF平面坐标”直接作为 contour_xy 存储（强制转 int），
      后续可增加坐标系/比例尺映射与真正 CAD 底图出图。
    """

    try:
        import ezdxf  # type: ignore
    except Exception as e:
        raise RuntimeError("解析DXF需要安装 ezdxf：pip install ezdxf") from e

    if logger:
        logger(f"开始解析DXF：{path}")

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    # 过滤：只取“几何线框”，排除尺寸标注/文字等
    lines = []
    polylines = []

    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for e in msp:
        t = e.dxftype()

        if t in {"TEXT", "MTEXT", "DIMENSION", "LEADER", "MLEADER"}:
            continue

        if t == "LINE":
            p1 = e.dxf.start
            p2 = e.dxf.end
            lines.append(((float(p1.x), float(p1.y)), (float(p2.x), float(p2.y))))
            minx, miny, maxx, maxy = _update_extent(minx, miny, maxx, maxy, [p1, p2])
            continue

        if t in {"LWPOLYLINE", "POLYLINE"}:
            pts = [(float(x), float(y)) for x, y, *_ in e.get_points()]  # 兼容 lwpolyline
            if len(pts) >= 2:
                polylines.append(pts)
                minx, miny, maxx, maxy = _update_extent_xy(minx, miny, maxx, maxy, pts)
            continue

    if not np.isfinite(minx):
        minx = miny = 0.0
        maxx = maxy = 1.0

    # 优先使用闭合 polyline 当作房间边界
    rooms: List[List[Tuple[float, float]]] = []
    for pts in polylines:
        if _is_closed(pts):
            rooms.append(_close_ring(pts))

    # 若没有闭合 polyline，则尝试从线段拼接闭合环（简化：阈值吸附）
    if not rooms and lines:
        rooms = _stitch_lines_to_rings(lines, snap_tol=1e-3, max_rings=200)

    features: List[HouseFeature] = []
    idx = 0
    for ring in rooms:
        idx += 1
        uid = str(uuid.uuid4())
        number = f"{idx:03d}" if params.auto_number else ""
        area = abs(_polygon_area(ring))

        # DXF 的“面积单位”未知：如果是图纸坐标（米/毫米），需要项目配置。
        # 这里仍填入 contour_area_px 字段作为“原始面积”，area_m2 暂填 0，避免误导。
        # 后续建议：增加“DXF单位（mm/m）”与“比例尺/换算”配置。
        bbox = _bbox(ring)
        cx, cy = _centroid(ring)
        features.append(
            HouseFeature(
                uid=uid,
                number=number,
                building_no=None,
                contour_area_px=float(area),
                area_m2=0.0,
                bbox_xywh=(int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                centroid_xy=(float(cx), float(cy)),
                contour_xy=[(int(x), int(y)) for x, y in ring],
                ocr_text=None,
                source_path=path,
            )
        )

    if logger:
        logger(f"DXF解析完成：提取房间边界 {len(features)} 处。")

    return DxfParseResult(features=features, extent=(minx, miny, maxx, maxy))


def _update_extent(minx, miny, maxx, maxy, pts) -> Tuple[float, float, float, float]:
    for p in pts:
        minx = min(minx, float(p.x))
        miny = min(miny, float(p.y))
        maxx = max(maxx, float(p.x))
        maxy = max(maxy, float(p.y))
    return minx, miny, maxx, maxy


def _update_extent_xy(minx, miny, maxx, maxy, pts) -> Tuple[float, float, float, float]:
    for x, y in pts:
        minx = min(minx, float(x))
        miny = min(miny, float(y))
        maxx = max(maxx, float(x))
        maxy = max(maxy, float(y))
    return minx, miny, maxx, maxy


def _is_closed(pts: List[Tuple[float, float]]) -> bool:
    if len(pts) < 3:
        return False
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    return (abs(x0 - x1) + abs(y0 - y1)) < 1e-9


def _close_ring(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not pts:
        return pts
    if _is_closed(pts):
        return pts
    return pts + [pts[0]]


def _polygon_area(ring: List[Tuple[float, float]]) -> float:
    if len(ring) < 4:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _bbox(ring: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(ring: List[Tuple[float, float]]) -> Tuple[float, float]:
    if len(ring) < 4:
        return ring[0] if ring else (0.0, 0.0)
    a = _polygon_area(ring)
    if abs(a) < 1e-12:
        # 退化：取均值
        xs = [x for x, _ in ring]
        ys = [y for _, y in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    cx /= (6.0 * a)
    cy /= (6.0 * a)
    return (cx, cy)


def _stitch_lines_to_rings(
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    *,
    snap_tol: float = 1e-3,
    max_rings: int = 200,
) -> List[List[Tuple[float, float]]]:
    """
    简化线段闭合：将端点吸附后按邻接关系追踪环。
    注意：这是“通用基线”，复杂CAD建议做图层/线型过滤与拓扑更稳的闭合算法。
    """

    def snap(p):
        return (round(p[0] / snap_tol) * snap_tol, round(p[1] / snap_tol) * snap_tol)

    # 构建邻接表
    adj = {}
    edges = set()
    for a, b in lines:
        a2 = snap(a)
        b2 = snap(b)
        if a2 == b2:
            continue
        adj.setdefault(a2, []).append(b2)
        adj.setdefault(b2, []).append(a2)
        edges.add((a2, b2) if a2 < b2 else (b2, a2))

    rings = []
    used = set()

    for e in list(edges):
        if e in used:
            continue
        a, b = e
        path = [a, b]
        used.add(e)

        cur = b
        prev = a
        for _ in range(2000):
            nbrs = adj.get(cur, [])
            # 选一个没走过的边
            nxt = None
            for n in nbrs:
                ee = (cur, n) if cur < n else (n, cur)
                if ee in used:
                    continue
                nxt = n
                used.add(ee)
                break
            if nxt is None:
                break
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == path[0] and len(path) >= 4:
                rings.append(path)
                break
        if len(rings) >= max_rings:
            break

    # 只保留面积较大的环（过滤尺寸线/小噪声）
    out = []
    for r in rings:
        r2 = _close_ring(r)
        if abs(_polygon_area(r2)) > 1.0:
            out.append(r2)
    return out

