"""
导出模块（V1.0）

说明：
- Excel/PDF/PNG 导出已在 core.py 中实现主流程
- 本文件用于后续扩展：CAD底图、SHP图层、登记表模板等
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def export_shp(df: pd.DataFrame, out_path: str, logger=None) -> Optional[str]:
    """
    导出SHP（可选GIS扩展）

    当前版本：仅做依赖探测与占位。
    若需要真正导出，需要将轮廓点集保存为 geometry（Polygon），并写入 GeoDataFrame。
    """

    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import Point  # type: ignore
    except Exception:
        if logger:
            logger("未安装 geopandas/shapely，无法导出SHP（可选：pip install geopandas shapely）。")
        return None

    # V1.0：示例使用质心点导出（真实项目建议用轮廓Polygon）
    if "质心x" not in df.columns or "质心y" not in df.columns:
        if logger:
            logger("数据缺少质心坐标字段，无法导出SHP。")
        return None

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(df["质心x"], df["质心y"])],
        crs=None,  # 未知坐标系，后续可让用户选择EPSG
    )

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(p), driver="ESRI Shapefile", encoding="utf-8")

    if logger:
        logger(f"SHP导出完成：{p}")
    return str(p)

