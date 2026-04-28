"""
案例展示模块（V2.0）

目标：
- 主界面“案例示例”按钮弹窗展示
- 展示：航拍影像、扫描图纸、CAD平面图、Excel属性表、最终输出图纸
- 每个案例：原图、识别结果图、说明文字

说明：
- 为避免仓库内放大体积图片，V2.0 默认在首次运行时自动生成“示例占位图”
- 你可以把真实案例图片放到 assets/cases/ 下覆盖同名文件即可
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class CaseItem:
    title: str
    desc: str
    orig_path: str
    result_path: str


def ensure_case_assets(base_dir: str) -> List[CaseItem]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    cases = [
        ("航拍影像识别案例", "支持航拍/倾斜摄影截图等影像，提取建筑外框与单体目标。", "case_aerial_orig.png", "case_aerial_result.png"),
        ("房产扫描图纸案例", "支持扫描图纸/平面图，增强预处理后做轮廓提取。", "case_scan_orig.png", "case_scan_result.png"),
        ("CAD平面图案例", "支持DXF图纸，提取线段/闭合多段线形成房间边界（需ezdxf）。", "case_cad_orig.png", "case_cad_result.png"),
        ("Excel属性表案例", "支持导入房号、楼栋号、业主、用途、层数等字段自动匹配。", "case_excel.png", "case_excel.png"),
        ("最终输出图纸案例", "输出 PNG/PDF 图纸与 Excel 台账、异常、汇总报表。", "case_output.png", "case_output.png"),
    ]

    out: List[CaseItem] = []
    for title, desc, orig, res in cases:
        orig_p = base / orig
        res_p = base / res
        if not orig_p.exists():
            _gen_placeholder_image(orig_p, title + "\n原图示例", theme="orig")
        if not res_p.exists():
            _gen_placeholder_image(res_p, title + "\n识别结果示例", theme="result")
        out.append(CaseItem(title=title, desc=desc, orig_path=str(orig_p), result_path=str(res_p)))
    return out


def _gen_placeholder_image(path: Path, text: str, theme: str = "orig") -> None:
    w, h = 960, 540
    if theme == "result":
        bg = (245, 250, 255)
        accent = (0, 120, 215)
    else:
        bg = (255, 255, 255)
        accent = (50, 50, 50)

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    # 边框
    draw.rectangle([20, 20, w - 20, h - 20], outline=accent, width=4)
    # 简单几何元素
    draw.rectangle([140, 160, 380, 360], outline=(0, 180, 90), width=5)
    draw.rectangle([420, 140, 800, 320], outline=(255, 140, 0), width=5)
    draw.ellipse([520, 340, 620, 440], outline=(200, 0, 0), width=5)

    # 文本
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", 28)
    except Exception:
        font = ImageFont.load_default()

    draw.text((60, 60), "案例示例（占位图）", fill=accent, font=font)
    draw.text((60, 110), text, fill=accent, font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))

