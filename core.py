"""
核心算法模块（V1.0）

包含用户要求的函数拆分：
- load_image()
- detect_house()
- ocr_text()
- calc_area()
- match_property()
- clean_data()
- draw_map()
- export_excel()
- export_pdf()
- write_log()
- main()  # 入口在 main.py，此处提供可复用的流程函数

说明：
V1.0 重点保证“能跑、可扩展”。不同数据源（航拍/平面/扫描/CAD导出图）
在识别策略上差异很大，后续可增加“识别模板/场景预设”与深度学习模型。
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .ocr_engine import ocr_rois
from .models import DetectParams, HouseFeature


LogFn = Callable[[str], None]


def write_log(logger: Optional[LogFn], msg: str) -> None:
    """统一日志输出（GUI可传入回调）"""

    if logger is not None:
        logger(msg)


def load_image(path: str) -> np.ndarray:
    """
    加载影像文件：
    - 支持 jpg/png/tif
    - pdf 需要可选依赖 PyMuPDF（fitz）
    """

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像文件：{path}")
        return img

    if suffix == ".pdf":
        # 可选：PyMuPDF
        try:
            import fitz  # type: ignore
        except Exception as e:
            raise RuntimeError("导入PDF需要安装 PyMuPDF：pip install PyMuPDF") from e

        doc = fitz.open(str(p))
        if doc.page_count < 1:
            raise ValueError(f"PDF无页面：{path}")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 简单放大提高分辨率
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    raise ValueError(f"不支持的文件类型：{suffix}")


def _preprocess_for_contours(img_bgr: np.ndarray) -> np.ndarray:
    """V1.0 兼容：轮廓识别前处理（保留原方法）"""

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值对扫描图/平面图更稳，航拍/彩色图可后续增加场景选择
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th


def _enhance_gray(gray: np.ndarray, params: DetectParams) -> np.ndarray:
    """
    影像预处理增强（V2.0）：
    - 去噪
    - 对比度增强（CLAHE）
    - 锐化
    """

    out = gray

    if params.denoise:
        # 双边滤波对线框保留更好，但开销略大
        out = cv2.bilateralFilter(out, d=7, sigmaColor=50, sigmaSpace=50)

    if params.enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if params.sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)

    return out


def _binarize(gray: np.ndarray, params: DetectParams) -> np.ndarray:
    """二值化策略（V2.0）：adaptive / otsu"""

    g = cv2.GaussianBlur(gray, (5, 5), 0)

    if params.binarize_mode.lower() == "otsu":
        _t, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return th

    th = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    return th


def _morph(bin_img: np.ndarray, params: DetectParams) -> np.ndarray:
    k = max(1, int(params.morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    out = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=int(params.morph_close_iter))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=int(params.morph_open_iter))
    return out


def _mask_from_edges(gray: np.ndarray, params: DetectParams) -> np.ndarray:
    """Canny 边缘 → 膨胀闭合成可找轮廓的mask"""

    e = cv2.Canny(gray, int(params.canny_th1), int(params.canny_th2))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    e = cv2.dilate(e, k, iterations=1)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=2)
    return e


def _simplify_contour_points(pts: np.ndarray, max_points: int) -> np.ndarray:
    """过多点的轮廓进一步简化，便于人工校正操作"""

    if pts is None or len(pts) <= max_points:
        return pts
    # 逐步增大 epsilon 直到点数降到阈值以内
    peri = cv2.arcLength(pts, True)
    eps = 0.005 * peri
    out = cv2.approxPolyDP(pts, eps, True)
    while len(out) > max_points and eps < 0.05 * peri:
        eps *= 1.2
        out = cv2.approxPolyDP(pts, eps, True)
    return out


def _contour_quality_filter(c: np.ndarray, approx: np.ndarray, params: DetectParams) -> bool:
    """轮廓筛选：面积/外接框/长宽比等过滤，尽量剔除标注线、辅助线"""

    x, y, w, h = cv2.boundingRect(approx)
    if w < params.min_bbox_side_px or h < params.min_bbox_side_px:
        return False

    area_px = float(cv2.contourArea(c))
    if area_px < params.min_contour_area_px:
        return False

    ar = max(w / max(h, 1), h / max(w, 1))
    if ar > params.max_aspect_ratio:
        return False

    # 轮廓“填充度”过滤：太空/太线状可剔除（对扫描图尺寸线有帮助）
    rect_area = float(w * h)
    if rect_area <= 0:
        return False
    solidity = area_px / rect_area
    if solidity < 0.05:
        return False

    # 点数过少的可能是线段/小矩形噪声
    if len(approx) < 4:
        return False
    return True


def detect_house(img_bgr: np.ndarray, params: DetectParams, logger: Optional[LogFn] = None) -> List[HouseFeature]:
    """
    识别房屋轮廓（V2.0：增强预处理 + Canny/二值多策略 + 多轮廓筛选）

    输出：
    - 每个轮廓生成一个 HouseFeature
    - 轮廓过滤：面积过小的噪声剔除
    """

    write_log(logger, "开始识别：图像增强 + 轮廓提取中...")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = _enhance_gray(gray, params)

    # 多策略mask融合：二值化 + 边缘
    bin1 = _binarize(gray2, params)
    bin1 = _morph(bin1, params)

    if params.use_canny:
        edge_mask = _mask_from_edges(gray2, params)
        mask = cv2.bitwise_or(bin1, edge_mask)
    else:
        mask = bin1

    # 再做一次形态学，增强闭合边界
    mask = _morph(mask, params)

    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    write_log(logger, f"发现候选轮廓 {len(contours)} 处，正在筛选优化...")

    features: List[HouseFeature] = []
    idx = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        epsilon = params.approx_epsilon_ratio * peri
        approx = cv2.approxPolyDP(c, epsilon, True)
        approx = _simplify_contour_points(approx, params.simplify_max_points)

        if not _contour_quality_filter(c, approx, params):
            continue

        area_px = float(cv2.contourArea(c))

        x, y, w, h = cv2.boundingRect(approx)
        m = cv2.moments(approx)
        if m["m00"] != 0:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
        else:
            cx = float(x + w / 2)
            cy = float(y + h / 2)

        idx += 1
        uid = str(uuid.uuid4())
        number = f"{idx:03d}" if params.auto_number else ""

        area_m2 = calc_area(area_px, params)

        features.append(
            HouseFeature(
                uid=uid,
                number=number,
                building_no=None,
                contour_area_px=area_px,
                area_m2=area_m2,
                bbox_xywh=(int(x), int(y), int(w), int(h)),
                centroid_xy=(cx, cy),
                contour_xy=[(int(p[0][0]), int(p[0][1])) for p in approx],
                ocr_text=None,
                source_path=None,
            )
        )

    write_log(logger, f"识别房屋轮廓 {len(features)} 处。")
    return features


def calc_area(area_px: float, params: DetectParams) -> float:
    """
    面积换算：像素面积 -> 平方米

    采用“比例尺 + DPI”估算每像素对应实际长度：
    - 纸面每像素长度（米） = 0.0254 / DPI
    - 实际每像素长度（米） = (0.0254 / DPI) * 比例尺分母
    - 面积（㎡） = px² * (m/px)²
    """

    dpi = max(1, int(params.dpi))
    scale = max(1, int(params.scale_denominator))
    m_per_px = (0.0254 / dpi) * scale
    area_m2 = float(area_px) * (m_per_px**2)
    return round(area_m2, 3)


def ocr_text(
    img_bgr: np.ndarray,
    rois_xywh: List[Tuple[int, int, int, int]],
    logger: Optional[LogFn] = None,
) -> List[Optional[str]]:
    """
    OCR 文字识别（可选依赖）

    V2.0：优先 PaddleOCR（支持旋转文字），回退 easyocr（多角度尝试）
    说明：识别结果默认返回“最佳文本”，后续可在GUI提供人工修正入口。
    """
    items = ocr_rois(img_bgr, rois_xywh, prefer_paddle=True, allow_rotate=True)
    out: List[Optional[str]] = []
    for it in items:
        out.append(it.text if it else None)
    if any(out):
        write_log(logger, "OCR识别完成（已启用旋转支持/中文优化）。")
    else:
        write_log(logger, "OCR未识别到有效文本（可选安装 paddleocr 或 easyocr）。")
    return out


def parse_ocr_fields(text: Optional[str]) -> Dict[str, Optional[str]]:
    """
    从 OCR 文本中粗提取字段（房号/楼栋号/面积/用途）
    说明：规则仅作为基线，建议结合项目数据进一步定制。
    """

    if not text:
        return {"房号": None, "楼栋号": None, "面积": None, "用途": None}
    t = str(text).strip()

    # 面积：如 89.23㎡ / 120 m2
    m_area = re.search(r"(\d+(?:\.\d+)?)\s*(?:㎡|m2|m²|M2|M²)", t)
    area = m_area.group(1) if m_area else None

    # 楼栋号：如 1# / 1栋 / 3号楼
    m_build = re.search(r"(\d+)\s*(?:#|栋|号楼)", t)
    building = m_build.group(1) if m_build else None

    # 房号：如 1-101 / 101 / A101
    m_house = re.search(r"([A-Za-z]?\d{2,4}(?:-\d{2,4})?)", t)
    house = m_house.group(1) if m_house else None

    # 用途（简单关键词）
    use = None
    for kw in ["住宅", "商铺", "厂房", "办公", "公寓", "车位"]:
        if kw in t:
            use = kw
            break

    return {"房号": house, "楼栋号": building, "面积": area, "用途": use}


def match_property(
    features: List[HouseFeature],
    prop_df: Optional[pd.DataFrame],
    logger: Optional[LogFn] = None,
) -> pd.DataFrame:
    """
    属性匹配：将识别结果与Excel属性表关联

    规则（V1.0 简化）：
    - 若属性表存在“房号”字段，与 feature.number 或 ocr_text 进行匹配
    - 识别结果作为主表，属性字段左连接补齐
    """

    base_rows: List[Dict] = []
    for f in features:
        row = {
            "uid": f.uid,
            "房号": f.number,
            "楼栋号": f.building_no,
            "OCR": f.ocr_text,
            "像素面积": f.contour_area_px,
            "建筑面积(㎡)": f.area_m2,
            "x": f.bbox_xywh[0],
            "y": f.bbox_xywh[1],
            "w": f.bbox_xywh[2],
            "h": f.bbox_xywh[3],
            "质心x": f.centroid_xy[0],
            "质心y": f.centroid_xy[1],
            "源文件": f.source_path,
        }
        base_rows.append(row)

    df = pd.DataFrame(base_rows)
    if prop_df is None or prop_df.empty:
        write_log(logger, "未加载Excel属性表，已跳过属性匹配。")
        return df

    # 统一字段名容错
    cols = {c.strip(): c for c in prop_df.columns}
    key_col = cols.get("房号") or cols.get("房间号") or cols.get("编号")
    if not key_col:
        write_log(logger, "属性表缺少“房号”字段（或同义字段），已跳过匹配。")
        return df

    prop = prop_df.copy()
    prop["_match_key"] = prop[key_col].astype(str).str.strip()

    # 识别表匹配优先：房号，其次OCR文本
    df["_match_key"] = df["房号"].astype(str).str.strip()
    merged = df.merge(prop.drop_duplicates("_match_key"), how="left", on="_match_key", suffixes=("", "_属性"))
    merged.drop(columns=["_match_key"], inplace=True)

    write_log(logger, "属性匹配完成。")
    return merged


def clean_data(df: pd.DataFrame, logger: Optional[LogFn] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    数据清洗：生成异常数据表

    检查项：
    1) 面积为空/为0
    2) 房号重复
    3) 属性缺失（业主姓名/用途/楼栋号等存在则检查）
    4) 面积异常（过大过小，阈值可后续参数化）
    """

    if df.empty:
        return df, pd.DataFrame()

    df2 = df.copy()
    abn_rows: List[Dict] = []

    # 规则阈值（V1.0固定，后续可在GUI中配置）
    min_area = 5.0
    max_area = 2000.0

    # 房号重复
    dup_mask = df2["房号"].astype(str).str.strip().duplicated(keep=False)

    for _, row in df2.iterrows():
        reasons: List[str] = []

        area = row.get("建筑面积(㎡)")
        try:
            area_f = float(area)
        except Exception:
            area_f = math.nan

        if not (area_f > 0):
            reasons.append("面积为空/为0")
        else:
            if area_f < min_area:
                reasons.append("面积过小")
            if area_f > max_area:
                reasons.append("面积过大")

        house_no = str(row.get("房号", "")).strip()
        if not house_no:
            reasons.append("房号缺失")

        if bool(dup_mask.loc[row.name]):
            reasons.append("房号重复")

        # 常见属性字段缺失检查（若存在这些列才检查）
        for col in ["楼栋号", "业主姓名", "用途", "层数", "地址"]:
            if col in df2.columns:
                v = row.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
                    reasons.append(f"{col}缺失")

        if reasons:
            r = row.to_dict()
            r["异常原因"] = "；".join(sorted(set(reasons)))
            abn_rows.append(r)

    abnormal_df = pd.DataFrame(abn_rows)
    write_log(logger, f"数据清洗完成：异常 {len(abnormal_df)} 条。")
    return df2, abnormal_df


def _draw_north_arrow(ax) -> None:
    """绘制指北针（简版）"""

    ax.annotate(
        "N",
        xy=(0.95, 0.15),
        xytext=(0.95, 0.05),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
        fontsize=12,
        fontweight="bold",
    )


def draw_map(
    img_bgr: np.ndarray,
    features: List[HouseFeature],
    params: DetectParams,
    out_png: str,
    out_pdf: Optional[str] = None,
    logger: Optional[LogFn] = None,
) -> None:
    """
    自动制图（V1.0：matplotlib叠加绘制）

    输出：
    - PNG：叠加轮廓/编号/面积/图例/指北针/比例尺说明
    - PDF：可选（同图导出）
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433（运行时导入以减少GUI启动开销）

    write_log(logger, "正在生成识别图纸（PNG/PDF）...")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(w / 120, h / 120), dpi=120)
    ax.imshow(img_rgb)

    for f in features:
        if f.contour_xy:
            xs = [p[0] for p in f.contour_xy] + [f.contour_xy[0][0]]
            ys = [p[1] for p in f.contour_xy] + [f.contour_xy[0][1]]
            ax.plot(xs, ys, color="lime", linewidth=1.4)
        else:
            x, y, bw, bh = f.bbox_xywh
            rect = plt.Rectangle((x, y), bw, bh, fill=False, linewidth=1.2, edgecolor="lime")
            ax.add_patch(rect)
        cx, cy = f.centroid_xy
        label = f"{f.number}\n{f.area_m2}㎡"
        ax.text(cx, cy, label, color="yellow", fontsize=8, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.35, pad=2))

    _draw_north_arrow(ax)
    ax.set_title(f"房屋识别成果图（比例尺 1:{params.scale_denominator}，DPI={params.dpi}）", fontsize=12)
    ax.axis("off")

    # 图例/说明（简版）
    ax.text(
        0.01,
        0.99,
        "图例：绿色框=房屋外接范围；黄色文字=编号+面积",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, pad=3),
    )

    out_png_path = Path(out_png)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png_path), bbox_inches="tight", pad_inches=0.05)
    write_log(logger, f"识别结果PNG已生成：{out_png_path}")

    if out_pdf:
        out_pdf_path = Path(out_pdf)
        out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_pdf_path), bbox_inches="tight", pad_inches=0.05)
        write_log(logger, f"识别图纸PDF已生成：{out_pdf_path}")

    plt.close(fig)


def stat_summary(df: pd.DataFrame, logger: Optional[LogFn] = None) -> pd.DataFrame:
    """统计分析：输出汇总报表（单表）"""

    if df.empty:
        return pd.DataFrame([{"指标": "总房屋数量", "数值": 0}])

    total_count = int(len(df))
    total_area = float(pd.to_numeric(df.get("建筑面积(㎡)"), errors="coerce").fillna(0).sum())

    # 分类统计（若存在“房屋类别/用途”等字段则更准确）
    cat_col = "房屋类别" if "房屋类别" in df.columns else None

    rows = [
        {"指标": "总房屋数量", "数值": total_count},
        {"指标": "总建筑面积(㎡)", "数值": round(total_area, 3)},
    ]

    if cat_col:
        for cat, g in df.groupby(cat_col):
            rows.append({"指标": f"{cat}数量", "数值": int(len(g))})

    if "楼栋号" in df.columns:
        bcount = int(df["楼栋号"].astype(str).str.strip().replace({"nan": ""}).ne("").sum())
        rows.append({"指标": "楼栋号非空数量", "数值": bcount})

    write_log(logger, "统计分析完成。")
    return pd.DataFrame(rows)


def _safe_sheet_name(name: str) -> str:
    """Excel工作表名容错"""

    invalid = ["\\", "/", "*", "?", ":", "[", "]"]
    for ch in invalid:
        name = name.replace(ch, "_")
    return name[:31] if len(name) > 31 else name


def export_excel(
    ledger_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_xlsx: str,
    logger: Optional[LogFn] = None,
) -> None:
    """导出 Excel：房屋台账、异常数据、汇总报表"""

    out_path = Path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        ledger_df.to_excel(writer, index=False, sheet_name=_safe_sheet_name("房屋台账"))
        if abnormal_df is not None and not abnormal_df.empty:
            abnormal_df.to_excel(writer, index=False, sheet_name=_safe_sheet_name("异常数据"))
        summary_df.to_excel(writer, index=False, sheet_name=_safe_sheet_name("汇总报表"))

    write_log(logger, f"Excel导出完成：{out_path}")


def export_pdf() -> None:
    """
    预留：复杂多页PDF制图/模板化出图
    V1.0 已在 draw_map() 中实现单页导出。
    """

    return


def run_pipeline(
    image_paths: List[str],
    params: DetectParams,
    prop_df: Optional[pd.DataFrame],
    out_dir: str,
    logger: Optional[LogFn] = None,
    cad_paths: Optional[List[str]] = None,
    return_data: bool = False,
) -> Dict[str, str]:
    """
    统一流程：识别 → OCR → 属性匹配 → 清洗 → 统计 → 制图 → 导出

    返回输出文件路径字典，便于GUI显示/打开目录。
    """

    out = {}
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    all_features: List[HouseFeature] = []
    last_img: Optional[np.ndarray] = None

    for p in image_paths:
        write_log(logger, f"已加载影像：{p}")
        img = load_image(p)
        last_img = img
        feats = detect_house(img, params, logger=logger)
        for f in feats:
            f.source_path = p
        all_features.extend(feats)

    # DXF（可选）
    if cad_paths:
        try:
            from .cad_dxf import load_dxf_detect_rooms  # noqa: WPS433（可选依赖）

            for dxf in cad_paths:
                feats2 = load_dxf_detect_rooms(dxf, params, logger=logger).features
                all_features.extend(feats2)
        except Exception as e:
            write_log(logger, f"DXF解析失败/未启用：{e}")

    if not all_features:
        write_log(logger, "未识别到有效房屋轮廓，请调整影像质量或后续增加场景参数。")
        # 仍导出空表，保证流程不崩
        ledger = match_property(all_features, prop_df, logger=logger)
        ledger, abnormal = clean_data(ledger, logger=logger)
        summary = stat_summary(ledger, logger=logger)
        excel_path = str(out_base / "房屋成果.xlsx")
        export_excel(ledger, abnormal, summary, excel_path, logger=logger)
        out["excel"] = excel_path
        return out

    # OCR：默认对每个bbox做一次识别（V1.0粗粒度）
    if params.auto_ocr and last_img is not None:
        rois = [f.bbox_xywh for f in all_features]
        texts = ocr_text(last_img, rois, logger=logger)
        for f, t in zip(all_features, texts):
            f.ocr_text = t
            # 若房号为空则用OCR补齐
            if (not f.number) and t:
                fields = parse_ocr_fields(t)
                f.number = fields.get("房号") or t
                if not f.building_no and fields.get("楼栋号"):
                    f.building_no = fields.get("楼栋号")

    ledger = match_property(all_features, prop_df, logger=logger)
    # 将GUI里的类别写入表（便于统计）
    ledger["房屋类别"] = params.house_category

    ledger, abnormal = clean_data(ledger, logger=logger)
    summary = stat_summary(ledger, logger=logger)

    # 输出文件路径
    ledger_xlsx = str(out_base / "房屋台账.xlsx")
    export_excel(ledger, abnormal, summary, ledger_xlsx, logger=logger)
    out["excel"] = ledger_xlsx

    # 制图：使用最后一张图作为底图（V1.0）；后续可按每张影像分别出图
    if last_img is not None:
        out_png = str(out_base / "识别结果.png")
        out_pdf = str(out_base / "识别图纸.pdf")
        draw_map(last_img, all_features, params, out_png=out_png, out_pdf=out_pdf, logger=logger)
        out["png"] = out_png
        out["pdf"] = out_pdf

    write_log(logger, "一键导出完成。")
    if return_data:
        # 注意：features/df 为内存对象，GUI可用于预览与人工校正
        out["__has_data__"] = "1"
        out["_ledger_rows"] = str(len(ledger))
        out["_abnormal_rows"] = str(len(abnormal))
        # 将对象挂到 dict 上（GUI线程内使用，不做序列化）
        out_obj = {"paths": out, "features": all_features, "ledger_df": ledger, "abnormal_df": abnormal, "summary_df": summary, "last_img": last_img}
        return out_obj  # type: ignore[return-value]
    return out

