"""
OCR 引擎封装（V2.0）

目标：
- 支持旋转文字（角度分类/多角度尝试）
- 提高中文识别率（优先 PaddleOCR）
- 输出可用于人工修正的结构化结果
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class OcrItem:
    text: str
    score: float


def _rotate_image(img: np.ndarray, angle_deg: int) -> np.ndarray:
    """按 90 度倍数旋转"""

    if angle_deg % 360 == 0:
        return img
    if angle_deg % 360 == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle_deg % 360 == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle_deg % 360 == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # 非90倍数角度：简单仿射（保守实现，避免ROI裁切）
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def ocr_rois(
    img_bgr: np.ndarray,
    rois_xywh: Sequence[Tuple[int, int, int, int]],
    *,
    prefer_paddle: bool = True,
    allow_rotate: bool = True,
) -> List[Optional[OcrItem]]:
    """
    对一组 ROI 做 OCR。

    返回：
    - 每个 ROI 返回一个 OcrItem（text/score）或 None
    """

    if prefer_paddle:
        try:
            from paddleocr import PaddleOCR  # type: ignore

            # 中文 + 英文，支持角度分类（旋转文字）
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            out: List[Optional[OcrItem]] = []
            for (x, y, w, h) in rois_xywh:
                roi = _safe_crop(img_bgr, x, y, w, h)
                if roi is None:
                    out.append(None)
                    continue
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                res = ocr.ocr(roi_rgb, cls=True)
                best = _pick_best_paddle(res)
                out.append(best)
            return out
        except Exception:
            # 回退 easyocr
            pass

    try:
        import easyocr  # type: ignore

        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        angles = [0, 90, 180, 270] if allow_rotate else [0]

        out2: List[Optional[OcrItem]] = []
        for (x, y, w, h) in rois_xywh:
            roi = _safe_crop(img_bgr, x, y, w, h)
            if roi is None:
                out2.append(None)
                continue

            best_item: Optional[OcrItem] = None
            for a in angles:
                rr = _rotate_image(roi, a)
                rr_rgb = cv2.cvtColor(rr, cv2.COLOR_BGR2RGB)
                # easyocr detail=1 可拿到置信度
                res = reader.readtext(rr_rgb, detail=1, paragraph=False)
                item = _pick_best_easyocr(res)
                if item is None:
                    continue
                if (best_item is None) or (item.score > best_item.score) or (len(item.text) > len(best_item.text) and item.score >= best_item.score * 0.9):
                    best_item = item

            out2.append(best_item)
        return out2
    except Exception:
        return [None for _ in rois_xywh]


def _safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    """ROI 裁切安全处理"""

    if img is None or img.size == 0:
        return None
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(int(img.shape[1]), int(x + w)), min(int(img.shape[0]), int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = img[y0:y1, x0:x1]
    return roi if roi.size else None


def _pick_best_easyocr(res) -> Optional[OcrItem]:
    """从 easyocr 输出中挑选最佳文本"""

    if not res:
        return None
    best = None
    for r in res:
        # r = (bbox, text, conf)
        if len(r) < 3:
            continue
        text = str(r[1]).strip()
        score = float(r[2])
        if not text:
            continue
        item = OcrItem(text=text, score=score)
        if (best is None) or (item.score > best.score) or (len(item.text) > len(best.text) and item.score >= best.score * 0.9):
            best = item
    return best


def _pick_best_paddle(res) -> Optional[OcrItem]:
    """从 PaddleOCR 输出中挑选最佳文本"""

    # paddleocr: res 形如 [ [ [box], (text, score) ], ... ]
    if not res:
        return None
    cand = []
    for line in res:
        if not line:
            continue
        for item in line:
            if not item or len(item) < 2:
                continue
            text_score = item[1]
            if not text_score or len(text_score) < 2:
                continue
            text = str(text_score[0]).strip()
            score = float(text_score[1])
            if text:
                cand.append(OcrItem(text=text, score=score))
    if not cand:
        return None
    return max(cand, key=lambda x: (x.score, len(x.text)))

