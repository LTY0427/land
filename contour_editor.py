"""
人工校正模式：轮廓点编辑（V2.0）

实现：
- QGraphicsView 中展示底图 + 多边形点
- 顶点可拖动；保存后写回 HouseFeature.contour_xy

说明：
- V2.0 提供“可用的基础编辑器”（单目标编辑）
- 后续可扩展：新增/删除点、吸附、约束直角、批量编辑等
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QImage, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QVBoxLayout,
)


class DraggablePoint(QGraphicsEllipseItem):
    """
    可拖拽顶点。

    注意：
    - QGraphicsItem 不是 QObject，不能直接使用 QtCore.Signal
    - 这里用回调方式通知外部更新路径
    """

    def __init__(self, x: float, y: float, r: float = 5.0, on_moved: Optional[Callable[[], None]] = None):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(QPointF(x, y))
        self.setBrush(QBrush(Qt.white))
        self.setPen(QPen(Qt.blue, 2))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setZValue(10)
        self._on_moved = on_moved

    def itemChange(self, change, value):  # noqa: N802（Qt约定命名）
        if change == QGraphicsEllipseItem.ItemPositionHasChanged:
            if self._on_moved:
                self._on_moved()
        return super().itemChange(change, value)


class ContourEditorDialog(QDialog):
    """
    单个轮廓编辑对话框。
    """

    def __init__(self, img_qimage: Optional[QImage], contour_xy: List[Tuple[int, int]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("人工校正 - 轮廓点编辑")
        self.resize(1000, 700)

        self._orig_contour = contour_xy
        self._points: List[DraggablePoint] = []

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        if img_qimage is not None:
            pix = QPixmap.fromImage(img_qimage)
            self.scene.addItem(QGraphicsPixmapItem(pix))

        self.path_item = QGraphicsPathItem()
        self.path_item.setPen(QPen(Qt.red, 2))
        self.path_item.setZValue(5)
        self.scene.addItem(self.path_item)

        for x, y in contour_xy:
            p = DraggablePoint(float(x), float(y), r=5.0, on_moved=self._update_path)
            self._points.append(p)
            self.scene.addItem(p)

        self._update_path()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        layout.addWidget(btns)

    def _update_path(self) -> None:
        if not self._points:
            self.path_item.setPath(QPainterPath())
            return
        pts = [p.pos() for p in self._points]
        path = QPainterPath()
        path.moveTo(pts[0])
        for q in pts[1:]:
            path.lineTo(q)
        path.closeSubpath()
        self.path_item.setPath(path)

    def get_contour_xy(self) -> List[Tuple[int, int]]:
        """获取编辑后的轮廓点"""

        out = []
        for p in self._points:
            out.append((int(round(p.pos().x())), int(round(p.pos().y()))))
        return out

