"""
PySide6 主界面（V2.0）

要求实现：
- 左侧导航栏：首页/数据导入/识别处理/统计分析/案例展示/系统设置
- 顶部系统名称：房屋影像识别与属性建库制图系统 V2.0
- 蓝白科技风 + 卡片式布局 + 图标按钮化（使用Qt标准图标，避免额外资源依赖）
- 进度条、状态栏、提示动画（轻量实现）

并保持原有功能逻辑：
- 导入影像/Excel属性表/CAD(dxf)
- 开始识别/一键导出/打开输出目录/清空
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd
import cv2
import numpy as np
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .analytics import generate_charts
from .case_gallery import ensure_case_assets
from .contour_editor import ContourEditorDialog
from .core import run_pipeline
from .models import DetectParams, HouseFeature
from .utils import ensure_dir, now_str, open_in_explorer


class Worker(QThread):
    """后台任务线程：避免GUI卡顿"""

    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        image_paths: List[str],
        cad_paths: List[str],
        params: DetectParams,
        prop_df: Optional[pd.DataFrame],
        out_dir: str,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.cad_paths = cad_paths
        self.params = params
        self.prop_df = prop_df
        self.out_dir = out_dir

    def run(self) -> None:
        try:
            out_obj = run_pipeline(
                image_paths=self.image_paths,
                params=self.params,
                prop_df=self.prop_df,
                out_dir=self.out_dir,
                logger=self._logger,
                cad_paths=self.cad_paths,
                return_data=True,
            )
            # 识别完成后生成图表（PNG + 可选HTML）
            try:
                ledger_df = out_obj.get("ledger_df")
                abnormal_df = out_obj.get("abnormal_df")
                if ledger_df is not None and abnormal_df is not None:
                    charts = generate_charts(ledger_df, abnormal_df, self.out_dir)
                    out_obj["charts"] = charts
            except Exception as e:
                self._logger(f"图表生成失败：{e}")

            self.finished.emit(out_obj)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

    def _logger(self, msg: str) -> None:
        self.log.emit(msg)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("房屋影像识别与属性建库制图系统 V2.0")
        self.resize(1320, 820)
        self.setAcceptDrops(True)  # 启用拖拽导入

        self.image_paths: List[str] = []
        self.cad_paths: List[str] = []
        self.excel_path: Optional[str] = None
        self.prop_df: Optional[pd.DataFrame] = None

        self.out_dir = str(ensure_dir(Path.cwd() / "输出结果"))
        self._worker: Optional[Worker] = None

        # 识别结果缓存（用于预览/人工校正/统计预览）
        self._features: List[HouseFeature] = []
        self._last_img_bgr: Optional[np.ndarray] = None
        self._ledger_df: Optional[pd.DataFrame] = None
        self._abnormal_df: Optional[pd.DataFrame] = None
        self._summary_df: Optional[pd.DataFrame] = None
        self._charts: dict = {}

        self._init_ui()
        self._apply_theme()

    def _init_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # 左侧导航
        self.nav = QListWidget()
        self.nav.setFixedWidth(220)
        self.nav.setSpacing(4)
        self.nav.setSelectionMode(QAbstractItemView.SingleSelection)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._add_nav_item("首页", "🏠")
        self._add_nav_item("数据导入", "📥")
        self._add_nav_item("识别处理", "🧠")
        self._add_nav_item("统计分析", "📊")
        self._add_nav_item("案例展示", "🗂")
        self._add_nav_item("系统设置", "⚙")
        self.nav.currentRowChanged.connect(self._on_nav_changed)

        # 右侧内容区：顶部标题 + 页面
        right = QVBoxLayout()
        right.setContentsMargins(16, 16, 16, 16)
        right.setSpacing(12)

        header = QHBoxLayout()
        self.lab_title = QLabel("房屋影像识别与属性建库制图系统 V2.0")
        self.lab_title.setObjectName("TitleLabel")
        header.addWidget(self.lab_title)
        header.addStretch(1)

        self.btn_case_popup = QPushButton("案例示例")
        # 兼容不同Qt版本的标准图标枚举
        self.btn_case_popup.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        self.btn_case_popup.clicked.connect(self.on_show_cases)
        header.addWidget(self.btn_case_popup)

        right.addLayout(header)

        self.pages = QStackedWidget()
        self.page_home = self._build_page_home()
        self.page_import = self._build_page_import()
        self.page_process = self._build_page_process()
        self.page_analytics = self._build_page_analytics()
        self.page_cases = self._build_page_cases()
        self.page_settings = self._build_page_settings()

        for p in [self.page_home, self.page_import, self.page_process, self.page_analytics, self.page_cases, self.page_settings]:
            self.pages.addWidget(p)

        right.addWidget(self.pages, 1)

        # 底部：日志 + 进度条
        bottom = QHBoxLayout()
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 10))
        self.txt_log.setFixedHeight(170)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 不确定进度（V2.0先保证可用）
        self.progress.setVisible(False)
        self.progress.setFixedHeight(18)

        bottom_col = QVBoxLayout()
        bottom_col.addWidget(self.txt_log)
        bottom_col.addWidget(self.progress)
        bottom.addLayout(bottom_col, 1)

        right.addLayout(bottom)

        main.addWidget(self.nav)
        main.addLayout(right, 1)

        # 状态栏 + 轻提示
        self.statusBar().showMessage("就绪")
        self.toast = QLabel("", self)
        self.toast.setObjectName("Toast")
        self.toast.setVisible(False)
        self.toast_anim = QPropertyAnimation(self.toast, b"windowOpacity", self)
        self.toast_anim.setDuration(1500)
        self.toast_anim.setEasingCurve(QEasingCurve.InOutQuad)

        self.nav.setCurrentRow(0)

    def _apply_theme(self) -> None:
        """
        蓝白科技风：尽量用样式表实现，保持无额外资源依赖。
        """

        qss = """
        QWidget { background: #F6F9FF; }
        #TitleLabel { font-size: 20px; font-weight: 700; color: #0B3D91; }
        QListWidget { background: #0B3D91; color: white; border: 0px; padding: 10px; }
        QListWidget::item { padding: 10px 10px; border-radius: 10px; }
        QListWidget::item:selected { background: rgba(255,255,255,0.18); }
        QTextEdit { background: white; border: 1px solid #E3EAF6; border-radius: 12px; padding: 8px; }
        QPushButton { background: white; border: 1px solid #D7E3F7; border-radius: 10px; padding: 8px 12px; }
        QPushButton:hover { border-color: #2E6BE6; }
        QPushButton:disabled { color: #999; }
        QProgressBar { background: white; border: 1px solid #D7E3F7; border-radius: 8px; text-align: center; }
        QProgressBar::chunk { background: #2E6BE6; border-radius: 8px; }
        QLabel#Toast { background: rgba(11,61,145,0.92); color: white; padding: 10px 14px; border-radius: 10px; }
        """
        self.setStyleSheet(qss)

    def resizeEvent(self, event) -> None:  # noqa: N802（Qt约定命名）
        super().resizeEvent(event)
        # Toast 放右下角
        if self.toast:
            w = min(520, int(self.width() * 0.5))
            self.toast.setFixedWidth(w)
            self.toast.move(self.width() - w - 24, self.height() - 90)

    def _toast(self, msg: str) -> None:
        self.toast.setText(msg)
        self.toast.setVisible(True)
        self.toast.setWindowOpacity(0.0)
        self.toast_anim.stop()
        self.toast_anim.setStartValue(0.0)
        self.toast_anim.setKeyValueAt(0.15, 1.0)
        self.toast_anim.setKeyValueAt(0.85, 1.0)
        self.toast_anim.setEndValue(0.0)
        self.toast_anim.finished.connect(lambda: self.toast.setVisible(False))
        self.toast_anim.start()

    def _add_nav_item(self, name: str, emoji: str) -> None:
        item = QListWidgetItem(f"{emoji}  {name}")
        self.nav.addItem(item)

    def _on_nav_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self.pages.setCurrentIndex(idx)
        self.statusBar().showMessage(f"当前页面：{self.nav.item(idx).text().strip()}")

    # -------------------------
    # 页面构建
    # -------------------------
    def _card(self, title: str) -> QWidget:
        w = QWidget()
        w.setObjectName("Card")
        w.setStyleSheet("QWidget#Card{background:white; border:1px solid #E3EAF6; border-radius:14px;}")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)
        lab = QLabel(title)
        lab.setStyleSheet("font-size:14px; font-weight:700; color:#163A78;")
        lay.addWidget(lab)
        return w

    def _build_page_home(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)

        card = self._card("欢迎使用")
        lay = card.layout()
        lab = QLabel(
            "本系统用于影像/图纸识别房屋轮廓、面积换算、属性匹配、制图与台账导出。\n"
            "建议流程：数据导入 → 识别处理 → 统计分析 → 导出成果。\n"
            "提示：如需 DXF 识别请安装 ezdxf；如需高质量 OCR 请安装 paddleocr。"
        )
        lab.setStyleSheet("color:#3A4A66; line-height:1.6;")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        v.addWidget(card)
        v.addStretch(1)
        return page

    def _build_page_import(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        card = self._card("文件导入")
        lay = card.layout()

        btn_row = QHBoxLayout()
        self.btn_import_images = QPushButton("导入影像")
        self.btn_import_images.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.btn_import_cad = QPushButton("导入DXF")
        self.btn_import_cad.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_import_excel = QPushButton("导入属性表")
        self.btn_import_excel.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        btn_row.addWidget(self.btn_import_images)
        btn_row.addWidget(self.btn_import_cad)
        btn_row.addWidget(self.btn_import_excel)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self.lab_images = QLabel("影像：未加载")
        self.lab_cad = QLabel("CAD：未加载")
        self.lab_excel = QLabel("属性表：未加载")
        self.lab_out = QLabel(f"输出目录：{self.out_dir}")
        for lab in [self.lab_images, self.lab_cad, self.lab_excel, self.lab_out]:
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setStyleSheet("color:#3A4A66;")
            lay.addWidget(lab)

        tip = QLabel("支持拖拽导入：影像/pdf/excel/dxf。")
        tip.setStyleSheet("color:#667799;")
        lay.addWidget(tip)

        self.btn_import_images.clicked.connect(self.on_import_images)
        self.btn_import_cad.clicked.connect(self.on_import_cad)
        self.btn_import_excel.clicked.connect(self.on_import_excel)

        v.addWidget(card)
        v.addStretch(1)
        return page

    def _build_page_process(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        # 参数卡片
        card_params = self._card("参数设置（识别增强已默认开启）")
        lay = card_params.layout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("比例尺(1:)"))
        self.sp_scale = QSpinBox()
        self.sp_scale.setRange(1, 100000)
        self.sp_scale.setValue(100)
        row1.addWidget(self.sp_scale)
        row1.addSpacing(10)
        row1.addWidget(QLabel("DPI"))
        self.sp_dpi = QSpinBox()
        self.sp_dpi.setRange(50, 2400)
        self.sp_dpi.setValue(300)
        row1.addWidget(self.sp_dpi)
        row1.addStretch(1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("房屋类别"))
        self.cb_category = QComboBox()
        self.cb_category.addItems(["住宅", "商铺", "厂房"])
        row2.addWidget(self.cb_category)

        self.ck_auto_number = QCheckBox("自动编号")
        self.ck_auto_ocr = QCheckBox("自动OCR（旋转支持）")
        self.ck_auto_number.setChecked(True)
        self.ck_auto_ocr.setChecked(True)
        row2.addSpacing(20)
        row2.addWidget(self.ck_auto_number)
        row2.addWidget(self.ck_auto_ocr)
        row2.addStretch(1)
        lay.addLayout(row2)

        hint = QLabel("V2.0：已启用去噪/对比度增强/锐化 + Canny 边缘融合 + 轮廓筛选。")
        hint.setStyleSheet("color:#667799;")
        lay.addWidget(hint)

        # 操作卡片
        card_ops = self._card("识别与导出")
        lay2 = card_ops.layout()

        ops = QHBoxLayout()
        self.btn_start = QPushButton("开始识别")
        self.btn_start.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_export = QPushButton("一键导出")
        self.btn_export.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_open_out = QPushButton("打开输出目录")
        self.btn_open_out.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_clear = QPushButton("清空数据")
        self.btn_clear.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.btn_edit = QPushButton("人工校正（编辑轮廓点）")
        self.btn_edit.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))

        for b in [self.btn_start, self.btn_export, self.btn_edit, self.btn_open_out, self.btn_clear]:
            b.setMinimumHeight(38)
            ops.addWidget(b)
        ops.addStretch(1)
        lay2.addLayout(ops)

        # 预览与结果列表
        content = QHBoxLayout()

        left = QVBoxLayout()
        self.lab_preview = QLabel("识别预览将在此显示")
        self.lab_preview.setAlignment(Qt.AlignCenter)
        self.lab_preview.setStyleSheet("background:white; border:1px dashed #D7E3F7; border-radius:14px; color:#667799;")
        self.lab_preview.setMinimumHeight(320)
        left.addWidget(self.lab_preview, 1)

        content.addLayout(left, 2)

        right = QVBoxLayout()
        self.list_features = QListWidget()
        self.list_features.setStyleSheet("background:white; border:1px solid #E3EAF6; border-radius:14px;")
        right.addWidget(QLabel("识别对象列表（双击可定位）"))
        right.addWidget(self.list_features, 1)
        content.addLayout(right, 1)

        lay2.addLayout(content)

        self.btn_start.clicked.connect(self.on_start)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_open_out.clicked.connect(self.on_open_out)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_edit.clicked.connect(self.on_manual_edit)
        self.list_features.itemDoubleClicked.connect(self.on_feature_double_click)

        v.addWidget(card_params)
        v.addWidget(card_ops, 1)
        return page

    def _build_page_analytics(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        card = self._card("统计分析（识别完成后自动生成）")
        lay = card.layout()

        self.tabs_charts = QTabWidget()
        self.tab_png = QWidget()
        self.tab_html = QWidget()
        self.tabs_charts.addTab(self.tab_png, "PNG图表")
        self.tabs_charts.addTab(self.tab_html, "HTML动态图表（pyecharts）")

        self.png_grid = QVBoxLayout(self.tab_png)
        self.png_grid.setContentsMargins(8, 8, 8, 8)
        self.lab_chart1 = QLabel("等待生成图表...")
        self.lab_chart1.setAlignment(Qt.AlignCenter)
        self.lab_chart1.setStyleSheet("background:white; border:1px dashed #D7E3F7; border-radius:14px; color:#667799;")
        self.png_grid.addWidget(self.lab_chart1)

        self.html_layout = QVBoxLayout(self.tab_html)
        self.html_layout.setContentsMargins(8, 8, 8, 8)
        self.lab_html_tip = QLabel("未生成/未安装 pyecharts。安装后可输出 HTML 并在此预览。")
        self.lab_html_tip.setStyleSheet("color:#667799;")
        self.html_layout.addWidget(self.lab_html_tip)
        self.web = None
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore

            self.web = QWebEngineView()
            self.html_layout.addWidget(self.web, 1)
        except Exception:
            self.web = None

        lay.addWidget(self.tabs_charts, 1)
        v.addWidget(card, 1)
        return page

    def _build_page_cases(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        card = self._card("案例展示")
        lay = card.layout()
        lab = QLabel("点击右上角“案例示例”弹窗查看案例。你也可以将真实案例图片放到 `assets/cases/` 覆盖占位图。")
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#3A4A66;")
        lay.addWidget(lab)
        v.addWidget(card)
        v.addStretch(1)
        return page

    def _build_page_settings(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        card = self._card("系统设置（预留）")
        lay = card.layout()
        lab = QLabel(
            "V2.0 预留设置项：\n"
            "- DXF 单位（mm/m）与面积换算\n"
            "- 识别参数模板（航拍/扫描/平面图）\n"
            "- 图层过滤规则（CAD）\n"
            "- SQLite 项目管理\n"
        )
        lab.setStyleSheet("color:#3A4A66;")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        v.addWidget(card)
        v.addStretch(1)
        return page

    # -------------------------
    # 拖拽导入
    # -------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802（Qt约定命名）
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802（Qt约定命名）
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
        self._ingest_paths(paths)

    def _ingest_paths(self, paths: List[str]) -> None:
        imgs, excels, cads = [], [], []
        for p in paths:
            suf = Path(p).suffix.lower()
            if suf in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}:
                imgs.append(p)
            elif suf in {".xlsx", ".xls"}:
                excels.append(p)
            elif suf in {".dxf"}:
                cads.append(p)

        if imgs:
            self.image_paths.extend(imgs)
            self.image_paths = sorted(set(self.image_paths))
            self._refresh_file_labels()
            self._log(f"拖拽导入影像 {len(imgs)} 个。")

        if cads:
            self.cad_paths.extend(cads)
            self.cad_paths = sorted(set(self.cad_paths))
            self._refresh_file_labels()
            self._log(f"拖拽导入CAD(dxf) {len(cads)} 个。")

        if excels:
            # 仅取最后一个属性表
            self.excel_path = excels[-1]
            self._load_excel(self.excel_path)

    # -------------------------
    # 日志
    # -------------------------
    def _log(self, msg: str) -> None:
        self.txt_log.append(f"[{now_str()}] {msg}")
        self.txt_log.ensureCursorVisible()
        self.statusBar().showMessage(msg)

    # -------------------------
    # 文件导入
    # -------------------------
    def on_import_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择影像文件",
            "",
            "Image/PDF (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.pdf)",
        )
        if not paths:
            return
        self.image_paths.extend(paths)
        self.image_paths = sorted(set(self.image_paths))
        self._refresh_file_labels()
        self._log(f"已导入影像文件 {len(paths)} 个。")

    def on_import_cad(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择DXF文件", "", "DXF (*.dxf)")
        if not paths:
            return
        self.cad_paths.extend(paths)
        self.cad_paths = sorted(set(self.cad_paths))
        self._refresh_file_labels()
        self._log("已导入CAD图纸（V2.0：支持ezdxf解析房间边界）。")

    def on_import_excel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择Excel属性表", "", "Excel (*.xlsx *.xls)")
        if not path:
            return
        self.excel_path = path
        self._load_excel(path)

    def _load_excel(self, path: str) -> None:
        try:
            self.prop_df = pd.read_excel(path)
            self._refresh_file_labels()
            self._log(f"已加载Excel属性表：{path}（行数：{len(self.prop_df)}）")
        except Exception as e:
            QMessageBox.critical(self, "Excel读取失败", str(e))

    def _refresh_file_labels(self) -> None:
        if hasattr(self, "lab_images"):
            self.lab_images.setText(f"影像：{len(self.image_paths)} 个；示例：{self.image_paths[0] if self.image_paths else '未加载'}")
        if hasattr(self, "lab_cad"):
            self.lab_cad.setText(f"CAD：{len(self.cad_paths)} 个；示例：{self.cad_paths[0] if self.cad_paths else '未加载'}")
        if hasattr(self, "lab_excel"):
            self.lab_excel.setText(f"属性表：{self.excel_path or '未加载'}")
        if hasattr(self, "lab_out"):
            self.lab_out.setText(f"输出目录：{self.out_dir}")

    # -------------------------
    # 参数读取
    # -------------------------
    def _get_params(self) -> DetectParams:
        return DetectParams(
            scale_denominator=int(self.sp_scale.value()),
            dpi=int(self.sp_dpi.value()),
            house_category=str(self.cb_category.currentText()),
            auto_number=bool(self.ck_auto_number.isChecked()),
            auto_ocr=bool(self.ck_auto_ocr.isChecked()),
        )

    # -------------------------
    # 功能操作
    # -------------------------
    def _guard_can_run(self) -> bool:
        if (not self.image_paths) and (not self.cad_paths):
            QMessageBox.warning(self, "提示", "请先导入影像文件或DXF文件。")
            return False
        return True

    def on_start(self) -> None:
        """开始识别（识别+出图+Excel+图表）"""
        if not self._guard_can_run():
            return
        self._run_full_pipeline()

    def on_export(self) -> None:
        if not self._guard_can_run():
            return
        self._run_full_pipeline()

    def on_clear(self) -> None:
        self.image_paths = []
        self.cad_paths = []
        self.excel_path = None
        self.prop_df = None
        self._refresh_file_labels()
        self.txt_log.clear()
        self._features = []
        self._last_img_bgr = None
        self._ledger_df = None
        self._abnormal_df = None
        self._summary_df = None
        self._charts = {}
        self.list_features.clear()
        self.lab_preview.setText("识别预览将在此显示")
        self.lab_chart1.setText("等待生成图表...")
        self._log("数据已清空。")

    def on_open_out(self) -> None:
        open_in_explorer(self.out_dir)

    # -------------------------
    # 运行后台任务
    # -------------------------
    def _run_full_pipeline(self) -> None:
        params = self._get_params()

        # 输出目录允许后续扩展：按项目名分目录
        ensure_dir(self.out_dir)

        self._log("开始识别（V2.0增强算法）...")
        self._set_buttons_enabled(False)
        self.progress.setVisible(True)
        self._toast("正在识别处理中，请稍候…")

        self._worker = Worker(
            image_paths=list(self.image_paths),
            cad_paths=list(self.cad_paths),
            params=params,
            prop_df=self.prop_df,
            out_dir=self.out_dir,
        )
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_worker_finished(self, out: dict) -> None:
        self._set_buttons_enabled(True)
        self.progress.setVisible(False)
        self._log("任务完成。")
        # V2.0：out 为对象包
        paths = out.get("paths", {})
        self._features = out.get("features", []) or []
        self._ledger_df = out.get("ledger_df")
        self._abnormal_df = out.get("abnormal_df")
        self._summary_df = out.get("summary_df")
        self._last_img_bgr = out.get("last_img")
        self._charts = out.get("charts", {}) or {}

        if paths:
            tips = "\n".join([f"- {k}: {v}" for k, v in paths.items() if isinstance(v, str)])
            if tips:
                self._log("输出文件：\n" + tips)

        self._refresh_features_list()
        self._refresh_preview_image()
        self._refresh_charts_view()

        self._toast("识别完成，已生成成果与图表。")
        QMessageBox.information(self, "完成", "识别与导出已完成。可在“统计分析”页预览图表。")

    def _on_worker_failed(self, err: str) -> None:
        self._set_buttons_enabled(True)
        self.progress.setVisible(False)
        self._log("任务失败。")
        QMessageBox.critical(self, "运行失败", err)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for b in [
            self.btn_import_images,
            self.btn_import_cad,
            self.btn_import_excel,
            self.btn_start,
            self.btn_export,
            self.btn_clear,
            self.btn_open_out,
            self.btn_edit,
            self.btn_case_popup,
        ]:
            b.setEnabled(enabled)

    # -------------------------
    # 结果预览 / 人工校正 / 案例
    # -------------------------
    def _refresh_features_list(self) -> None:
        self.list_features.clear()
        for i, f in enumerate(self._features):
            txt = f"{i+1:03d} | 房号:{f.number or '-'} | 面积:{f.area_m2}㎡ | 源:{Path(f.source_path or '').name}"
            it = QListWidgetItem(txt)
            self.list_features.addItem(it)

    def _render_overlay(self, img_bgr: np.ndarray, features: List[HouseFeature]) -> QImage:
        # 用 OpenCV 叠加绘制，速度快
        vis = img_bgr.copy()
        for f in features[:800]:
            if f.contour_xy and len(f.contour_xy) >= 3:
                pts = np.array(f.contour_xy, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                x, y, w, h = f.bbox_xywh
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = int(f.centroid_xy[0]), int(f.centroid_xy[1])
            label = f"{f.number}"
            cv2.putText(vis, label, (max(0, cx - 10), max(0, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w = vis_rgb.shape[:2]
        qimg = QImage(vis_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return qimg.copy()

    def _refresh_preview_image(self) -> None:
        if self._last_img_bgr is None or self._last_img_bgr.size == 0:
            self.lab_preview.setText("无影像底图可预览（仅DXF识别时可后续增加CAD底图渲染）")
            return
        qimg = self._render_overlay(self._last_img_bgr, self._features)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.lab_preview.width(), self.lab_preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lab_preview.setPixmap(pix)

    def on_feature_double_click(self, item: QListWidgetItem) -> None:
        # 预留：定位/高亮某个目标（V2.0先不做复杂交互）
        self._toast("已选中目标，可点击“人工校正”进行点位编辑。")

    def on_manual_edit(self) -> None:
        if not self._features:
            QMessageBox.information(self, "提示", "请先完成识别，生成目标列表后再进行人工校正。")
            return
        idx = self.list_features.currentRow()
        if idx < 0:
            idx = 0
        f = self._features[idx]
        if not f.contour_xy:
            QMessageBox.information(self, "提示", "该目标缺少轮廓点数据，无法进入点位编辑。")
            return

        qimg = None
        if self._last_img_bgr is not None and self._last_img_bgr.size:
            rgb = cv2.cvtColor(self._last_img_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        dlg = ContourEditorDialog(qimg, f.contour_xy, parent=self)
        if dlg.exec() == QDialog.Accepted:
            f.contour_xy = dlg.get_contour_xy()
            self._toast("轮廓已更新。")
            self._refresh_preview_image()

    def on_show_cases(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("案例示例")
        dlg.resize(1100, 720)

        base_dir = str(ensure_dir(Path.cwd() / "assets" / "cases"))
        cases = ensure_case_assets(base_dir)

        tabs = QTabWidget(dlg)
        for c in cases:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(10)

            row = QHBoxLayout()
            lab1 = QLabel()
            lab2 = QLabel()
            lab1.setAlignment(Qt.AlignCenter)
            lab2.setAlignment(Qt.AlignCenter)
            lab1.setStyleSheet("background:white; border:1px solid #E3EAF6; border-radius:14px;")
            lab2.setStyleSheet("background:white; border:1px solid #E3EAF6; border-radius:14px;")

            p1 = QPixmap(c.orig_path).scaled(520, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p2 = QPixmap(c.result_path).scaled(520, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lab1.setPixmap(p1)
            lab2.setPixmap(p2)
            row.addWidget(lab1, 1)
            row.addWidget(lab2, 1)
            lay.addLayout(row, 1)

            desc = QLabel(c.desc)
            desc.setWordWrap(True)
            desc.setStyleSheet("color:#3A4A66;")
            lay.addWidget(desc)

            tabs.addTab(w, c.title)

        root = QVBoxLayout(dlg)
        root.addWidget(tabs, 1)
        dlg.exec()

    def _refresh_charts_view(self) -> None:
        # PNG：展示第一张图（V2.0简化），后续可做多图轮播/网格
        if not self._charts:
            self.lab_chart1.setText("未生成图表（请先运行识别）。")
            return
        # 优先展示饼图
        png = self._charts.get("pie_type_png") or next((v for k, v in self._charts.items() if k.endswith("_png")), None)
        if png and Path(png).exists():
            pix = QPixmap(png).scaled(980, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lab_chart1.setPixmap(pix)
        else:
            self.lab_chart1.setText("图表文件不存在。")

        html = self._charts.get("charts_html")
        if html and self.web is not None and Path(html).exists():
            self.lab_html_tip.setText(f"已生成：{html}")
            self.web.load(f"file:///{Path(html).resolve().as_posix()}")
        elif html and Path(html).exists():
            self.lab_html_tip.setText(f"已生成HTML（未启用内嵌预览）：{html}")


