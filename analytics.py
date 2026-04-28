"""
统计结果可视化（V2.0）

输出：
- PNG（matplotlib）
- HTML（pyecharts，可选优先）

并提供给 GUI 内嵌预览使用的路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_charts(ledger_df: pd.DataFrame, abnormal_df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    """
    生成图表文件并返回路径字典。

    图表：
    1) 房屋类型占比饼图
    2) 各楼栋面积柱状图
    3) 房屋数量统计图（按类型/用途）
    4) 楼层分布图（若存在“层数”字段）
    5) 异常数据统计图（按异常原因）
    """

    out = {}
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    # 1) 类型占比
    out["pie_type_png"] = str(base / "图表_房屋类型占比.png")
    _pie_type(ledger_df, out["pie_type_png"])

    # 2) 楼栋面积
    out["bar_building_area_png"] = str(base / "图表_楼栋面积.png")
    _bar_building_area(ledger_df, out["bar_building_area_png"])

    # 3) 数量统计
    out["bar_count_png"] = str(base / "图表_房屋数量统计.png")
    _bar_count(ledger_df, out["bar_count_png"])

    # 4) 楼层分布
    out["bar_floor_png"] = str(base / "图表_楼层分布.png")
    _bar_floor(ledger_df, out["bar_floor_png"])

    # 5) 异常统计
    out["bar_abnormal_png"] = str(base / "图表_异常统计.png")
    _bar_abnormal(abnormal_df, out["bar_abnormal_png"])

    # HTML（pyecharts 优先）
    html = _generate_pyecharts_html(ledger_df, abnormal_df, str(base / "图表汇总.html"))
    if html:
        out["charts_html"] = html

    return out


def _pie_type(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
    else:
        col = "房屋类别" if "房屋类别" in df.columns else ("用途" if "用途" in df.columns else None)
        if not col:
            ax.text(0.5, 0.5, "缺少类型字段（房屋类别/用途）", ha="center", va="center")
        else:
            s = df[col].astype(str).replace({"nan": ""}).str.strip()
            s = s[s != ""]
            if s.empty:
                ax.text(0.5, 0.5, "类型字段为空", ha="center", va="center")
            else:
                counts = s.value_counts().head(12)
                ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
                ax.set_title("房屋类型占比")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _bar_building_area(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    if df is None or df.empty or "楼栋号" not in df.columns:
        ax.text(0.5, 0.5, "缺少楼栋号或无数据", ha="center", va="center")
    else:
        s = df["楼栋号"].astype(str).replace({"nan": ""}).str.strip()
        area = pd.to_numeric(df.get("建筑面积(㎡)"), errors="coerce").fillna(0)
        tmp = pd.DataFrame({"楼栋号": s, "建筑面积": area})
        tmp = tmp[tmp["楼栋号"] != ""]
        if tmp.empty:
            ax.text(0.5, 0.5, "楼栋号为空", ha="center", va="center")
        else:
            g = tmp.groupby("楼栋号")["建筑面积"].sum().sort_values(ascending=False).head(30)
            ax.bar(g.index.astype(str), g.values)
            ax.set_title("各楼栋建筑面积汇总")
            ax.set_ylabel("面积(㎡)")
            ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _bar_count(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
    else:
        col = "房屋类别" if "房屋类别" in df.columns else ("用途" if "用途" in df.columns else None)
        if not col:
            ax.text(0.5, 0.5, "缺少统计字段（房屋类别/用途）", ha="center", va="center")
        else:
            s = df[col].astype(str).replace({"nan": ""}).str.strip()
            s = s[s != ""]
            g = s.value_counts().head(30)
            ax.bar(g.index.astype(str), g.values)
            ax.set_title("房屋数量统计")
            ax.set_ylabel("数量")
            ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _bar_floor(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    if df is None or df.empty or "层数" not in df.columns:
        ax.text(0.5, 0.5, "缺少“层数”字段或无数据", ha="center", va="center")
    else:
        floors = pd.to_numeric(df["层数"], errors="coerce").dropna().astype(int)
        if floors.empty:
            ax.text(0.5, 0.5, "层数为空/不可解析", ha="center", va="center")
        else:
            g = floors.value_counts().sort_index()
            ax.plot(g.index, g.values, marker="o")
            ax.set_title("楼层分布")
            ax.set_xlabel("层数")
            ax.set_ylabel("数量")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _bar_abnormal(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    if df is None or df.empty or "异常原因" not in df.columns:
        ax.text(0.5, 0.5, "无异常数据", ha="center", va="center")
    else:
        s = df["异常原因"].astype(str).replace({"nan": ""}).str.strip()
        s = s[s != ""]
        if s.empty:
            ax.text(0.5, 0.5, "无异常原因", ha="center", va="center")
        else:
            # 将“；”分隔的原因拆分统计
            parts = []
            for x in s.tolist():
                parts.extend([p.strip() for p in x.split("；") if p.strip()])
            g = pd.Series(parts).value_counts().head(30)
            ax.bar(g.index.astype(str), g.values)
            ax.set_title("异常数据统计")
            ax.set_ylabel("数量")
            ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _generate_pyecharts_html(df: pd.DataFrame, abnormal_df: pd.DataFrame, out_html: str) -> Optional[str]:
    try:
        from pyecharts.charts import Bar, Page, Pie  # type: ignore
        from pyecharts import options as opts  # type: ignore
    except Exception:
        return None

    page = Page(page_title="房屋识别统计图表")

    # 饼图
    col = "房屋类别" if "房屋类别" in df.columns else ("用途" if "用途" in df.columns else None)
    if col and df is not None and not df.empty:
        s = df[col].astype(str).replace({"nan": ""}).str.strip()
        s = s[s != ""]
        counts = s.value_counts().head(12)
        pie = (
            Pie()
            .add("", [list(z) for z in zip(counts.index.astype(str), counts.values.tolist())])
            .set_global_opts(title_opts=opts.TitleOpts(title="房屋类型占比"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
        )
        page.add(pie)

    # 楼栋面积柱状
    if "楼栋号" in df.columns and "建筑面积(㎡)" in df.columns and df is not None and not df.empty:
        s = df["楼栋号"].astype(str).replace({"nan": ""}).str.strip()
        area = pd.to_numeric(df["建筑面积(㎡)"], errors="coerce").fillna(0)
        tmp = pd.DataFrame({"楼栋号": s, "建筑面积": area})
        tmp = tmp[tmp["楼栋号"] != ""]
        if not tmp.empty:
            g = tmp.groupby("楼栋号")["建筑面积"].sum().sort_values(ascending=False).head(30)
            bar = (
                Bar()
                .add_xaxis(g.index.astype(str).tolist())
                .add_yaxis("面积(㎡)", [round(float(x), 3) for x in g.values.tolist()])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="各楼栋建筑面积汇总"),
                    datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                )
            )
            page.add(bar)

    # 异常统计
    if abnormal_df is not None and not abnormal_df.empty and "异常原因" in abnormal_df.columns:
        s = abnormal_df["异常原因"].astype(str).replace({"nan": ""}).str.strip()
        parts = []
        for x in s.tolist():
            parts.extend([p.strip() for p in x.split("；") if p.strip()])
        if parts:
            g = pd.Series(parts).value_counts().head(30)
            bar2 = (
                Bar()
                .add_xaxis(g.index.astype(str).tolist())
                .add_yaxis("数量", [int(x) for x in g.values.tolist()])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="异常原因统计"),
                    datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                )
            )
            page.add(bar2)

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.render(str(out_path))
    return str(out_path)

