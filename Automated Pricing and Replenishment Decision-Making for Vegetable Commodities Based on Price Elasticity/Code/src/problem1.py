from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CATEGORIES, FIGURE_DIR


def category_daily(sales):
    d = (sales.groupby(["销售日期", "分类名称"], observed=True)["销量(千克)"]
         .sum().unstack(fill_value=0).reindex(columns=CATEGORIES, fill_value=0))
    return d.sort_index()


def build_tables(sales, wholesale=None):
    daily = category_daily(sales)
    stats = []
    revenue_daily = sales.assign(_revenue=sales["销量(千克)"] * sales["销售单价(元/千克)"]).groupby(["销售日期", "分类名称"], observed=True)["_revenue"].sum().unstack(fill_value=0).reindex(columns=CATEGORIES, fill_value=0)
    for c in CATEGORIES:
        # 论文统计口径：某品类无销售的日期不参与该品类日销量分布统计。
        x = daily[c]
        x_nonzero = x[x > 0]
        stats.append({
            "品类": c,
            "日均销量(kg)": x_nonzero.mean(),
            "中位数(kg)": x_nonzero.median(),
            "标准差(kg)": x_nonzero.std(ddof=1),
            "变异系数CV": x_nonzero.std(ddof=1) / x_nonzero.mean() if x_nonzero.mean() else np.nan,
            "最小值(kg)": x_nonzero.min(),
            "最大值(kg)": x_nonzero.max(),
            "日均销售额(元)": revenue_daily[c][revenue_daily[c] > 0].mean(),
        })
    t2 = pd.DataFrame(stats)

    weekday = daily.groupby(daily.index.dayofweek).mean().reindex(range(7))
    t3 = pd.DataFrame([weekday.sum(axis=1).values], columns=["周一","周二","周三","周四","周五","周六","周日"]).T.reset_index()
    t3.columns = ["项目", "平均日销量(kg)"]

    month = daily.groupby(daily.index.month).mean().sum(axis=1)
    t4 = pd.DataFrame({"月份": range(1,13), "平均日销量(kg)": [month.get(i, np.nan) for i in range(1,13)]})

    # Pearson 相关性按共同存在销售的日期计算，避免无销售日的结构性零值人为拉低相关性。
    corr = daily.loc[(daily > 0).all(axis=1)].corr(method="pearson")
    t5 = corr.copy()
    return {"table2": t2, "table3": t3, "table4": t4, "table5": t5, "daily": daily}


def save_figures(sales, tables):
    daily = tables["daily"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1
    ax = daily.rolling(30, min_periods=1).mean().plot(figsize=(11,5))
    ax.set_title("各品类日销量趋势（30日移动平均）"); ax.set_xlabel("日期"); ax.set_ylabel("销量（kg）")
    ax.grid(alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(FIGURE_DIR/"图1_各品类日销量趋势.png", dpi=180); plt.close(ax.figure)

    # 图2
    fig, ax = plt.subplots(figsize=(10,5)); ax.boxplot([daily[c].values for c in CATEGORIES], labels=CATEGORIES, showfliers=False)
    ax.set_title("各品类日销量分布箱线图"); ax.set_ylabel("销量（kg）"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图2_各品类日销量箱线图.png", dpi=180); plt.close(fig)

    # 图3
    item = sales.groupby(["单品编码", "分类名称"])["销量(千克)"].sum().sort_values(ascending=False).head(20).reset_index()
    # 关联商品名称；如果名称缺失则回退为单品编码，避免 KeyError。
    name_map = sales[["单品编码"] + (["单品名称"] if "单品名称" in sales.columns else [])].drop_duplicates("单品编码")
    if "单品名称" in name_map.columns:
        item = item.merge(name_map, on="单品编码", how="left")
        labels = item["单品名称"].fillna(item["单品编码"].astype(str))
    else:
        labels = item["单品编码"].astype(str)
    fig, ax = plt.subplots(figsize=(10,7)); ax.barh(labels.iloc[::-1], item["销量(千克)"].values[::-1]); ax.set_title("单品销量 TOP20"); ax.set_xlabel("累计销量（kg）"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图3_单品销量TOP20.png", dpi=180); plt.close(fig)

    # 图4
    fig, ax = plt.subplots(figsize=(9,5)); ax.bar(tables["table3"]["项目"], tables["table3"]["平均日销量(kg)"]); ax.set_title("全品类日销量的星期效应"); ax.set_ylabel("平均日销量（kg）"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图4_星期效应.png", dpi=180); plt.close(fig)

    # 图5
    fig, ax = plt.subplots(figsize=(10,5)); ax.plot(tables["table4"]["月份"], tables["table4"]["平均日销量(kg)"], marker="o"); ax.set_xticks(range(1,13)); ax.set_xlabel("月份"); ax.set_ylabel("平均日销量（kg）"); ax.set_title("全品类日销量的月度效应"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图5_月度效应.png", dpi=180); plt.close(fig)

    # 图6
    fig, ax = plt.subplots(figsize=(8,6)); im=ax.imshow(tables["table5"].values, vmin=-1, vmax=1, cmap="coolwarm"); fig.colorbar(im, ax=ax, label="Pearson r"); ax.set_xticks(range(6), CATEGORIES, rotation=35, ha="right"); ax.set_yticks(range(6), CATEGORIES); ax.set_title("品类日销量 Pearson 相关系数矩阵"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图6_品类相关系数热力图.png", dpi=180); plt.close(fig)

    return item
