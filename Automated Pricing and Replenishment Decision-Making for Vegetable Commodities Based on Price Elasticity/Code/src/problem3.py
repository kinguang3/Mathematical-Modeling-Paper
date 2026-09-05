import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

from config import CATEGORIES, FIGURE_DIR, MIN_ITEMS, MAX_ITEMS, MIN_DISPLAY_KG, COVERAGE


def build_candidates(sales, wholesale, goods, loss_item, pricing):
    end=pd.Timestamp("2023-06-30"); start=pd.Timestamp("2023-06-24")
    s=sales[(sales["销售日期"]>=start)&(sales["销售日期"]<=end)].copy()
    daily=s.groupby(["单品编码","销售日期"])["销量(千克)"].sum()
    avg=daily.groupby(level=0).mean().rename("周日均销量")
    base=avg.reset_index().merge(goods[["单品编码","单品名称","分类名称"]],on="单品编码",how="left")
    # 只保留最近一周确有销量的候选品种；论文为49个。
    base=base.dropna(subset=["分类名称"])
    # 论文口径：周六放大系数使用完整历史销售期的周六均值/周一至周五均值，
    # 而不是仅使用 2023-06-24 至 06-30 这一周，避免单周偶然波动造成系数失真。
    cat_daily=sales.groupby(["销售日期","分类名称"])["销量(千克)"].sum().unstack(fill_value=0).reindex(columns=CATEGORIES,fill_value=0)
    sat=cat_daily[cat_daily.index.dayofweek==5].mean()
    weekday=cat_daily[cat_daily.index.dayofweek<5].mean()
    factor=(sat/weekday).replace([np.inf,-np.inf],np.nan).fillna(1.0).to_dict()
    base["周六放大系数"]=base["分类名称"].map(factor).fillna(1.0); base["预测需求(kg)"]=base["周日均销量"]*base["周六放大系数"]
    c=wholesale[wholesale["日期"].dt.month==6].groupby("单品编码")["批发价格(元/千克)"].mean().rename("批发价")
    loss=dict(zip(loss_item["单品编码"], loss_item["损耗率(%)"]))
    base=base.join(c,on="单品编码"); base["损耗率(%)"]=base["单品编码"].map(loss)
    base["损耗率(%)"]=base["损耗率(%)"].fillna(0)
    pm=dict(zip(pricing["品类"],pricing["最优加成率"]))
    base["加成率"]=base["分类名称"].map(pm); base["售价(元/kg)"]=base["批发价"]*(1+base["加成率"])
    base["补货量(kg)"]=np.maximum(base["预测需求(kg)"]/(1-base["损耗率(%)"]/100),MIN_DISPLAY_KG)
    base["利润(元)"]=base["售价(元/kg)"]*base["预测需求(kg)"]-base["批发价"]*base["补货量(kg)"]
    base=base.dropna(subset=["批发价","利润(元)"]).copy()
    return base, pd.Series(factor)


def solve(candidates, category_forecast):
    n=len(candidates); cats=candidates["分类名称"].values
    c=-candidates["利润(元)"].to_numpy()
    A=lil_matrix((2+len(CATEGORIES),n),dtype=float)
    # 单品数上下界
    A[0,:]=1; A[1,:]=1
    for j,cat in enumerate(CATEGORIES): A[2+j,:]=(cats==cat).astype(float)*candidates["补货量(kg)"].to_numpy()
    lb=np.r_[MIN_ITEMS,-np.inf*np.ones(1+len(CATEGORIES))]
    ub=np.r_[np.inf,MAX_ITEMS,np.inf*np.ones(len(CATEGORIES))]
    # 更方便：两个单品数约束分别使用 [MIN,MAX]，覆盖约束使用 [0.75D, inf]
    lower=np.r_[MIN_ITEMS, -np.inf, [COVERAGE*category_forecast.get(cat,0) for cat in CATEGORIES]]
    upper=np.r_[np.inf, MAX_ITEMS, [np.inf]*len(CATEGORIES)]
    cons=LinearConstraint(A.tocsr(),lower,upper)
    res=milp(c,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=cons,options={"time_limit":120,"mip_rel_gap":1e-7})
    if not res.success:
        raise RuntimeError(f"MILP求解失败: {res.message}")
    out=candidates.copy(); out["是否选中"]=(res.x>0.5).astype(int); chosen=out[out["是否选中"]==1].copy()
    return chosen,res


def summary(chosen, category_forecast):
    rows=[]
    for cat in CATEGORIES:
        z=chosen[chosen["分类名称"]==cat]
        q=z["补货量(kg)"].sum(); d=float(category_forecast.get(cat,0)); rows.append({"品类":cat,"选中单品数":len(z),"补货量(kg)":q,"预测需求(kg)":d,"补货量覆盖需求比例(%)":100*q/d if d else np.nan})
    return pd.DataFrame(rows)


def sensitivity(candidates, category_forecast):
    base_profit=candidates["利润(元)"].sum() # 仅用于记录，不等于最优解利润
    rows=[]
    # 快速情景：对需求、成本、损耗率做参数扰动并重新MILP。
    for name,factor,kind in [("需求+10%",1.1,"demand"),("需求-10%",.9,"demand"),("成本+10%",1.1,"cost"),("成本-10%",.9,"cost"),("损耗率+10%",1.1,"loss"),("损耗率-10%",.9,"loss")]:
        x=candidates.copy(); d=category_forecast.copy()
        if kind=="demand": x["预测需求(kg)"]*=factor; x["补货量(kg)"]=np.maximum(x["预测需求(kg)"]/(1-x["损耗率(%)"]/100),MIN_DISPLAY_KG); x["利润(元)"]=x["售价(元/kg)"]*x["预测需求(kg)"]-x["批发价"]*x["补货量(kg)"]; d=d*factor
        elif kind=="cost": x["批发价"]*=factor; x["售价(元/kg)"]=x["批发价"]*(1+x["加成率"]); x["利润(元)"]=x["售价(元/kg)"]*x["预测需求(kg)"]-x["批发价"]*x["补货量(kg)"]
        else: x["损耗率(%)"]=np.clip(x["损耗率(%)"]*factor,0,95); x["补货量(kg)"]=np.maximum(x["预测需求(kg)"]/(1-x["损耗率(%)"]/100),MIN_DISPLAY_KG); x["利润(元)"]=x["售价(元/kg)"]*x["预测需求(kg)"]-x["批发价"]*x["补货量(kg)"]
        chosen,_=solve(x,d); profit=chosen["利润(元)"].sum(); rows.append({"情景":name,"总利润(元)":profit})
    # 基准情景
    chosen,_=solve(candidates,category_forecast); bp=chosen["利润(元)"].sum();
    for r in rows: r["相对变化(%)"]=(r["总利润(元)"]/bp-1)*100
    rows.insert(0,{"情景":"基准","总利润(元)":bp,"相对变化(%)":0.0})
    return pd.DataFrame(rows)


def save_figures(summary_df):
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(summary_df["品类"],summary_df["补货量覆盖需求比例(%)"]); ax.axhline(100,linewidth=1,linestyle="--"); ax.set_ylabel("覆盖比例（%）"); ax.set_title("问题三各品类需求覆盖比例"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图10_品类需求覆盖比例.png",dpi=180); plt.close(fig)
