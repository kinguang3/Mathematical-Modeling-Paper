import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import linregress
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import CATEGORIES, FIGURE_DIR, FORECAST_START, FORECAST_END

LOSS_CATEGORY_DEFAULT = {"花菜类":15.51,"水生根茎类":13.65,"花叶类":12.83,"食用菌":9.45,"辣椒类":9.24,"茄类":6.68}


def forecast_categories(daily):
    train_end = pd.Timestamp("2023-05-31")
    valid_end = pd.Timestamp("2023-06-30")
    metrics=[]; forecasts={}
    for c in CATEGORIES:
        y=daily[c].asfreq("D").fillna(0)
        train=y.loc[:train_end]
        valid=y.loc[train_end+pd.Timedelta(days=1):valid_end]
        model=ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated").fit(optimized=True, use_brute=False)
        pred=model.forecast(len(valid))
        err=pred.values-valid.values
        mae=np.mean(np.abs(err)); rmse=np.sqrt(np.mean(err**2)); mape=np.mean(np.abs(err/np.where(valid.values==0,1e-9,valid.values)))*100
        metrics.append({"品类":c,"MAE(kg)":mae,"RMSE(kg)":rmse,"MAPE(%)":mape})
        full_model=ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated").fit(optimized=True, use_brute=False)
        forecasts[c]=full_model.forecast(7)
    fc=pd.DataFrame(forecasts, index=pd.date_range(FORECAST_START,FORECAST_END))
    return fc, pd.DataFrame(metrics)


def price_elasticity(sales):
    rows=[]
    for c in CATEGORIES:
        x=sales.loc[sales["分类名称"]==c].copy()
        g=x.groupby("销售日期").agg(Q=("销量(千克)","sum"), P=("销售单价(元/千克)","mean"))
        g=g[(g.Q>0)&(g.P>0)]
        r=linregress(np.log(g.P), np.log(g.Q))
        rows.append({"品类":c,"价格弹性 ε":r.slope,"t统计量":r.slope/r.stderr,"p值":r.pvalue,"R²":r.rvalue**2})
    return pd.DataFrame(rows)


def category_costs(wholesale, loss_cat):
    cost=(wholesale.groupby("单品编码")["批发价格(元/千克)"].mean().rename("采购成本"))
    # 以分类编码对应的小分类名称作为优先来源；若名称无法映射，使用论文给出的类别均值。
    loss=dict(zip(loss_cat["小分类名称"], loss_cat.iloc[:,-1]))
    return cost, {c: float(loss.get(c, LOSS_CATEGORY_DEFAULT[c]))/100 for c in CATEGORIES}


def _current_markup(sales, wholesale, goods):
    # sales 已经在 prepare_sales 中关联了分类名称；这里不要引用未定义的局部变量 s。
    s = sales
    if "分类名称" not in s.columns:
        s = s.merge(goods[["单品编码", "分类名称"]], on="单品编码", how="left", validate="many_to_one")
    daily=s.groupby(["销售日期","分类名称"]).agg(P=("销售单价(元/千克)","mean"), Q=("销量(千克)","sum")).reset_index()
    w=wholesale.groupby(["日期","单品编码"])["批发价格(元/千克)"].mean().reset_index()
    z=s.merge(w,left_on=["销售日期","单品编码"],right_on=["日期","单品编码"],how="left")
    z=z.dropna(subset=["批发价格(元/千克)"])
    z["markup"]=z["销售单价(元/千克)"]/z["批发价格(元/千克)"]-1
    return z.groupby("分类名称")["markup"].quantile(.5).to_dict()


def optimize_pricing(fc, elasticity, wholesale, loss_cat, sales=None, goods=None):
    cost_item=wholesale.groupby("单品编码")["批发价格(元/千克)"].mean()
    # 品类采购成本：所有该品类单品历史批发价均值
    if goods is not None:
        tmp=wholesale.merge(goods[["单品编码","分类名称"]],on="单品编码",how="left")
        cat_cost=tmp.groupby("分类名称")["批发价格(元/千克)"].mean().reindex(CATEGORIES)
    else:
        cat_cost=pd.Series(index=CATEGORIES,dtype=float)
    loss=dict(zip(loss_cat["小分类名称"], loss_cat.iloc[:,-1]/100))
    eps=dict(zip(elasticity["品类"], elasticity["价格弹性 ε"]))
    current=_current_markup(sales,wholesale,goods) if sales is not None and goods is not None else {}
    rows=[]
    opt_markup={}
    for c in CATEGORIES:
        c0=float(cat_cost[c]); w=float(loss.get(c,LOSS_CATEGORY_DEFAULT[c]/100)); e=float(eps[c])
        # 论文口径：历史加成率95%分位作为上界；缺失时使用 2.0。
        upper=max(0.0, float(pd.Series(current.get(c,0)).quantile(.95))) if c in current else 2.0
        # 更稳定地从所有日数据重新计算95分位
        if sales is not None and goods is not None:
            z=sales[sales["分类名称"]==c].merge(wholesale[["日期","单品编码","批发价格(元/千克)"]],left_on=["销售日期","单品编码"],right_on=["日期","单品编码"],how="inner")
            mm=z.loc[z["批发价格(元/千克)"]>0,"销售单价(元/千克)"]/z.loc[z["批发价格(元/千克)"]>0,"批发价格(元/千克)"]-1
            if len(mm): upper=float(mm.quantile(.95))
        upper=max(0.05, min(upper, 5.0))
        q0=fc[c].values
        # 直接最大化7日总利润，避免把边界问题误判为解析内点。
        def total_profit(m):
            p=c0*(1+m); base=1.0
            # 用预测需求作为基准，历史平均价格作为参考价格。
            p0=float(c0*(1+max(current.get(c,0),0)))
            demand=q0*np.power(p/max(p0,1e-9),e)
            q=demand/(1-w)
            return np.sum(p*demand-c0*q)
        res=minimize_scalar(lambda m:-total_profit(m),bounds=(0,upper),method="bounded",options={"xatol":1e-6})
        m=float(res.x); opt_markup[c]=m
        rows.append({"品类":c,"采购成本(元/kg)":c0,"损耗率":w,"价格弹性":e,"当前加成率":current.get(c,np.nan),"最优加成率":m,"最优售价(元/kg)":c0*(1+m),"加成上界":upper})
    opt=pd.DataFrame(rows)
    result=[]
    for dt,row in fc.iterrows():
        for c in CATEGORIES:
            r=opt.loc[opt["品类"]==c].iloc[0]; c0=r["采购成本(元/kg)"]; w=r["损耗率"]; m=r["最优加成率"]; p0=c0*(1+max(r["当前加成率"] if pd.notna(r["当前加成率"]) else 0,0)); p=c0*(1+m); e=r["价格弹性"]
            qd=float(row[c])*float((p/p0)**e); replenish=qd/(1-w); profit=p*qd-c0*replenish
            result.append({"日期":dt,"品类":c,"预测需求(kg)":qd,"建议补货量(kg)":replenish,"建议售价(元/kg)":p,"加成率":m,"预计利润(元)":profit})
    return opt,pd.DataFrame(result)


def save_figures(fc, elasticity, pricing):
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
    fig,ax=plt.subplots(figsize=(11,5));
    for c in CATEGORIES: ax.plot(fc.index,fc[c],marker="o",label=c)
    ax.set_title("各品类 2023-07-01 至 07-07 日需求预测"); ax.set_ylabel("预测需求（kg）"); ax.legend(ncol=3); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图7_一周需求预测.png",dpi=180); plt.close(fig)
    # 图8 每类一张，避免把六个散点图压成难以阅读的一张图；论文图号对应主图文件。
    # 主图使用所有日均观测，并按类别分色。
    fig,ax=plt.subplots(figsize=(9,6))
    # elasticity 仅有参数，因此这里从结果生成拟合关系的摘要图；详细散点另存 CSV。
    ax.bar(elasticity["品类"],elasticity["价格弹性 ε"]); ax.axhline(0,linewidth=1); ax.set_ylabel("需求价格弹性 ε"); ax.set_title("各品类需求价格弹性估计"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图8_价格弹性估计.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(len(pricing)); width=.36; ax.bar(x-width/2,pricing["当前加成率"],width,label="当前"); ax.bar(x+width/2,pricing["最优加成率"],width,label="最优"); ax.set_xticks(x,pricing["品类"],rotation=35); ax.set_ylabel("成本加成率"); ax.set_title("各品类当前与最优成本加成率对比"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURE_DIR/"图9_当前与最优加成率.png",dpi=180); plt.close(fig)
