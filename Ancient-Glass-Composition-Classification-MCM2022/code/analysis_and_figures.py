from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SOURCE_DIR = Path(
    r"C:\Users\13956\Downloads\第4轮模拟_第六大组_2022年高教杯C题 古代玻璃制品的成分分析与鉴别"
    r"\第4轮模拟_第六大组_2022年高教杯C题 古代玻璃制品的成分分析与鉴别"
)
XLSX = SOURCE_DIR / "附件.xlsx"

RNG = np.random.default_rng(20220818)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial",
    "DejaVu Sans",
]
plt.rcParams.update({
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "highk": "#0F4D92",
    "lead": "#B64342",
    "unweathered": "#42949E",
    "weathered": "#E28E2C",
    "severe": "#9A4D8E",
    "neutral": "#767676",
    "light": "#D8D8D8",
    "green": "#2E9E44",
}

COMPONENTS_CN = [
    "二氧化硅(SiO2)",
    "氧化钠(Na2O)",
    "氧化钾(K2O)",
    "氧化钙(CaO)",
    "氧化镁(MgO)",
    "氧化铝(Al2O3)",
    "氧化铁(Fe2O3)",
    "氧化铜(CuO)",
    "氧化铅(PbO)",
    "氧化钡(BaO)",
    "五氧化二磷(P2O5)",
    "氧化锶(SrO)",
    "氧化锡(SnO2)",
    "二氧化硫(SO2)",
]
SHORT = {
    "二氧化硅(SiO2)": "SiO2",
    "氧化钠(Na2O)": "Na2O",
    "氧化钾(K2O)": "K2O",
    "氧化钙(CaO)": "CaO",
    "氧化镁(MgO)": "MgO",
    "氧化铝(Al2O3)": "Al2O3",
    "氧化铁(Fe2O3)": "Fe2O3",
    "氧化铜(CuO)": "CuO",
    "氧化铅(PbO)": "PbO",
    "氧化钡(BaO)": "BaO",
    "五氧化二磷(P2O5)": "P2O5",
    "氧化锶(SrO)": "SrO",
    "氧化锡(SnO2)": "SnO2",
    "二氧化硫(SO2)": "SO2",
}
REV_SHORT = {v: k for k, v in SHORT.items()}


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def close_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sums = x.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Composition row with non-positive total")
    return x / sums * 100.0


def multiplicative_zero_replacement(x: np.ndarray, delta_cap: float = 0.0005) -> np.ndarray:
    """Replace zeros in closed fractions, preserving closure row by row."""
    x = close_rows(np.asarray(x, dtype=float)) / 100.0
    out = np.empty_like(x)
    for i, row in enumerate(x):
        zero = row <= 0
        k = int(zero.sum())
        if k == 0:
            out[i] = row
            continue
        min_pos = row[~zero].min()
        delta = min(delta_cap, 0.65 * min_pos, 0.9 / k)
        out[i, zero] = delta
        out[i, ~zero] = row[~zero] * ((1.0 - k * delta) / row[~zero].sum())
    return out


def clr(x: np.ndarray) -> np.ndarray:
    xr = multiplicative_zero_replacement(x)
    lx = np.log(xr)
    return lx - lx.mean(axis=1, keepdims=True)


def inv_clr(z: np.ndarray) -> np.ndarray:
    ez = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return ez / ez.sum(axis=-1, keepdims=True) * 100.0


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Positive means values in b (weathered) tend to exceed a (unweathered).
    return float((np.sum(b[:, None] > a[None, :]) - np.sum(b[:, None] < a[None, :])) / (len(a) * len(b)))


def cramer_v(table: np.ndarray) -> float:
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt((chi2 / n) / min(table.shape[0] - 1, table.shape[1] - 1)))


def permutation_chi2_p(categories: pd.Series, weather: pd.Series, n_perm: int = 5000) -> tuple[float, float]:
    cat_codes, _ = pd.factorize(categories.astype(str), sort=True)
    weather_codes = (weather.astype(str).to_numpy() == "风化").astype(int)
    n_cat = int(cat_codes.max() + 1)
    observed_table = np.bincount(cat_codes * 2 + weather_codes, minlength=n_cat * 2).reshape(n_cat, 2)
    observed = chi2_contingency(observed_table, correction=False)[0]
    exceed = 0
    for _ in range(n_perm):
        wp = RNG.permutation(weather_codes)
        table = np.bincount(cat_codes * 2 + wp, minlength=n_cat * 2).reshape(n_cat, 2)
        stat = chi2_contingency(table, correction=False)[0]
        exceed += stat >= observed - 1e-12
    return float(observed), float((exceed + 1) / (n_perm + 1))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info = pd.read_excel(XLSX, sheet_name="表单1", dtype={"文物编号": str})
    chem = pd.read_excel(XLSX, sheet_name="表单2", dtype={"文物采样点": str})
    unknown = pd.read_excel(XLSX, sheet_name="表单3", dtype={"文物编号": str})
    info["文物编号"] = info["文物编号"].str.zfill(2)
    info["颜色"] = info["颜色"].fillna("未记录")
    chem["artifact_id"] = chem["文物采样点"].str.extract(r"(\d{2})", expand=False)
    chem = chem.merge(info, left_on="artifact_id", right_on="文物编号", how="left", validate="many_to_one")
    chem[COMPONENTS_CN] = chem[COMPONENTS_CN].fillna(0.0).astype(float)
    chem["成分总和"] = chem[COMPONENTS_CN].sum(axis=1)
    chem["有效"] = chem["成分总和"].between(85, 105, inclusive="both")
    chem["采样点风化"] = np.where(
        chem["文物采样点"].str.contains("未风化"),
        "无风化",
        np.where(chem["文物采样点"].str.contains("严重风化"), "严重风化", chem["表面风化"]),
    )
    chem_valid = chem.loc[chem["有效"]].copy()
    chem_valid[COMPONENTS_CN] = close_rows(chem_valid[COMPONENTS_CN].to_numpy())
    unknown[COMPONENTS_CN] = unknown[COMPONENTS_CN].fillna(0.0).astype(float)
    unknown["成分总和"] = unknown[COMPONENTS_CN].sum(axis=1)
    unknown[COMPONENTS_CN] = close_rows(unknown[COMPONENTS_CN].to_numpy())
    return info, chem_valid, unknown


def artifact_balanced_weathering(chem: pd.DataFrame) -> pd.DataFrame:
    temp = chem.copy()
    temp["二元风化"] = np.where(temp["采样点风化"] == "无风化", "无风化", "风化")
    grouped = (
        temp.groupby(["artifact_id", "类型", "二元风化"], as_index=False)[COMPONENTS_CN]
        .mean()
    )
    grouped[COMPONENTS_CN] = close_rows(grouped[COMPONENTS_CN].to_numpy())
    return grouped


def association_analysis(info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    weather = info["表面风化"]
    for factor in ["类型", "纹饰", "颜色"]:
        tab = pd.crosstab(info[factor], weather)
        chi2, p_asym, dof, expected = chi2_contingency(tab, correction=False)
        _, p_perm = permutation_chi2_p(info[factor], weather)
        rows.append(
            {
                "因素": factor,
                "χ²": chi2,
                "df": dof,
                "渐近p": p_asym,
                "置换p": p_perm,
                "Cramér V": cramer_v(tab.to_numpy()),
                "最小期望频数": expected.min(),
            }
        )
    result = pd.DataFrame(rows)
    type_tab = pd.crosstab(info["类型"], info["表面风化"])
    odds, fisher_p = fisher_exact(type_tab.loc[["高钾", "铅钡"], ["风化", "无风化"]].to_numpy())
    # Odds returned for the first row; invert to express Pb-Ba relative to high-K.
    odds_pb_vs_hk = 1 / odds
    a = type_tab.loc["铅钡", "风化"]
    b = type_tab.loc["铅钡", "无风化"]
    c = type_tab.loc["高钾", "风化"]
    d = type_tab.loc["高钾", "无风化"]
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = math.exp(math.log(odds_pb_vs_hk) - 1.96 * se)
    hi = math.exp(math.log(odds_pb_vs_hk) + 1.96 * se)
    result.attrs["type_fisher_p"] = float(fisher_p)
    result.attrs["odds_ratio_pb_vs_hk"] = float(odds_pb_vs_hk)
    result.attrs["odds_ratio_ci"] = [float(lo), float(hi)]
    result.to_csv(TAB / "association_tests.csv", index=False, encoding="utf-8-sig")
    return result


def weathering_component_analysis(balanced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for glass_type in ["高钾", "铅钡"]:
        subset = balanced[balanced["类型"] == glass_type]
        un = subset[subset["二元风化"] == "无风化"]
        we = subset[subset["二元风化"] == "风化"]
        pvals = []
        temp_rows = []
        for comp in COMPONENTS_CN:
            a = un[comp].to_numpy()
            b = we[comp].to_numpy()
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            pvals.append(p)
            temp_rows.append(
                {
                    "类型": glass_type,
                    "成分": SHORT[comp],
                    "无风化n": len(a),
                    "风化n": len(b),
                    "无风化均值": a.mean(),
                    "风化均值": b.mean(),
                    "无风化中位数": np.median(a),
                    "风化中位数": np.median(b),
                    "均值差(风化-无风化)": b.mean() - a.mean(),
                    "Cliff_delta": cliffs_delta(a, b),
                    "p": p,
                }
            )
        qvals = bh_adjust(np.array(pvals))
        for row, q in zip(temp_rows, qvals):
            row["BH_q"] = q
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(TAB / "weathering_component_stats.csv", index=False, encoding="utf-8-sig")
    return result


def weathering_prediction(chem: pd.DataFrame, balanced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    deltas = {}
    for glass_type in ["高钾", "铅钡"]:
        s = balanced[balanced["类型"] == glass_type]
        z_un = clr(s.loc[s["二元风化"] == "无风化", COMPONENTS_CN].to_numpy())
        z_we = clr(s.loc[s["二元风化"] == "风化", COMPONENTS_CN].to_numpy())
        deltas[glass_type] = z_we.mean(axis=0) - z_un.mean(axis=0)

    weathered = chem[chem["采样点风化"] != "无风化"].copy()
    predictions = []
    for _, row in weathered.iterrows():
        x = row[COMPONENTS_CN].to_numpy(dtype=float)[None, :]
        z_pred = clr(x)[0] - deltas[row["类型"]]
        pred = inv_clr(z_pred[None, :])[0]
        record = {
            "文物采样点": row["文物采样点"],
            "artifact_id": row["artifact_id"],
            "类型": row["类型"],
            "采样点风化": row["采样点风化"],
        }
        record.update({SHORT[c]: float(v) for c, v in zip(COMPONENTS_CN, pred)})
        predictions.append(record)
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(TAB / "predicted_preweathering_compositions.csv", index=False, encoding="utf-8-sig")

    paired_rows = []
    for artifact_id in ["49", "50"]:
        sample_w = chem[(chem["artifact_id"] == artifact_id) & (chem["文物采样点"] == artifact_id)]
        sample_u = chem[(chem["artifact_id"] == artifact_id) & chem["文物采样点"].str.contains("未风化")]
        if sample_w.empty or sample_u.empty:
            continue
        glass_type = sample_w.iloc[0]["类型"]
        train = balanced[balanced["artifact_id"] != artifact_id]
        s = train[train["类型"] == glass_type]
        delta_loo = (
            clr(s.loc[s["二元风化"] == "风化", COMPONENTS_CN].to_numpy()).mean(axis=0)
            - clr(s.loc[s["二元风化"] == "无风化", COMPONENTS_CN].to_numpy()).mean(axis=0)
        )
        observed_w = sample_w.iloc[0][COMPONENTS_CN].to_numpy(dtype=float)[None, :]
        actual_u = sample_u[COMPONENTS_CN].mean(axis=0).to_numpy(dtype=float)
        actual_u = close_rows(actual_u[None, :])[0]
        predicted_u = inv_clr((clr(observed_w)[0] - delta_loo)[None, :])[0]
        naive_u = close_rows(
            s.loc[s["二元风化"] == "无风化", COMPONENTS_CN].mean(axis=0).to_numpy(dtype=float)[None, :]
        )[0]
        mae_model = float(np.mean(np.abs(predicted_u - actual_u)))
        mae_naive = float(np.mean(np.abs(naive_u - actual_u)))
        aitch_model = float(np.linalg.norm(clr(predicted_u[None, :])[0] - clr(actual_u[None, :])[0]))
        aitch_naive = float(np.linalg.norm(clr(naive_u[None, :])[0] - clr(actual_u[None, :])[0]))
        paired_rows.append(
            {
                "文物编号": artifact_id,
                "模型MAE": mae_model,
                "组均值MAE": mae_naive,
                "模型Aitchison距离": aitch_model,
                "组均值Aitchison距离": aitch_naive,
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(TAB / "paired_prediction_validation.csv", index=False, encoding="utf-8-sig")
    summary = {
        "paired_n": len(paired),
        "mean_model_mae": float(paired["模型MAE"].mean()),
        "mean_naive_mae": float(paired["组均值MAE"].mean()),
        "mean_model_aitchison": float(paired["模型Aitchison距离"].mean()),
        "mean_naive_aitchison": float(paired["组均值Aitchison距离"].mean()),
    }
    return pred_df, paired, summary


def classification_analysis(chem: pd.DataFrame, unknown: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    X = chem[COMPONENTS_CN].to_numpy()
    y = (chem["类型"] == "铅钡").astype(int).to_numpy()
    groups = chem["artifact_id"].to_numpy()
    pbo_idx = COMPONENTS_CN.index("氧化铅(PbO)")
    highk_pbo = X[y == 0, pbo_idx]
    lead_pbo = X[y == 1, pbo_idx]
    threshold = float((highk_pbo.max() + lead_pbo.min()) / 2)

    # Leave-one-artifact-out threshold validation. The threshold is recalculated without the held-out artifact.
    pred_logo = np.empty_like(y)
    fold_thresholds = []
    for group in np.unique(groups):
        train = groups != group
        test = groups == group
        thr = float((X[train & (y == 0), pbo_idx].max() + X[train & (y == 1), pbo_idx].min()) / 2)
        pred_logo[test] = (X[test, pbo_idx] > thr).astype(int)
        fold_thresholds.append(thr)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2022)
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=2022,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )
    rf_cv_pred = cross_val_predict(rf, X, y, groups=groups, cv=cv, method="predict", n_jobs=1)
    rf.fit(X, y)
    importances = pd.DataFrame(
        {"成分": [SHORT[c] for c in COMPONENTS_CN], "重要性": rf.feature_importances_}
    ).sort_values("重要性", ascending=False)
    importances.to_csv(TAB / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y, pred_logo)
    cm_rf = confusion_matrix(y, rf_cv_pred)
    metrics = {
        "threshold": threshold,
        "max_highk_pbo": float(highk_pbo.max()),
        "min_lead_pbo": float(lead_pbo.min()),
        "logo_accuracy": float(accuracy_score(y, pred_logo)),
        "logo_balanced_accuracy": float(balanced_accuracy_score(y, pred_logo)),
        "logo_f1": float(f1_score(y, pred_logo)),
        "logo_confusion": cm.tolist(),
        "threshold_cv_min": float(np.min(fold_thresholds)),
        "threshold_cv_max": float(np.max(fold_thresholds)),
        "rf_group5_accuracy": float(accuracy_score(y, rf_cv_pred)),
        "rf_group5_balanced_accuracy": float(balanced_accuracy_score(y, rf_cv_pred)),
        "rf_group5_f1": float(f1_score(y, rf_cv_pred)),
        "rf_group5_confusion": cm_rf.tolist(),
    }

    Xu = unknown[COMPONENTS_CN].to_numpy()
    prob_lead = rf.predict_proba(Xu)[:, 1]
    rule_pred = (Xu[:, pbo_idx] > threshold).astype(int)
    rf_pred = (prob_lead >= 0.5).astype(int)
    unknown_result = pd.DataFrame(
        {
            "样本": unknown["文物编号"],
            "表面风化": unknown["表面风化"],
            "PbO": Xu[:, pbo_idx],
            "K2O": Xu[:, COMPONENTS_CN.index("氧化钾(K2O)")],
            "BaO": Xu[:, COMPONENTS_CN.index("氧化钡(BaO)")],
            "SiO2": Xu[:, COMPONENTS_CN.index("二氧化硅(SiO2)")],
            "阈值规则": np.where(rule_pred == 1, "铅钡", "高钾"),
            "随机森林": np.where(rf_pred == 1, "铅钡", "高钾"),
            "P(铅钡)": prob_lead,
            "边界距离": np.abs(Xu[:, pbo_idx] - threshold),
        }
    )

    # Monte Carlo closure-aware component perturbation across increasing error amplitudes.
    amplitudes = np.arange(0.0, 0.61, 0.05)
    perturb_rows = []
    base_rule = rule_pred.copy()
    base_rf = rf_pred.copy()
    for amp in amplitudes:
        n_iter = 500 if amp > 0 else 1
        flip_rule = np.zeros(len(Xu), dtype=int)
        flip_rf = np.zeros(len(Xu), dtype=int)
        for _ in range(n_iter):
            factors = RNG.uniform(1 - amp, 1 + amp, size=Xu.shape)
            Xp = close_rows(np.maximum(Xu * factors, 0))
            pr = (Xp[:, pbo_idx] > threshold).astype(int)
            pf = rf.predict(Xp)
            flip_rule += pr != base_rule
            flip_rf += pf != base_rf
        for i, sid in enumerate(unknown["文物编号"]):
            perturb_rows.append(
                {
                    "扰动幅度": amp,
                    "样本": sid,
                    "阈值翻转率": flip_rule[i] / n_iter,
                    "随机森林翻转率": flip_rf[i] / n_iter,
                }
            )
    perturb = pd.DataFrame(perturb_rows)
    perturb.to_csv(TAB / "unknown_sensitivity.csv", index=False, encoding="utf-8-sig")
    at_20 = perturb[np.isclose(perturb["扰动幅度"], 0.20)]
    unknown_result = unknown_result.merge(
        at_20[["样本", "阈值翻转率", "随机森林翻转率"]], on="样本", how="left"
    )
    unknown_result.to_csv(TAB / "unknown_classification.csv", index=False, encoding="utf-8-sig")
    return metrics, unknown_result, perturb


def cluster_label_by_profile(labels: np.ndarray, centers_original: np.ndarray, glass_type: str) -> tuple[np.ndarray, dict[int, str]]:
    if glass_type == "高钾":
        # HK1: high silica / low potash; HK2: lower silica / high potash.
        si_idx, k_idx = 0, 1
        order = np.argsort(centers_original[:, si_idx] - centers_original[:, k_idx])
        mapping = {order[-1]: "HK1", order[0]: "HK2"}
    else:
        # LB3 high barium/copper; among remaining, LB2 is higher PbO/lower SiO2.
        si_idx, pb_idx, ba_idx, cu_idx = 0, 1, 2, 3
        lb3 = int(np.argmax(centers_original[:, ba_idx] + centers_original[:, cu_idx]))
        rest = [i for i in range(len(centers_original)) if i != lb3]
        lb2 = rest[int(np.argmax([centers_original[i, pb_idx] - centers_original[i, si_idx] for i in rest]))]
        lb1 = [i for i in rest if i != lb2][0]
        mapping = {lb1: "LB1", lb2: "LB2", lb3: "LB3"}
    named = np.array([mapping[int(x)] for x in labels], dtype=object)
    return named, mapping


def clustering_analysis(chem: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    selections = {
        "高钾": ["二氧化硅(SiO2)", "氧化钾(K2O)", "氧化钙(CaO)", "氧化铝(Al2O3)"],
        "铅钡": ["二氧化硅(SiO2)", "氧化铅(PbO)", "氧化钡(BaO)", "氧化铜(CuO)"],
    }
    target_k = {"高钾": 2, "铅钡": 3}
    silhouette_records = []
    assignment_records = []
    centroid_records = []
    stability = {}
    pca_data = {}
    summary = {}

    for glass_type in ["高钾", "铅钡"]:
        d = chem[chem["类型"] == glass_type].copy().reset_index(drop=True)
        comps = selections[glass_type]
        X = d[comps].to_numpy()
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
        for k in range(2, min(7, len(d) - 1)):
            km = KMeans(n_clusters=k, random_state=2022, n_init=100)
            labels = km.fit_predict(Z)
            silhouette_records.append(
                {"类型": glass_type, "k": k, "轮廓系数": silhouette_score(Z, labels)}
            )
        k = target_k[glass_type]
        km = KMeans(n_clusters=k, random_state=2022, n_init=200)
        raw_labels = km.fit_predict(Z)
        centers_original = scaler.inverse_transform(km.cluster_centers_)
        named, mapping = cluster_label_by_profile(raw_labels, centers_original, glass_type)
        for i, row in d.iterrows():
            assignment_records.append(
                {
                    "文物采样点": row["文物采样点"],
                    "artifact_id": row["artifact_id"],
                    "类型": glass_type,
                    "风化": row["采样点风化"],
                    "亚类": named[i],
                }
            )
        for raw, label_name in mapping.items():
            idx = raw_labels == raw
            record = {"类型": glass_type, "亚类": label_name, "样本数": int(idx.sum())}
            for j, comp in enumerate(comps):
                record[SHORT[comp]] = float(X[idx, j].mean())
            record["风化占比"] = float(np.mean(d.loc[idx, "采样点风化"] != "无风化"))
            centroid_records.append(record)

        pca = PCA(n_components=2, random_state=2022)
        pcs = pca.fit_transform(Z)
        pca_data[glass_type] = {
            "scores": pcs,
            "labels": named,
            "ids": d["文物采样点"].tolist(),
            "variance": pca.explained_variance_ratio_,
        }

        # Robustness: multiplicative Gaussian perturbation and alternative scaling.
        aris = []
        for _ in range(120):
            pert = np.maximum(X * RNG.normal(1.0, 0.05, size=X.shape), 0)
            zp = StandardScaler().fit_transform(pert)
            lp = KMeans(n_clusters=k, random_state=int(RNG.integers(0, 2**31 - 1)), n_init=10).fit_predict(zp)
            aris.append(adjusted_rand_score(raw_labels, lp))
        z_mm = MinMaxScaler().fit_transform(X)
        l_mm = KMeans(n_clusters=k, random_state=2022, n_init=200).fit_predict(z_mm)
        stability[glass_type] = {
            "perturb_ari_median": float(np.median(aris)),
            "perturb_ari_q1": float(np.quantile(aris, 0.25)),
            "perturb_ari_q3": float(np.quantile(aris, 0.75)),
            "perturb_ari_ge_08": float(np.mean(np.array(aris) >= 0.8)),
            "minmax_ari": float(adjusted_rand_score(raw_labels, l_mm)),
            "ari_values": aris,
        }
        sil_df = pd.DataFrame(silhouette_records)
        glass_sil = sil_df[sil_df["类型"] == glass_type].copy()
        best_row = glass_sil.sort_values("轮廓系数", ascending=False).iloc[0]
        selected_row = glass_sil[glass_sil["k"] == k].iloc[0]
        summary[glass_type] = {
            "selected_components": [SHORT[x] for x in comps],
            "selected_k": k,
            "best_k_by_silhouette": int(best_row["k"]),
            "best_silhouette": float(best_row["轮廓系数"]),
            "selected_silhouette": float(selected_row["轮廓系数"]),
        }

    silhouette_df = pd.DataFrame(silhouette_records)
    assignment_df = pd.DataFrame(assignment_records)
    centroid_df = pd.DataFrame(centroid_records)
    silhouette_df.to_csv(TAB / "silhouette_scores.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(TAB / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
    centroid_df.to_csv(TAB / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    pca_data["stability"] = stability
    return summary, assignment_df, centroid_df, pca_data


def correlation_analysis(chem: pd.DataFrame) -> tuple[dict, dict]:
    selected = [
        "二氧化硅(SiO2)",
        "氧化钾(K2O)",
        "氧化钙(CaO)",
        "氧化铝(Al2O3)",
        "氧化铁(Fe2O3)",
        "氧化铜(CuO)",
        "氧化铅(PbO)",
        "氧化钡(BaO)",
        "五氧化二磷(P2O5)",
    ]
    idx = [COMPONENTS_CN.index(c) for c in selected]
    corr = {}
    pvals = {}
    sample_sizes = {}
    for glass_type in ["高钾", "铅钡"]:
        X = chem.loc[chem["类型"] == glass_type, COMPONENTS_CN].to_numpy()
        Z = clr(X)[:, idx]
        rho, p = spearmanr(Z, axis=0)
        corr[glass_type] = rho
        pvals[glass_type] = p
        sample_sizes[glass_type] = len(Z)

    diff = corr["高钾"] - corr["铅钡"]
    observed = float(np.linalg.norm(diff[np.triu_indices(len(selected), 1)]))
    Xall = chem[COMPONENTS_CN].to_numpy()
    Zall = clr(Xall)[:, idx]
    labels = chem["类型"].to_numpy()
    exceed = 0
    n_perm = 1000
    for _ in range(n_perm):
        lp = RNG.permutation(labels)
        r1 = spearmanr(Zall[lp == "高钾"], axis=0).statistic
        r2 = spearmanr(Zall[lp == "铅钡"], axis=0).statistic
        val = np.linalg.norm((r1 - r2)[np.triu_indices(len(selected), 1)])
        exceed += val >= observed - 1e-12
    perm_p = (exceed + 1) / (n_perm + 1)

    pair_rows = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pair_rows.append(
                {
                    "成分对": f"{SHORT[selected[i]]}-{SHORT[selected[j]]}",
                    "高钾rho": corr["高钾"][i, j],
                    "铅钡rho": corr["铅钡"][i, j],
                    "差值": diff[i, j],
                    "高钾p": pvals["高钾"][i, j],
                    "铅钡p": pvals["铅钡"][i, j],
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs["绝对差"] = pairs["差值"].abs()
    pairs.sort_values("绝对差", ascending=False).to_csv(
        TAB / "correlation_pair_comparison.csv", index=False, encoding="utf-8-sig"
    )
    edge_counts = {
        g: int(np.sum(np.abs(corr[g][np.triu_indices(len(selected), 1)]) >= 0.6))
        for g in ["高钾", "铅钡"]
    }
    summary = {
        "selected_components": [SHORT[c] for c in selected],
        "sample_sizes": sample_sizes,
        "matrix_difference_norm": observed,
        "matrix_difference_permutation_p": float(perm_p),
        "strong_edge_counts_abs_rho_ge_06": edge_counts,
    }
    arrays = {"corr": corr, "diff": diff, "pairs": pairs, "selected": selected}
    return summary, arrays


def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.30, 0.16, 0.42, "数据质控", "85%–105%有效性\n缺失值置零并闭合"),
        (0.22, 0.30, 0.16, 0.42, "风化分析", "列联检验 + 效应量\nCLR位移复原"),
        (0.42, 0.30, 0.16, 0.42, "类型判别", "PbO阈值规则\n文物分组交叉验证"),
        (0.62, 0.30, 0.16, 0.42, "亚类划分", "特征筛选 + K-means\n轮廓系数与ARI"),
        (0.82, 0.30, 0.16, 0.42, "关联比较", "CLR-Spearman\n置换检验与差异网络"),
    ]
    for i, (x, y, w, h, title, body) in enumerate(boxes):
        color = COLORS["highk"] if i in (0, 2) else COLORS["lead"] if i == 4 else "#42949E"
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F7F9FC", edgecolor=color, linewidth=1.3
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.29, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=color)
        ax.text(x + w / 2, y + 0.13, body, ha="center", va="center", fontsize=6.7, color="#333333", linespacing=1.35)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.01, 0.51), (boxes[i + 1][0] - 0.01, 0.51), arrowstyle="-|>", mutation_scale=10, linewidth=1, color="#767676"))
    ax.text(0.5, 0.88, "古代玻璃成分分析的分层稳健建模框架", ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.12, "文物层面检验外部属性；采样点层面建模化学组成；所有验证按文物编号分组，避免重复采样泄漏", ha="center", va="center", fontsize=6.8, color="#4D4D4D")
    save_figure(fig, "fig1_workflow")


def figure_weathering(info: pd.DataFrame, component_stats: pd.DataFrame, paired: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 5.65))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a: weathering rates by glass type
    groups = ["高钾", "铅钡"]
    rates, lows, highs, counts = [], [], [], []
    for g in groups:
        s = info[info["类型"] == g]
        k = int((s["表面风化"] == "风化").sum())
        n = len(s)
        lo, hi = wilson_interval(k, n)
        rates.append(k / n)
        lows.append(k / n - lo)
        highs.append(hi - k / n)
        counts.append((k, n))
    ax_a.bar(groups, rates, color=[COLORS["highk"], COLORS["lead"]], width=0.58)
    ax_a.errorbar(np.arange(2), rates, yerr=[lows, highs], fmt="none", color="#272727", capsize=3, lw=0.9)
    for i, (r, (k, n)) in enumerate(zip(rates, counts)):
        ax_a.text(i, r + highs[i] + 0.045, f"{r:.0%}\n({k}/{n})", ha="center", va="bottom", fontsize=7)
    ax_a.set_ylim(0, 1.03)
    ax_a.set_ylabel("风化比例")
    ax_a.set_title("a  玻璃类型与风化比例", loc="left", fontweight="bold")
    ax_a.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])

    # b: pattern rates
    patterns = ["A", "B", "C"]
    prates = []
    for p in patterns:
        s = info[info["纹饰"] == p]
        prates.append((s["表面风化"] == "风化").mean())
    ax_b.bar(patterns, prates, color=["#B4C0E4", "#E9A6A1", "#8BCF8B"], width=0.58, edgecolor="#4D4D4D", linewidth=0.5)
    for i, r in enumerate(prates):
        n = int((info["纹饰"] == patterns[i]).sum())
        ax_b.text(i, r + 0.035, f"{r:.0%} (n={n})", ha="center", va="bottom", fontsize=6.7)
    ax_b.set_ylim(0, 1.08)
    ax_b.set_ylabel("风化比例")
    ax_b.set_title("b  纹饰类别与风化比例", loc="left", fontweight="bold")
    ax_b.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])

    def dumbbell(ax, glass_type, comps, panel):
        s = component_stats[(component_stats["类型"] == glass_type) & component_stats["成分"].isin(comps)].copy()
        s["order"] = s["成分"].map({c: i for i, c in enumerate(comps)})
        s = s.sort_values("order", ascending=False)
        y = np.arange(len(s))
        for yi, (_, row) in zip(y, s.iterrows()):
            ax.plot([row["无风化均值"], row["风化均值"]], [yi, yi], color="#B8B8B8", lw=1.5, zorder=1)
        ax.scatter(s["无风化均值"], y, color=COLORS["unweathered"], s=30, label="无风化", zorder=2)
        ax.scatter(s["风化均值"], y, color=COLORS["weathered"], s=30, label="风化", zorder=2)
        sig = s["BH_q"] < 0.05
        for yi, is_sig, (_, row) in zip(y, sig, s.iterrows()):
            if is_sig:
                ax.text(max(row["无风化均值"], row["风化均值"]) + 1.2, yi, "*", va="center", fontsize=8, color=COLORS["lead"])
        ax.set_yticks(y, s["成分"])
        ax.set_xlabel("闭合校正后的平均含量（%）")
        ax.set_title(f"{panel}  {glass_type}玻璃风化前后成分位移", loc="left", fontweight="bold")
        ax.legend(loc="lower right", ncol=2, fontsize=6.5)
        ax.grid(axis="x", color="#EEEEEE", lw=0.5)

    dumbbell(ax_c, "高钾", ["SiO2", "K2O", "CaO", "Al2O3", "Fe2O3"], "c")
    dumbbell(ax_d, "铅钡", ["SiO2", "PbO", "BaO", "P2O5", "CuO"], "d")
    fig.text(0.5, 0.008, "注：误差线为Wilson 95%置信区间；* 表示Mann–Whitney U检验经BH校正后 q<0.05。", ha="center", fontsize=6.4, color="#4D4D4D")
    save_figure(fig, "fig2_weathering")


def figure_classification(chem: pd.DataFrame, unknown: pd.DataFrame, metrics: dict) -> None:
    fig = plt.figure(figsize=(7.25, 4.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 0.82], wspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    pbo = "氧化铅(PbO)"
    k2o = "氧化钾(K2O)"
    marker_map = {"无风化": "o", "风化": "s", "严重风化": "D"}
    for glass_type, color in [("高钾", COLORS["highk"]), ("铅钡", COLORS["lead"])]:
        for status, marker in marker_map.items():
            s = chem[(chem["类型"] == glass_type) & (chem["采样点风化"] == status)]
            if len(s):
                ax_a.scatter(s[pbo], s[k2o], s=26, c=color, marker=marker, alpha=0.78, edgecolor="white", linewidth=0.45)
    for _, row in unknown.iterrows():
        ax_a.scatter(row[pbo], row[k2o], s=75, marker="*", c="#FFD700", edgecolor="#272727", linewidth=0.65, zorder=5)
        ax_a.annotate(row["文物编号"], (row[pbo], row[k2o]), xytext=(3, 3), textcoords="offset points", fontsize=6.2)
    ax_a.axvline(metrics["threshold"], color="#272727", lw=1.1, ls="--")
    ax_a.text(metrics["threshold"] + 0.7, ax_a.get_ylim()[1] * 0.92, f"PbO={metrics['threshold']:.2f}%", rotation=90, va="top", fontsize=6.3)
    ax_a.set_xlabel("PbO（%）")
    ax_a.set_ylabel("K2O（%）")
    ax_a.set_title("a  可解释判别边界与未知样品", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["highk"], markeredgecolor="none", label="高钾"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["lead"], markeredgecolor="none", label="铅钡"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFD700", markeredgecolor="#272727", label="未知样品", markersize=9),
    ]
    ax_a.legend(handles=handles, loc="upper right", fontsize=6.3)

    imp = pd.read_csv(TAB / "rf_feature_importance.csv").head(6).sort_values("重要性")
    ax_b.barh(imp["成分"], imp["重要性"], color=[COLORS["lead"] if x == "PbO" else "#B4C0E4" for x in imp["成分"]])
    ax_b.set_xlabel("随机森林重要性")
    ax_b.set_title("b  关键判别成分", loc="left", fontweight="bold")
    ax_b.grid(axis="x", color="#EEEEEE", lw=0.5)

    cm = np.array(metrics["logo_confusion"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(COLORS["highk"], as_cmap=True), cbar=False, square=True, ax=ax_c, annot_kws={"fontsize": 9})
    ax_c.set_xticklabels(["高钾", "铅钡"], rotation=0)
    ax_c.set_yticklabels(["高钾", "铅钡"], rotation=0)
    ax_c.set_xlabel("预测类型")
    ax_c.set_ylabel("真实类型")
    ax_c.set_title("c  留一文物验证", loc="left", fontweight="bold")
    fig.text(0.5, 0.01, "点形区分采样点风化状态；交叉验证以文物编号为分组单位，防止同件文物的多个采样点跨越训练集与验证集。", ha="center", fontsize=6.3, color="#4D4D4D")
    save_figure(fig, "fig3_classification")


def figure_clustering(summary: dict, assignments: pd.DataFrame, centroids: pd.DataFrame, pca_data: dict) -> None:
    sil = pd.read_csv(TAB / "silhouette_scores.csv")
    fig = plt.figure(figsize=(7.25, 6.15))
    gs = fig.add_gridspec(2, 2, hspace=0.43, wspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for g, color, marker in [("高钾", COLORS["highk"], "o"), ("铅钡", COLORS["lead"], "s")]:
        s = sil[sil["类型"] == g]
        ax_a.plot(s["k"], s["轮廓系数"], marker=marker, color=color, lw=1.5, ms=4.5, label=g)
        best = s.loc[s["轮廓系数"].idxmax()]
        ax_a.scatter([best["k"]], [best["轮廓系数"]], s=55, facecolor="none", edgecolor=color, lw=1.2)
    ax_a.set_xlabel("聚类数 k")
    ax_a.set_ylabel("轮廓系数")
    ax_a.set_xticks(sorted(sil["k"].unique()))
    ax_a.legend()
    ax_a.set_title("a  亚类数的轮廓系数选择", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color="#EEEEEE", lw=0.5)

    cluster_colors = {"HK1": "#0F4D92", "HK2": "#77D7D1", "LB1": "#E9A6A1", "LB2": "#B64342", "LB3": "#9A4D8E"}
    for ax, g, panel in [(ax_b, "高钾", "b"), (ax_c, "铅钡", "c")]:
        data = pca_data[g]
        scores = data["scores"]
        labels = data["labels"]
        for lab in sorted(set(labels)):
            idx = labels == lab
            ax.scatter(scores[idx, 0], scores[idx, 1], s=32, c=cluster_colors[lab], label=lab, alpha=0.82, edgecolor="white", linewidth=0.45)
            center = scores[idx].mean(axis=0)
            ax.scatter(center[0], center[1], marker="X", s=70, c=cluster_colors[lab], edgecolor="#272727", linewidth=0.55)
        ax.axhline(0, color="#D8D8D8", lw=0.6)
        ax.axvline(0, color="#D8D8D8", lw=0.6)
        ax.set_xlabel(f"PC1（{data['variance'][0]:.1%}）")
        ax.set_ylabel(f"PC2（{data['variance'][1]:.1%}）")
        ax.set_title(f"{panel}  {g}玻璃亚类的PCA投影", loc="left", fontweight="bold")
        ax.legend(fontsize=6.2)

    # centroid profile heatmap; normalize each type-specific selected component across its clusters.
    rows = []
    for _, row in centroids.iterrows():
        for col in [c for c in centroids.columns if c in ["SiO2", "K2O", "CaO", "Al2O3", "Fe2O3", "PbO", "BaO", "CuO", "P2O5"]]:
            if pd.notna(row.get(col)):
                rows.append({"亚类": row["亚类"], "成分": col, "均值": row[col], "类型": row["类型"]})
    long = pd.DataFrame(rows)
    mats = []
    for g in ["高钾", "铅钡"]:
        s = long[long["类型"] == g].pivot(index="亚类", columns="成分", values="均值")
        z = (s - s.mean(axis=0)) / s.std(axis=0, ddof=0).replace(0, 1)
        mats.append(z)
    mat = pd.concat(mats).reindex(["HK1", "HK2", "LB1", "LB2", "LB3"])
    desired_cols = ["SiO2", "K2O", "CaO", "Al2O3", "Fe2O3", "PbO", "BaO", "CuO", "P2O5"]
    mat = mat.reindex(columns=desired_cols)
    sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5, linewidths=0.4, linecolor="white", cbar_kws={"label": "类内标准化均值"}, ax=ax_d)
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.set_title("d  亚类中心的特征画像", loc="left", fontweight="bold")
    fig.text(0.5, 0.01, "PCA仅用于二维展示，K-means在筛选后的标准化原始成分空间中完成；叉号表示亚类中心。", ha="center", fontsize=6.3, color="#4D4D4D")
    save_figure(fig, "fig4_clustering")


def figure_unknown(unknown_result: pd.DataFrame, perturb: pd.DataFrame, threshold: float) -> None:
    fig = plt.figure(figsize=(7.25, 4.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    s = unknown_result.sort_values("PbO")
    colors = [COLORS["highk"] if x == "高钾" else COLORS["lead"] for x in s["阈值规则"]]
    y = np.arange(len(s))
    ax_a.hlines(y, threshold, s["PbO"], color=colors, lw=2.1, alpha=0.8)
    ax_a.scatter(s["PbO"], y, color=colors, s=38, zorder=3)
    ax_a.axvline(threshold, color="#272727", ls="--", lw=1)
    ax_a.set_yticks(y, s["样本"])
    ax_a.set_xlabel("PbO（%）")
    ax_a.set_title("a  未知样品与阈值的判别裕度", loc="left", fontweight="bold")
    ax_a.text(threshold + 0.5, len(s) - 0.45, f"阈值 {threshold:.2f}%", rotation=90, va="top", fontsize=6.3)

    focus = perturb.groupby(["扰动幅度", "样本"], as_index=False)["随机森林翻转率"].mean()
    a5 = focus[focus["样本"] == "A5"]
    rest = focus[focus["样本"] != "A5"].groupby("扰动幅度", as_index=False)["随机森林翻转率"].max()
    ax_b.plot(a5["扰动幅度"] * 100, a5["随机森林翻转率"] * 100, lw=1.7, marker="o", ms=3.2, color=COLORS["lead"], label="A5")
    ax_b.plot(rest["扰动幅度"] * 100, rest["随机森林翻转率"] * 100, lw=1.2, marker="o", ms=2.7, color=COLORS["highk"], label="其余7件的最大值")
    ax_b.axvline(20, color="#767676", ls="--", lw=0.8)
    ax_b.text(20.8, 96, "报告设定 ±20%", rotation=90, va="top", fontsize=6.2, color="#4D4D4D")
    ax_b.set_xlim(0, 66)
    ax_b.set_ylim(-2, 102)
    ax_b.set_xlabel("成分乘性扰动幅度（%）")
    ax_b.set_ylabel("随机森林标签翻转率（%）")
    ax_b.set_title("b  逐级加压的蒙特卡洛稳健性", loc="left", fontweight="bold")
    ax_b.legend(loc="upper left", fontsize=6.4)
    ax_b.grid(axis="y", color="#EEEEEE", lw=0.5)
    fig.text(0.5, 0.01, "每个非零扰动幅度进行500次模拟并重新闭合至100%；蓝线为高钾样品，红线为铅钡样品。", ha="center", fontsize=6.3, color="#4D4D4D")
    save_figure(fig, "fig5_unknown_sensitivity")


def figure_correlation(arrays: dict) -> None:
    corr = arrays["corr"]
    selected = arrays["selected"]
    labels = [SHORT[c] for c in selected]
    fig = plt.figure(figsize=(7.25, 3.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    for ax, g, panel in [(ax_a, "高钾", "a"), (ax_b, "铅钡", "b")]:
        mask = np.triu(np.ones_like(corr[g], dtype=bool), 1)
        sns.heatmap(corr[g], mask=mask, vmin=-1, vmax=1, center=0, cmap="RdBu_r", square=True, cbar=False, xticklabels=labels, yticklabels=labels, linewidths=0.3, linecolor="white", ax=ax)
        ax.tick_params(axis="x", rotation=55, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)
        ax.set_title(f"{panel}  {g}玻璃 CLR-Spearman矩阵", loc="left", fontweight="bold")

    pairs = arrays["pairs"].copy()
    ax_c.scatter(pairs["高钾rho"], pairs["铅钡rho"], s=24, color="#7884B4", alpha=0.75, edgecolor="white", linewidth=0.35)
    ax_c.plot([-1, 1], [-1, 1], color="#767676", ls="--", lw=0.8)
    top = pairs.nlargest(7, "绝对差")
    for _, row in top.iterrows():
        ax_c.annotate(row["成分对"], (row["高钾rho"], row["铅钡rho"]), xytext=(3, 3), textcoords="offset points", fontsize=5.6)
    ax_c.axhline(0, color="#D8D8D8", lw=0.6)
    ax_c.axvline(0, color="#D8D8D8", lw=0.6)
    ax_c.set_xlim(-1.03, 1.03)
    ax_c.set_ylim(-1.03, 1.03)
    ax_c.set_xlabel("高钾玻璃相关系数")
    ax_c.set_ylabel("铅钡玻璃相关系数")
    ax_c.set_title("c  两类关联强度的成分对比较", loc="left", fontweight="bold")
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.24, wspace=0.42)
    cax = fig.add_axes([0.075, 0.145, 0.555, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Spearman ρ")
    fig.text(0.5, 0.025, "相关系数在零值替代与中心化对数比（CLR）变换后计算；虚线表示两类相关强度相同。", ha="center", fontsize=6.3, color="#4D4D4D")
    save_figure(fig, "fig6_correlation")


def main() -> None:
    info, chem, unknown = load_data()
    balanced = artifact_balanced_weathering(chem)
    assoc = association_analysis(info)
    component_stats = weathering_component_analysis(balanced)
    pred_df, paired, pred_summary = weathering_prediction(chem, balanced)
    class_metrics, unknown_result, perturb = classification_analysis(chem, unknown)
    cluster_summary, assignments, centroids, pca_data = clustering_analysis(chem)
    corr_summary, corr_arrays = correlation_analysis(chem)

    figure_workflow()
    figure_weathering(info, component_stats, paired)
    figure_classification(chem, unknown, class_metrics)
    figure_clustering(cluster_summary, assignments, centroids, pca_data)
    figure_unknown(unknown_result, perturb, class_metrics["threshold"])
    figure_correlation(corr_arrays)

    metrics = {
        "data": {
            "artifact_n": int(len(info)),
            "highk_artifact_n": int((info["类型"] == "高钾").sum()),
            "lead_artifact_n": int((info["类型"] == "铅钡").sum()),
            "weathered_artifact_n": int((info["表面风化"] == "风化").sum()),
            "unweathered_artifact_n": int((info["表面风化"] == "无风化").sum()),
            "valid_point_n": int(len(chem)),
            "invalid_points": ["15", "17"],
            "unknown_n": int(len(unknown)),
            "artifact_balanced_rows": int(len(balanced)),
        },
        "association": {
            "rows": assoc.to_dict(orient="records"),
            "type_fisher_p": assoc.attrs["type_fisher_p"],
            "odds_ratio_pb_vs_hk": assoc.attrs["odds_ratio_pb_vs_hk"],
            "odds_ratio_ci": assoc.attrs["odds_ratio_ci"],
        },
        "prediction": pred_summary,
        "classification": class_metrics,
        "clustering": cluster_summary,
        "cluster_stability": {
            g: {k: v for k, v in pca_data["stability"][g].items() if k != "ari_values"}
            for g in ["高钾", "铅钡"]
        },
        "correlation": corr_summary,
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Analysis complete. Metrics saved to", OUT / "metrics.json")


if __name__ == "__main__":
    main()
