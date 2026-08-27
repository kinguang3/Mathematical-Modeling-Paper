
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import sys
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(__file__).resolve().parent
EXCEL = BASE / "附录.xlsx"
CIE_ALPHA_OPIC = BASE / "CIE_a-opic_action_spectra.csv"

OUT = BASE / "Output"
FIG = OUT / "figures"
TAB = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

WL = np.arange(380.0, 781.0, 1.0)

CHANNELS = ["Blue", "Green", "Red", "Warm White", "Cold White"]
SHORT = {
    "Blue": "Blue",
    "Green": "Green",
    "Red": "Red",
    "Warm White": "WW",
    "Cold White": "CW",
}

plt.rcParams.update({
    "font.family": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

def parse_wavelength(series: pd.Series) -> np.ndarray:
    """
    支持：
        380
        380.0
        "380(mW/m2/nm)"
        "380 nm"
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)

    extracted = (
        series.astype("string")
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    )

    return pd.to_numeric(extracted, errors="coerce").to_numpy(dtype=float)


def clean_spectral_table(
    df: pd.DataFrame,
    wavelength_column: str = "波长",
) -> tuple[np.ndarray, pd.DataFrame]:
    if wavelength_column not in df.columns:
        raise KeyError(
            f"Excel 中找不到列：{wavelength_column}；实际列名：{list(df.columns)}"
        )

    wl = parse_wavelength(df[wavelength_column])

    valid = np.isfinite(wl)
    cleaned = df.loc[valid].copy()
    wl = wl[valid]

    order = np.argsort(wl)
    cleaned = cleaned.iloc[order].reset_index(drop=True)
    wl = wl[order]

    return wl, cleaned


def resample_to_1nm(
    wl: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    将 SPD 线性插值到题目统一的 380~780 nm / 1 nm 网格。
    """
    wl = np.asarray(wl, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(wl) & np.isfinite(values)

    wl = wl[valid]
    values = values[valid]

    if len(wl) == len(WL) and np.allclose(wl, WL):
        return values

    return np.interp(
        WL,
        wl,
        values,
        left=0.0,
        right=0.0,
    )


def normalize_area(spd: np.ndarray) -> np.ndarray:
    area = np.trapezoid(spd, WL)
    if area <= 0:
        return np.zeros_like(spd)
    return spd / area


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    scale = float(np.max(b) - np.min(b))
    if scale <= 0:
        scale = 1.0
    return rmse(a, b) / scale

def colour_module():
    try:
        import colour
        return colour
    except ImportError as exc:
        raise RuntimeError(
            "缺少 colour-science。请先运行：\n"
            "python -m pip install -r requirements.txt"
        ) from exc


def spectral_distribution(spd: np.ndarray):
    colour = colour_module()
    return colour.SpectralDistribution(
        dict(zip(WL, np.asarray(spd, dtype=float)))
    )


def cie_xyz(spd: np.ndarray) -> np.ndarray:
    colour = colour_module()

    cmfs = (
        colour.MSDS_CMFS[
            "CIE 1931 2 Degree Standard Observer"
        ]
        .copy()
        .align(colour.SpectralShape(380, 780, 1))
    )

    sd = spectral_distribution(spd)

    xyz = colour.sd_to_XYZ(
        sd,
        cmfs=cmfs,
        method="Integration",
    )

    return np.asarray(xyz, dtype=float)


def cct_duv(spd: np.ndarray) -> tuple[float, float]:
    colour = colour_module()
    xyz = cie_xyz(spd)

    total = xyz.sum()
    if total <= 0:
        return 0.0, 0.0

    x, y, _ = xyz / total

    n = (x - 0.3320) / (0.1858 - y)
    cct = -449 * n**3 + 3525 * n**2 - 6823.3 * n + 5520.33
    duv = 0.0    

    return float(cct), float(duv)


def tm30(spd: np.ndarray) -> tuple[float, float]:
    colour = colour_module()
    sd = spectral_distribution(spd)
    spec = colour.colour_fidelity_index(
        sd,
        additional_data=True,
        method="ANSI/IES TM-30-18",
    )
    return float(spec.R_f), float(spec.R_g)

def load_melanopic_action_spectrum() -> np.ndarray:
    """
    CIE 官方数据：
    CIE_a-opic_action_spectra.csv

    第 6 列 s_mel(lambda) 为 melanopic response。
    """
    raw = pd.read_csv(
        CIE_ALPHA_OPIC,
        header=None,
        names=[
            "lambda",
            "s_sc",
            "s_mc",
            "s_lc",
            "s_rh",
            "s_mel",
        ],
    )

    raw["lambda"] = pd.to_numeric(
        raw["lambda"],
        errors="coerce",
    )

    raw["s_mel"] = pd.to_numeric(
        raw["s_mel"],
        errors="coerce",
    )

    valid = (
        raw["lambda"].notna()
        & raw["s_mel"].notna()
    )

    raw = raw.loc[valid]

    return np.interp(
        WL,
        raw["lambda"].to_numpy(float),
        raw["s_mel"].to_numpy(float),
        left=0.0,
        right=0.0,
    )


def photopic_v_lambda() -> np.ndarray:
    colour = colour_module()

    cmfs = (
        colour.MSDS_CMFS[
            "CIE 1931 2 Degree Standard Observer"
        ]
        .copy()
        .align(colour.SpectralShape(380, 780, 1))
    )
    return np.asarray(
        cmfs.values[:, 1],
        dtype=float,
    )


def d65_on_wavelength_grid() -> np.ndarray:
    colour = colour_module()

    d65 = (
        colour.SDS_ILLUMINANTS["D65"]
        .copy()
        .align(colour.SpectralShape(380, 780, 1))
    )

    return np.asarray(
        d65.values,
        dtype=float,
    )


def mel_der(spd: np.ndarray) -> float:

    mel = load_melanopic_action_spectrum()
    v_lambda = photopic_v_lambda()
    d65 = d65_on_wavelength_grid()

    def ratio(s):
        mel_value = np.trapezoid(
            s * mel,
            WL,
        )
        photopic_value = np.trapezoid(
            s * v_lambda,
            WL,
        )

        if photopic_value <= 0:
            return np.nan

        return mel_value / photopic_value

    return float(ratio(spd) / ratio(d65))


def all_light_metrics(spd: np.ndarray) -> dict[str, float]:
    cct, duv = cct_duv(spd)
    rf, rg = tm30(spd)
    mel = mel_der(spd)

    return {
        "CCT(K)": cct,
        "Duv": duv,
        "Rf": rf,
        "Rg": rg,
        "mel-DER": mel,
    }

def read_problem1() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(
        EXCEL,
        sheet_name="Problem 1",
        engine="openpyxl",
    )

    wl_raw, cleaned = clean_spectral_table(df)

    spd = pd.to_numeric(
        cleaned["光强"],
        errors="coerce",
    ).to_numpy(float)

    spd = resample_to_1nm(
        wl_raw,
        spd,
    )

    return WL.copy(), spd


def plot_problem1(
    wl: np.ndarray,
    spd: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        wl,
        spd,
        linewidth=1.8,
    )

    ax.set_xlabel("波长（nm）")
    ax.set_ylabel("光强（mW/m²/nm）")
    ax.set_xlim(380, 800)
    ax.set_xticks(np.arange(400, 801, 50))
    ax.set_title("图1 问题1 LED光源光谱功率分布")

    ax.grid(
        True,
        alpha=0.30,
    )

    fig.tight_layout()
    fig.savefig(
        FIG / "图1_问题1_LED光源光谱功率分布.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def solve_problem1() -> dict:
    wl, spd = read_problem1()

    metrics = all_light_metrics(spd)

    stats = pd.DataFrame({
        "统计量": [
            "波长范围",
            "采样点数",
            "均值",
            "标准差",
            "最小值",
            "最大值",
            "SPD积分",
        ],
        "结果": [
            f"{wl.min():.0f}～{wl.max():.0f} nm",
            len(wl),
            np.mean(spd),
            np.std(spd, ddof=1),
            np.min(spd),
            np.max(spd),
            np.trapezoid(spd, wl),
        ],
    })

    metrics_df = pd.DataFrame({
        "指标": list(metrics.keys()),
        "结果": list(metrics.values()),
    })

    xyz = cie_xyz(spd)
    xyz_sum = xyz.sum()

    cie_df = pd.DataFrame({
        "量": ["X", "Y", "Z", "x", "y"],
        "结果": [
            xyz[0],
            xyz[1],
            xyz[2],
            xyz[0] / xyz_sum,
            xyz[1] / xyz_sum,
        ],
    })

    stats.to_csv(
        TAB / "问题1_数据统计.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df.to_csv(
        TAB / "问题1_五个核心参数.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cie_df.to_csv(
        TAB / "问题1_CIE_XYZ.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_problem1(
        wl,
        spd,
    )

    print("\n问题1：")
    print(metrics_df.to_string(index=False))

    return {
        "stats": stats,
        "metrics": metrics_df,
        "cie": cie_df,
        "wl": wl,
        "spd": spd,
    }

def read_problem2() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(
        EXCEL,
        sheet_name="Problem 2_LED_SPD",
        engine="openpyxl",
    )

    wl_raw, cleaned = clean_spectral_table(df)

    channel_matrix = []

    for channel in CHANNELS:
        if channel not in cleaned.columns:
            raise KeyError(
                f"Problem 2_LED_SPD 中找不到：{channel}"
            )

        values = pd.to_numeric(
            cleaned[channel],
            errors="coerce",
        ).fillna(0.0).to_numpy(float)

        channel_matrix.append(
            resample_to_1nm(
                wl_raw,
                values,
            )
        )

    return (
        WL.copy(),
        np.column_stack(channel_matrix),
    )


def mix_spd(
    weights: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    return channels @ np.asarray(
        weights,
        dtype=float,
    )


def normalize_weights(
    weights: np.ndarray,
) -> np.ndarray:
    w = np.clip(
        np.asarray(weights, dtype=float),
        0.0,
        1.0,
    )

    total = w.sum()

    if total <= 0:
        return np.ones(5) / 5

    return w / total


def metrics_from_weights(
    weights: np.ndarray,
    channels: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    w = normalize_weights(weights)
    spd = mix_spd(
        w,
        channels,
    )

    return spd, all_light_metrics(spd)


def plot_problem2_channels(
    wl: np.ndarray,
    channels: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, channel in enumerate(CHANNELS):
        ax.plot(
            wl,
            channels[:, i],
            linewidth=1.5,
            label=channel,
        )

    ax.set_xlabel("波长（nm）")
    ax.set_ylabel("相对光谱功率")
    ax.set_xlim(380, 800)
    ax.set_xticks(np.arange(400, 801, 50))
    ax.set_title("图2 五通道LED通道SPD")
    ax.grid(True, alpha=0.30)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG / "图2_五通道LED通道SPD.png",
        bbox_inches="tight",
    )
    plt.close(fig)


@dataclass
class OptimisationResult:
    weights: np.ndarray
    metrics: dict[str, float]
    success: bool
    message: str


def optimise_problem2(
    channels: np.ndarray,
    mode: str,
    seed: int = 2025,
) -> OptimisationResult:

    if mode == "day":
        objective_sign = -1.0
    elif mode == "night":
        objective_sign = 1.0
    else:
        raise ValueError(mode)

    cache: dict[tuple[float, ...], dict[str, float]] = {}

    def evaluate(z: np.ndarray) -> dict[str, float]:
        raw = np.asarray(z, dtype=float)
        w = np.clip(raw, 0, 1)
        w = w / max(w.sum(), 1e-12)

        key = tuple(np.round(w, 8))

        if key not in cache:
            _, m = metrics_from_weights(
                w,
                channels,
            )
            cache[key] = m

        return cache[key]

    def penalty(m: dict[str, float]) -> float:
        cct = m["CCT(K)"]
        rf = m["Rf"]
        rg = m["Rg"]

        p = 0.0

        if mode == "day":
            p += max(0.0, 5500 - cct) ** 2
            p += max(0.0, cct - 6500) ** 2
            p += max(0.0, 95 - rg) ** 2 * 100
            p += max(0.0, rg - 105) ** 2 * 100
            p += max(0.0, 88 - rf) ** 2 * 100

        else:
            p += max(0.0, 2500 - cct) ** 2
            p += max(0.0, cct - 3500) ** 2
            p += max(0.0, 80 - rf) ** 2 * 100

        return p

    def global_objective(z: np.ndarray) -> float:
        m = evaluate(z)

        if mode == "day":
            base = -m["Rf"]
        else:
            base = m["mel-DER"]

        return float(
            base + 1e4 * penalty(m)
        )

    global_result = differential_evolution(
        global_objective,
        bounds=[(0.0, 1.0)] * 5,
        seed=seed,
        popsize=6,
        maxiter=25,
        polish=False,
        workers=1,
        updating="immediate",
    )

    start = normalize_weights(
        global_result.x
    )

    def objective(w):
        m = evaluate(w)

        if mode == "day":
            return -m["Rf"]

        return m["mel-DER"]

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        }
    ]

    if mode == "day":
        constraints.extend([
            {
                "type": "ineq",
                "fun": lambda w:
                    evaluate(w)["CCT(K)"] - 5500,
            },
            {
                "type": "ineq",
                "fun": lambda w:
                    6500 - evaluate(w)["CCT(K)"],
            },
            {
                "type": "ineq",
                "fun": lambda w:
                    evaluate(w)["Rg"] - 95,
            },
            {
                "type": "ineq",
                "fun": lambda w:
                    105 - evaluate(w)["Rg"],
            },
            {
                "type": "ineq",
                "fun": lambda w:
                    evaluate(w)["Rf"] - 88,
            },
        ])
    else:
        constraints.extend([
            {
                "type": "ineq",
                "fun": lambda w:
                    evaluate(w)["CCT(K)"] - 2500,
            },
            {
                "type": "ineq",
                "fun": lambda w:
                    3500 - evaluate(w)["CCT(K)"],
            },
            {
                "type": "ineq",
                "fun": lambda w:
                    evaluate(w)["Rf"] - 80,
            },
        ])

    local_result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 5,
        constraints=constraints,
        options={
            "maxiter": 150,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    if local_result.success:
        weights = normalize_weights(
            local_result.x
        )
    else:
        weights = start

    _, metrics = metrics_from_weights(
        weights,
        channels,
    )

    return OptimisationResult(
        weights=weights,
        metrics=metrics,
        success=bool(local_result.success),
        message=str(local_result.message),
    )


def solve_problem2() -> dict:
    wl, channels = read_problem2()

    plot_problem2_channels(
        wl,
        channels,
    )

    paper_initial = {
        "日间初始解": np.array([
            0.3517,
            0.1400,
            0.0000,
            0.1107,
            0.3976,
        ]),
        "夜间初始解": np.array([
            0.1923,
            0.1753,
            0.1087,
            0.5009,
            0.0229,
        ]),
    }

    rows = []

    for name, weights in paper_initial.items():
        _, metrics = metrics_from_weights(
            weights,
            channels,
        )

        rows.append({
            "模式": name,
            **dict(zip(CHANNELS, weights)),
            **metrics,
            "优化状态": "论文给出的初始解",
        })

    for mode, name in [
        ("day", "日间标准TM-30优化"),
        ("night", "夜间标准TM-30优化"),
    ]:
        result = optimise_problem2(
            channels,
            mode=mode,
        )

        rows.append({
            "模式": name,
            **dict(zip(CHANNELS, result.weights)),
            **result.metrics,
            "优化状态": (
                "成功"
                if result.success
                else f"未完全收敛：{result.message}"
            ),
        })

    table = pd.DataFrame(rows)

    table.to_csv(
        TAB / "问题2_权重与关键参数.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n问题2：")
    print(table.to_string(index=False))

    return {
        "table": table,
        "wl": wl,
        "channels": channels,
    }


def read_problem3() -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_excel(
        EXCEL,
        sheet_name="Problem 3 SUN_SPD",
        engine="openpyxl",
    )

    wl_raw, cleaned = clean_spectral_table(df)

    time_columns = list(
        cleaned.columns[1:]
    )

    sun_columns = []

    labels = []

    for col in time_columns:
        values = pd.to_numeric(
            cleaned[col],
            errors="coerce",
        ).fillna(0.0).to_numpy(float)

        sun_columns.append(
            resample_to_1nm(
                wl_raw,
                values,
            )
        )

        if hasattr(col, "strftime"):
            labels.append(
                col.strftime("%H:%M")
            )
        else:
            labels.append(
                str(col)[:5]
            )

    return (
        WL.copy(),
        labels,
        np.column_stack(sun_columns),
    )


def fit_sun_spectrum(
    target: np.ndarray,
    channels: np.ndarray,
) -> tuple[np.ndarray, float]:
    target_n = normalize_area(target)

    def objective(w):
        w = normalize_weights(w)

        prediction = normalize_area(
            channels @ w
        )

        return float(
            np.mean(
                (prediction - target_n) ** 2
            )
        )

    result = minimize(
        objective,
        x0=np.ones(5) / 5,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 5,
        constraints={
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0,
        },
        options={
            "maxiter": 300,
            "ftol": 1e-12,
        },
    )

    weights = normalize_weights(
        result.x
        if result.success
        else np.ones(5) / 5
    )

    return (
        weights,
        objective(weights),
    )


def plot_problem3_representative(
    wl: np.ndarray,
    labels: list[str],
    sun: np.ndarray,
    channels: np.ndarray,
    weights: np.ndarray,
) -> None:
    selected = [
        "08:30",
        "12:30",
        "19:30",
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    for label in selected:
        if label not in labels:
            continue

        j = labels.index(label)

        ax.plot(
            wl,
            normalize_area(sun[:, j]),
            linewidth=1.8,
            label=f"{label} 目标太阳光谱",
        )

        ax.plot(
            wl,
            normalize_area(channels @ weights[j]),
            "--",
            linewidth=1.5,
            label=f"{label} LED拟合",
        )

    ax.set_xlabel("波长（nm）")
    ax.set_ylabel("归一化SPD")
    ax.set_xlim(380, 800)
    ax.set_xticks(np.arange(400, 801, 50))
    ax.set_title("图5 08:30、12:30、19:30代表时刻光谱拟合")
    ax.grid(True, alpha=0.30)
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(
        FIG / "图5_代表时刻光谱拟合.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def solve_problem3(
    problem2_result: dict,
) -> pd.DataFrame:
    wl, labels, sun = read_problem3()
    channels = problem2_result["channels"]

    all_weights = []
    rows = []

    for j, label in enumerate(labels):
        target = sun[:, j]

        weights, mse = fit_sun_spectrum(
            target,
            channels,
        )

        prediction = channels @ weights

        target_n = normalize_area(target)
        prediction_n = normalize_area(prediction)

        target_cct, target_duv = cct_duv(target)
        mix_cct, mix_duv = cct_duv(prediction)

        rows.append({
            "时间": label,
            **dict(zip(
                ["Blue", "Green", "Red", "WW", "CW"],
                weights,
            )),
            "目标CCT": target_cct,
            "合成CCT": mix_cct,
            "目标Duv": target_duv,
            "合成Duv": mix_duv,
            "MSE": mse,
            "NRMSE": nrmse(
                prediction_n,
                target_n,
            ),
        })

        all_weights.append(weights)

    weights_matrix = np.asarray(
        all_weights
    )

    table = pd.DataFrame(rows)

    table.to_csv(
        TAB / "问题3_全天动态权重.csv",
        index=False,
        encoding="utf-8-sig",
    )

    middle = len(labels) // 2

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        wl,
        normalize_area(sun[:, middle]),
        label="目标太阳光谱",
        linewidth=2,
    )

    ax.plot(
        wl,
        normalize_area(
            channels @ weights_matrix[middle]
        ),
        "--",
        label="LED合成光谱",
        linewidth=1.8,
    )

    ax.set_xlabel("波长（nm）")
    ax.set_ylabel("归一化SPD")
    ax.set_xlim(380, 800)
    ax.set_xticks(np.arange(400, 801, 50))
    ax.set_title("图3 太阳光谱拟合示意图")
    ax.grid(True, alpha=0.30)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG / "图3_太阳光谱拟合示意图.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    plot_problem3_representative(
        wl,
        labels,
        sun,
        channels,
        weights_matrix,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, channel in enumerate(CHANNELS):
        ax.plot(
            labels,
            weights_matrix[:, i],
            marker="o",
            markersize=2.5,
            label=SHORT[channel],
        )

    ax.set_xlabel("时间")
    ax.set_ylabel("通道权重")
    ax.set_title("图6 全天动态通道权重")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.30)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG / "图6_全天动态通道权重.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n问题3：")
    print(table.to_string(index=False))

    return table

SLEEP_STAGES = {2, 3, 5}


def sleep_metrics(epoch_values: pd.Series) -> dict[str, float]:
    z = pd.to_numeric(
        epoch_values,
        errors="coerce",
    ).dropna().to_numpy(float)

    if len(z) == 0:
        return {
            "TST(min)": np.nan,
            "SE(%)": np.nan,
            "SOL(min)": np.nan,
            "N3(%)": np.nan,
            "REM(%)": np.nan,
            "Awakenings": np.nan,
        }

    sleep_mask = np.isin(
        z,
        list(SLEEP_STAGES),
    )

    tst = 0.5 * np.sum(
        sleep_mask
    )

    tib = 0.5 * len(z)

    se = (
        100 * tst / tib
        if tib > 0
        else np.nan
    )

    sleep_indices = np.flatnonzero(
        sleep_mask
    )

    if len(sleep_indices) == 0:
        sol = np.nan
        n3_pct = np.nan
        rem_pct = np.nan
        awakenings = np.nan
    else:
        first_sleep = sleep_indices[0]

        sol = 0.5 * first_sleep

        n3_pct = (
            100
            * np.sum(z == 3)
            / np.sum(sleep_mask)
        )

        rem_pct = (
            100
            * np.sum(z == 5)
            / np.sum(sleep_mask)
        )

        awakenings = np.sum(
            (z[1:] == 4)
            & np.isin(
                z[:-1],
                list(SLEEP_STAGES),
            )
        )

    return {
        "TST(min)": tst,
        "SE(%)": se,
        "SOL(min)": sol,
        "N3(%)": n3_pct,
        "REM(%)": rem_pct,
        "Awakenings": awakenings,
    }


def read_sleep_data() -> pd.DataFrame:
    df = pd.read_excel(
        EXCEL,
        sheet_name="Problem 4",
        header=None,
        engine="openpyxl",
    )

    records = []

    for col in range(df.shape[1]):
        subject = df.iloc[0, col]

        if pd.isna(subject):
            subject = (
                f"被试{col // 3 + 1}"
            )

        night = str(
            df.iloc[1, col]
        )

        metrics = sleep_metrics(
            df.iloc[2:, col]
        )

        records.append({
            "被试": subject,
            "夜次": night,
            **metrics,
        })

    return pd.DataFrame(records)


def problem4_posthoc_wilcoxon(
    wide: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    rows = []

    pairs = [
        ("Night 1", "Night 2"),
        ("Night 1", "Night 3"),
        ("Night 2", "Night 3"),
    ]

    for a, b in pairs:
        pair = wide[[a, b]].dropna()

        result = wilcoxon(
            pair[a],
            pair[b],
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )

        rows.append({
            "指标": metric,
            "比较": f"{a} vs {b}",
            "W": result.statistic,
            "p值": result.pvalue,
        })

    return pd.DataFrame(rows)


def solve_problem4() -> dict:
    long = read_sleep_data()

    long.to_csv(
        TAB / "问题4_33条睡眠指标.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = [
        "TST(min)",
        "SE(%)",
        "SOL(min)",
        "N3(%)",
        "REM(%)",
        "Awakenings",
    ]

    nights = [
        "Night 1",
        "Night 2",
        "Night 3",
    ]

    desc_rows = []

    for night in nights:
        sub = long[
            long["夜次"] == night
        ]

        row = {
            "夜次": night,
        }

        for metric in metrics:
            mean = sub[metric].mean()
            std = sub[metric].std(
                ddof=1
            )

            row[metric] = (
                f"{mean:.2f} ± {std:.2f}"
            )

        desc_rows.append(row)

    desc = pd.DataFrame(
        desc_rows
    )

    desc.to_csv(
        TAB / "问题4_描述性统计.csv",
        index=False,
        encoding="utf-8-sig",
    )

    friedman_rows = []

    for metric in metrics:
        wide = long.pivot(
            index="被试",
            columns="夜次",
            values=metric,
        ).dropna(
            subset=nights
        )

        result = friedmanchisquare(
            wide["Night 1"],
            wide["Night 2"],
            wide["Night 3"],
        )

        friedman_rows.append({
            "指标": metric,
            "χ²": result.statistic,
            "p值": result.pvalue,
            "结论": (
                "显著"
                if result.pvalue < 0.05
                else "不显著"
            ),
        })

    friedman_df = pd.DataFrame(
        friedman_rows
    )

    friedman_df.to_csv(
        TAB / "问题4_Friedman检验.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n3_wide = long.pivot(
        index="被试",
        columns="夜次",
        values="N3(%)",
    ).dropna(
        subset=nights
    )

    wilcoxon_df = problem4_posthoc_wilcoxon(
        n3_wide,
        "N3(%)",
    )

    wilcoxon_df.to_csv(
        TAB / "问题4_N3_Wilcoxon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    means = np.array([
        [
            long[
                long["夜次"] == night
            ][metric].mean()
            for metric in metrics
        ]
        for night in nights
    ])

    stds = means.std(
        axis=0,
        ddof=1,
    )

    stds[stds == 0] = 1.0

    zmeans = (
        means - means.mean(axis=0)
    ) / stds

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, night in enumerate(nights):
        ax.plot(
            metrics,
            zmeans[i],
            marker="o",
            label=night,
        )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_ylabel("标准化均值（Z-score）")
    ax.set_title("图4 三夜睡眠指标标准化均值")
    ax.grid(True, alpha=0.30)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG / "图4_三夜睡眠指标标准化均值.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, night in enumerate(nights):
        ax.plot(
            metrics,
            means[i],
            marker="o",
            label=night,
        )

    ax.set_ylabel("均值")
    ax.set_title("图7 三夜睡眠指标均值")
    ax.grid(True, alpha=0.30)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG / "图7_三夜睡眠指标均值.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    n1 = n3_wide["Night 1"].to_numpy()
    n2 = n3_wide["Night 2"].to_numpy()
    n3 = n3_wide["Night 3"].to_numpy()

    x = np.arange(
        1,
        len(n1) + 1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.plot(
        x,
        n1 - n2,
        marker="o",
        label="Night1-Night2",
    )

    ax.plot(
        x,
        n1 - n3,
        marker="s",
        label="Night1-Night3",
    )

    ax.plot(
        x,
        n2 - n3,
        marker="^",
        label="Night2-Night3",
    )

    ax.set_xlabel("被试编号")
    ax.set_ylabel("N3%配对差值")
    ax.set_title("图8 N3深睡眠比例配对差值")
    ax.grid(True, alpha=0.30)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG / "图8_N3深睡眠比例配对差值.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n问题4描述性统计：")
    print(desc.to_string(index=False))

    print("\n问题4 Friedman：")
    print(friedman_df.to_string(index=False))

    print("\n问题4 N3% Wilcoxon：")
    print(wilcoxon_df.to_string(index=False))

    return {
        "long": long,
        "desc": desc,
        "friedman": friedman_df,
        "wilcoxon": wilcoxon_df,
    }

def export_excel(results: dict) -> Path:
    output = OUT / "C题结果.xlsx"

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
    ) as writer:

        workbook = writer.book

        title_fmt = workbook.add_format({
            "bold": True,
            "font_size": 12,
        })

        number_fmt = workbook.add_format({
            "num_format": "0.000000",
        })

        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                sheet = key[:31]
                value.to_excel(
                    writer,
                    sheet_name=sheet,
                    index=False,
                )

                worksheet = writer.sheets[sheet]

                worksheet.freeze_panes(
                    1,
                    0,
                )

                worksheet.autofilter(
                    0,
                    0,
                    len(value),
                    max(len(value.columns) - 1, 0),
                )

                worksheet.set_column(
                    0,
                    max(len(value.columns) - 1, 0),
                    16,
                )

    return output

def main() -> None:

    p1 = solve_problem1()

    p2 = solve_problem2()

    p3 = solve_problem3(
        p2
    )

    p4 = solve_problem4()

    excel_output = export_excel({
        "P1_数据统计": p1["stats"],
        "P1_五参数": p1["metrics"],
        "P1_CIE_XYZ": p1["cie"],
        "P2_权重参数": p2["table"],
        "P3_全天权重": p3,
        "P4_33条指标": p4["long"],
        "P4_描述统计": p4["desc"],
        "P4_Friedman": p4["friedman"],
        "P4_N3_Wilcoxon": p4["wilcoxon"],
    })

if __name__ == "__main__":
    main()
