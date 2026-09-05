from pathlib import Path
import hashlib
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import csv
import pandas as pd

from config import ATT1, ATT2, ATT3, ATT4, CACHE_DIR, validate_input_files


# 大文件读取策略：优先使用已有 Parquet，其次使用 LibreOffice 快速转换 CSV，
# 最后才退回 pandas.read_excel。这样首次运行和后续重复运行都尽量节省时间。
SALES_COLUMNS = [
    "销售日期", "扫码销售时间", "单品编码", "销量(千克)",
    "销售单价(元/千克)", "销售类型", "是否打折销售"
]


def _file_signature(source: Path) -> str:
    sig = f"{source.name}|{source.stat().st_size}|{source.stat().st_mtime_ns}"
    return hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]


def _cache_path(source: Path, name: str) -> Path:
    return CACHE_DIR / f"{name}_{_file_signature(source)}.parquet"


def _csv_cache_path(source: Path, name: str) -> Path:
    return CACHE_DIR / f"{name}_{_file_signature(source)}.csv"


def _convert_large_xlsx_with_libreoffice(path: Path, out_csv: Path) -> bool:
    """利用 LibreOffice 的 Calc 引擎把大型 xlsx 转成 CSV。

    对几十万行 Excel，通常比 Python 层逐单元格解析 XLSX XML 更快。
    Windows/Linux/macOS 均兼容，只要系统安装了 LibreOffice。
    """
    if out_csv.exists():
        return True

    office = shutil.which("libreoffice") or shutil.which("soffice")
    if not office:
        return False

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        office,
        "--headless",
        "--convert-to", "csv",
        "--outdir", str(out_csv.parent),
        str(path),
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return False

    # LibreOffice 输出文件名使用原始 stem。
    generated = out_csv.parent / f"{path.stem}.csv"
    if generated.exists() and generated != out_csv:
        generated.replace(out_csv)
    return out_csv.exists()


def _convert_large_xlsx_streaming(path: Path, out_csv: Path):
    """无 LibreOffice 时的纯 Python 后备方案。"""
    if out_csv.exists():
        return

    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                for _, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag.endswith("}si"):
                        text = "".join(
                            t.text or "" for t in elem.iter()
                            if t.tag.endswith("}t")
                        )
                        shared.append(text)
                        elem.clear()

        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        with z.open("xl/worksheets/sheet1.xml") as f, out_csv.open(
            "w", newline="", encoding="utf-8-sig"
        ) as out:
            writer = csv.writer(out)
            for _, row in ET.iterparse(f, events=("end",)):
                if not row.tag.endswith("}row"):
                    continue

                values = []
                for cell in row:
                    if not cell.tag.endswith("}c"):
                        continue
                    typ = cell.attrib.get("t")
                    value = cell.find(ns + "v")
                    val = "" if value is None else (value.text or "")
                    if typ == "s" and val:
                        val = shared[int(val)]
                    values.append(val)

                if values:
                    if values[0] == "销售日期":
                        writer.writerow(values)
                    else:
                        try:
                            n = float(values[0])
                            values[0] = (
                                datetime(1899, 12, 30) + timedelta(days=n)
                            ).strftime("%Y-%m-%d")
                        except (ValueError, TypeError):
                            pass
                        writer.writerow(values)
                row.clear()


def _read_excel_cached(path: Path, sheet_name=0, cache_name=None, **kwargs):
    cache = _cache_path(path, cache_name or path.stem)
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            cache.unlink(missing_ok=True)

    # 普通附件很小，直接读取即可。
    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return df


def _read_large_sales(path: Path):
    cache = _cache_path(path, "sales")
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            cache.unlink(missing_ok=True)

    csv_path = _csv_cache_path(path, "sales")

    # 第一选择：pandas + calamine（Rust 引擎），无需逐单元格 Python 解析。
    try:
        df = pd.read_excel(path, engine="calamine", usecols=SALES_COLUMNS)
        df["单品编码"] = df["单品编码"].astype("string")
        df["销量(千克)"] = pd.to_numeric(df["销量(千克)"], errors="coerce")
        df["销售单价(元/千克)"] = pd.to_numeric(df["销售单价(元/千克)"], errors="coerce")
        df["销售日期"] = pd.to_datetime(df["销售日期"], errors="coerce")
        try:
            df.to_parquet(cache, index=False)
        except Exception:
            pass
        return df
    except (ImportError, ModuleNotFoundError, ValueError, OSError):
        pass

    # 第二选择：LibreOffice 转 CSV，适合本题 88 万行销售流水。
    converted = _convert_large_xlsx_with_libreoffice(path, csv_path)

    # 第三选择：纯 Python XML 流式解析。
    if not converted:
        _convert_large_xlsx_streaming(path, csv_path)

    try:
        df = pd.read_csv(
            csv_path,
            usecols=SALES_COLUMNS,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            csv_path,
            usecols=SALES_COLUMNS,
            encoding="gb18030",
            low_memory=False,
        )

    # 提前压缩 dtype，减少后续 groupby 和 merge 的内存压力。
    df["单品编码"] = df["单品编码"].astype("string")
    df["销量(千克)"] = pd.to_numeric(df["销量(千克)"], errors="coerce")
    df["销售单价(元/千克)"] = pd.to_numeric(df["销售单价(元/千克)"], errors="coerce")
    df["销售日期"] = pd.to_datetime(df["销售日期"], errors="coerce")

    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return df


def load_all():
    # 注意：这里严格固定读取题目规定的四个文件名，不扫描目录、不猜测文件名。
    # 调用config.py中的函数，判断文件是否存在，如果raise Error，那么运行报错退出。
    validate_input_files()
    #用date_io内定义的函数读取各个文件，得到对应的返回值
    goods = _read_excel_cached(ATT1, cache_name="goods")
    sales = _read_large_sales(ATT2)
    wholesale = _read_excel_cached(ATT3, cache_name="wholesale")
    loss_cat = _read_excel_cached(ATT4, sheet_name=0, cache_name="loss_category")
    loss_item = _read_excel_cached(ATT4, sheet_name=1, cache_name="loss_item")

    #改变"单品编码"的类型，改为"string"类型
    for df in (goods, sales, wholesale, loss_item):
        if "单品编码" in df.columns:
            df["单品编码"] = df["单品编码"].astype("string")
    #改变
    sales["销售日期"] = pd.to_datetime(sales["销售日期"], errors="coerce")
    wholesale["日期"] = pd.to_datetime(wholesale["日期"], errors="coerce")
    return goods, sales, wholesale, loss_cat, loss_item


def prepare_sales(sales: pd.DataFrame, goods: pd.DataFrame) -> pd.DataFrame:
    # 论文口径：删除销量为负的 461 条退货记录，仅保留正常销售记录。
    sales = sales.loc[sales["销量(千克)"] >= 0].copy()
    sales = sales.merge(
        goods[["单品编码", "分类名称"]],
        on="单品编码",
        how="left",
        validate="many_to_one",
    )
    sales = sales.dropna(subset=["销售日期", "分类名称"])
    return sales
