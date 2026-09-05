from pathlib import Path
# 该文件目的在于创建Path变量存储路径，便于读取写入。
# 创建文件夹目录，判断文件是否存在，如果不存在data内部应该存在的数据，raise错误

#到达上级目录src的上级目录**代码部分**，即ROOT
ROOT = Path(__file__).resolve().parents[1]
#data
#output/tables
#output/figures
#output/cache
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"

#data下的附件数据
ATT1 = DATA_DIR / "附件1.xlsx"
ATT2 = DATA_DIR / "附件2.xlsx"
ATT3 = DATA_DIR / "附件3.xlsx"
ATT4 = DATA_DIR / "附件4.xlsx"

RANDOM_SEED = 2025
FORECAST_START = "2023-07-01"
FORECAST_END = "2023-07-07"
MIN_ITEMS = 27
MAX_ITEMS = 33
MIN_DISPLAY_KG = 2.5
COVERAGE = 0.75

CATEGORIES = ["花叶类", "辣椒类", "食用菌", "花菜类", "水生根茎类", "茄类"]

#依次创建目录/文件夹，注意从output文件夹开始创建，因为output文件夹是其余要创建的文件的根目录，所以必须先创建根目录。
for p in (OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)

INPUT_FILES = (ATT1, ATT2, ATT3, ATT4)

#判断文件是否存在
def validate_input_files():
    #用列表生成式来生成不存在文件的列表
    missing = [p.name for p in INPUT_FILES if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少题目规定的输入文件：" + ", ".join(missing) +
            "。请将附件1.xlsx、附件2.xlsx、附件3.xlsx、附件4.xlsx放入 data 文件夹。"
        )
