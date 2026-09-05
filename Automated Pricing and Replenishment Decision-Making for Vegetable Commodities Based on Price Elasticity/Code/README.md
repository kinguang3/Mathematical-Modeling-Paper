# C题数学建模代码

本项目严格依据《数学建模论文_C题_建模部分 (1)》及题目附件编写，用于重新计算论文中的表格、图表和模型数据。

## 1. 固定文件结构

```text
C题代码包/
├─ data/
│  ├─ 附件1.xlsx
│  ├─ 附件2.xlsx
│  ├─ 附件3.xlsx
│  └─ 附件4.xlsx
├─ src/
│  ├─ config.py
│  ├─ data_io.py
│  ├─ problem1.py
│  ├─ problem2.py
│  ├─ problem3.py
│  └─ report.py
├─ main.py
├─ requirements.txt
└─ README.md
```

## 2. 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

## 3. 大数据优化

附件2约88万条销售流水。程序优先尝试 `pandas + python-calamine`，失败后尝试 LibreOffice 转 CSV，最后才使用纯 Python XML 流式解析。

第一次成功读取后会生成 Parquet 缓存，之后再次运行直接读取缓存，避免重复解析大型 Excel。

程序全程使用 `pathlib.Path` 管理路径，不写死 Windows 路径。

## 4. 论文口径

- 问题一：删除销量为负的退货记录；品类统计按有销售日期计算；品类 Pearson 相关系数按共同有销售日期计算。
- 问题二：7天周期 Holt-Winters；双对数价格弹性；成本加成定价与损耗约束。
- 问题三：使用2023-06-24至06-30的49个可售品种；周六放大系数使用完整历史数据的周六/工作日均值；27~33个单品；单品补货量至少2.5 kg；各品类覆盖率至少75%；使用 SciPy HiGHS MILP。

## 5. 输出

运行后生成 `output/`，其中包括：

- `C题_全部计算表格.xlsx`
- `tables/` 中的各张计算表 CSV
- `figures/` 中的论文图表 PNG
- `运行说明.txt`

程序最后会输出“运行结束”。
