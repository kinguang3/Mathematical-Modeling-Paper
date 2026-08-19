# 数学建模竞赛项目仓库

> 本仓库收录数学建模竞赛的完整参赛项目：赛题、论文、代码与说明文档。

## 项目列表

| 项目 | 竞赛 | 赛题 | 核心方法 |
|------|------|------|----------|
| [JUFE-2026-MCM-C-UAV-Delivery](./JUFE-2026-MCM-C-UAV-Delivery) | 2026 江西财经大学数学建模竞赛 C 题 | 城市低空即时配送网络的容量规划与调度优化 | ILP + 贪心重调度（GCFR）+ 韧性指标（NFRI） |
| [Crop-Planting-Optimization-2024](./Crop-Planting-Optimization-2024) | 2024 高教社杯全国大学生数学建模竞赛 C 题 | 农作物的种植策略 | 滚动混合整数线性规划（MILP）+ 蒙特卡洛 CVaR |

---

## 1. JUFE-2026-MCM-C-UAV-Delivery

### 项目简介

针对城市低空无人机配送网络，设计从**静态规划到动态调度再到韧性加固**的三层优化方案：

1. **问题一：网络流规划与最小机队配置** - 在理想条件下求解满足所有需求所需的最小无人机数量（ILP 模型 + 显式解析解）
2. **问题二：气象扰动下的实时调度** - 贪心容量优先重分配算法（GCFR）处理突发航段关闭，最小化订单延误
3. **问题三：关键节点识别与韧性评估** - 定义节点失效韧性指数 NFRI，识别关键节点并设计迭代加固策略

### 核心结果

| 问题 | 方法 | 结果 |
|------|------|------|
| 问题一 | ILP（周转率显式解） | n=4 时最小机队 Z\*=4 架，等间隔流水线发车 |
| 问题二 | GCFR 贪心算法 O(nlogn) | 示例 3 轮迭代吸收 7 件，仅延误 5 件（1 小时） |
| 问题三 | NFRI 指标 + 迭代加固 | 关键节点 k\*=5（NFRI=0.909），两轮加固后 NFRImin → 1.000 |

### 目录结构

```
JUFE-2026-MCM-C-UAV-Delivery/
├── code/                        # 核心代码
│   ├── main.py                  # 主入口脚本
│   ├── model/                   # 数学模型（network / capacity / scheduling）
│   ├── algorithm/               # 求解算法（GA / 列生成 / 滚动时域）
│   ├── simulation/              # 仿真与可视化
│   └── utils/                   # 工具函数（数据加载 / 韧性指标）
├── data/
│   └── results/                 # 运行结果（CSV + 图表）
├── notebooks/                   # 数学建模介绍.md / 结构.md
└── paper/C题.docx               # 赛题原文与完整论文
```

### 运行方式

```bash
cd JUFE-2026-MCM-C-UAV-Delivery/code
pip install numpy scipy pulp networkx matplotlib

python main.py                     # 解决所有问题
python main.py --problem 1         # 仅问题一
python main.py --visualize         # 附加生成可视化图表
```

运行结果输出至 `data/results/`：`capacity_optimal.csv`、`schedule_log.csv`、`resilience_results.csv`、`figures/`。

详细说明见 [notebooks/数学建模介绍.md](./JUFE-2026-MCM-C-UAV-Delivery/notebooks/数学建模介绍.md)。

---

## 2. Crop-Planting-Optimization-2024

### 项目简介

针对华北某乡村农作物种植问题，建立**多年滚动混合整数线性规划（MILP）**模型，综合考虑地块类型适宜性、重茬禁止、豆类三年轮作、水浇地互斥等约束，以最大化 2024–2030 年总利润为目标：

1. **问题一**：参数稳定下分两种情景（超额滞销 / 超额半价出售）求解最优种植方案
2. **问题二**：考虑销量增长、产量波动、成本上升、价格分类变化等不确定性，求解稳健方案
3. **问题三**：引入作物替代/互补性与销售量-价格-成本相关性，蒙特卡洛模拟 + CVaR 风险决策

### 核心结果

| 问题 | 方法 | 总利润 |
|------|------|--------|
| 问题一(1) | MILP（超额滞销） | 约 4142 万元 |
| 问题一(2) | MILP（超额半价） | 约 6219 万元 |
| 问题二 | MILP（期望参数确定性等价） | 约 6642 万元 |
| 问题三 | 蒙特卡洛 + CVaR₀.₉₅ | 期望约 6580 万元，CVaR 改善约 7.7% |

灵敏度分析表明**价格是最关键因素**（每变动 10%，利润变动约 11%）。

### 目录结构

```
Crop-Planting-Optimization-2024/
├── code/                        # 核心代码
│   ├── C题代码问题1,2代码.py     # 问题1/2：滚动 MILP（scipy.optimize.milp）
│   ├── 问题3_完整蒙特卡洛_CVaR完整代码.py  # 问题3：1000 次蒙特卡洛 + CVaR
│   └── 三个绘图脚本.py           # 饼图 / 利润对比折线图 / 灵敏度分析
├── paper/                       # 论文.pdf + C题建模报告..docx
└── notebook/                    # 项目介绍.md
```

### 运行方式

```bash
cd Crop-Planting-Optimization-2024/code
pip install numpy pandas scipy openpyxl matplotlib

python C题代码问题1,2代码.py                  # 问题1/2（需附件1.xlsx、附件2.xlsx）
python 问题3_完整蒙特卡洛_CVaR完整代码.py     # 问题3（需先运行问题1/2生成 result2）
```

详细说明见 [notebook/项目介绍.md](./Crop-Planting-Optimization-2024/notebook/项目介绍.md)。

---

## 环境要求

- Python 3.9+
- 依赖：`numpy`、`pandas`、`scipy`、`openpyxl`、`matplotlib`（JUFE 项目另需 `pulp`、`networkx`）

## 许可证

MIT License

## 作者

- 项目维护：kinguang3
- 仓库地址：https://github.com/kinguang3/JUFE-C-UAV-CapSched