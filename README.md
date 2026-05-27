# 城市低空即时配送网络优化

> 数学建模竞赛 C 题 - 城市低空即时配送网络的容量规划与调度优化

## 项目简介

本项目针对城市低空配送系统设计了一套从静态规划到动态调度再到韧性加固的优化方案，包含三个核心问题的建模与求解：

1. **问题一：网络流规划与最小机队配置** - 在理想条件下求解满足所有需求所需的最小无人机数量
2. **问题二：气象扰动下的实时调度** - 设计算法处理突发气象导致的航段关闭问题
3. **问题三：关键节点识别与韧性评估** - 定义网络韧性指标并提出加固策略

## 目录结构

```
JUFE2026-C-LowAir-Delivery-Opt/
├── README.md                    # 项目说明
├── LICENSE                      # 开源协议
├── requirements.txt             # Python依赖
├── code/                        # 核心代码
│   ├── main.py                  # 主入口脚本
│   ├── model/                   # 数学模型
│   ├── algorithm/               # 求解算法
│   ├── simulation/              # 仿真模块
│   └── utils/                   # 工具函数
├── data/                        # 数据文件
│   ├── raw/                     # 原始数据
│   ├── processed/               # 预处理数据
│   └── results/                 # 运行结果
├── paper/                       # 论文相关
└── notebooks/                   # Jupyter笔记本
```

## 前置条件

### 环境要求
- Python 3.9+
- Git

### 依赖包
```bash
numpy>=1.21.0
scipy>=1.7.0
pulp>=2.7.0
networkx>=2.6.0
matplotlib>=3.4.0
python-docx>=0.8.11
```

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/kinguang3/JUFE-C-UAV-CapSched.git
cd JUFE2026-C-LowAir-Delivery-Opt
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行主程序

```bash
# 解决所有问题
python code/main.py

# 仅解决特定问题
python code/main.py --problem 1   # 问题一：容量规划
python code/main.py --problem 2   # 问题二：实时调度
python code/main.py --problem 3   # 问题三：韧性评估

# 生成可视化图表
python code/main.py --visualize
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--problem` | 指定要解决的问题（1/2/3），默认全部解决 |
| `--visualize` | 生成可视化图表 |

## 输出文件

运行后会在 `data/results/` 目录生成以下文件：

| 文件 | 内容 |
|------|------|
| `capacity_optimal.csv` | 问题一最优机队配置 |
| `schedule_log.csv` | 问题二调度日志 |
| `resilience_results.csv` | 问题三韧性评估结果 |
| `figures/network.png` | 网络拓扑图 |
| `figures/resilience.png` | 韧性指标分布图 |

## 注意事项

### 1. 求解器配置
- 使用 PuLP 库调用 CBC MILP 求解器
- 确保系统有足够内存处理大规模问题

### 2. 数据文件
- 原始数据位于 `data/raw/` 目录
- 运行结果会自动保存到 `data/results/` 目录
- 可视化图表需要 `--visualize` 参数才能生成

### 3. 环境变量
- 建议在虚拟环境中运行，避免依赖冲突
- 首次运行可能需要较长时间安装依赖

### 4. 性能建议
- 对于大规模网络，建议使用遗传算法（GA）求解
- 可通过调整参数平衡求解精度与速度

## 算法说明

| 问题 | 算法 | 说明 |
|------|------|------|
| 问题一 | 线性规划 + 遗传算法 | 精确求解 + 启发式验证 |
| 问题二 | 贪心重分配算法 | GCFR 算法处理扰动 |
| 问题三 | 敏感性分析 | 节点失效影响评估 |

## 许可证

MIT License

## 作者

- 项目维护：kinguang3
- 仓库地址：https://github.com/kinguang3/JUFE-C-UAV-CapSched