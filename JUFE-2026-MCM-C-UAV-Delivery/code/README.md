# 城市低空即时配送网络优化 - 代码说明

## 项目结构

```
code/
├── main.py              # 主入口脚本
├── model/               # 数学模型
│   ├── network.py       # 网络构建与管理
│   ├── capacity_model.py # 容量规划模型
│   └── scheduling_model.py # 调度优化模型
├── algorithm/           # 求解算法
│   ├── genetic_algorithm.py
│   ├── column_generation.py
│   └── rolling_horizon.py
├── simulation/          # 仿真模块
│   ├── simulator.py     # 仿真器
│   └── visualization.py # 可视化工具
└── utils/               # 工具函数
    ├── data_loader.py   # 数据加载
    └── metrics.py       # 指标计算
```

## 运行方式

### 安装依赖

```bash
pip install numpy scipy pulp networkx matplotlib
```

### 运行主程序

```bash
# 解决所有问题
python main.py

# 只解决问题一
python main.py --problem 1

# 解决所有问题并生成可视化图表
python main.py --visualize
```

## 模块说明

### model/network.py
- `DeliveryNetwork`: 配送网络类，管理节点、边、飞行时间和容量

### model/capacity_model.py
- `CapacityPlanningModel`: 容量规划模型，求解最小无人机数量

### model/scheduling_model.py
- `SchedulingModel`: 调度模型，处理初始调度和扰动重调度

### algorithm/genetic_algorithm.py
- `GeneticAlgorithm`: 遗传算法求解器

### algorithm/column_generation.py
- `ColumnGeneration`: 列生成算法

### algorithm/rolling_horizon.py
- `RollingHorizonScheduler`: 滚动时域调度器

### simulation/simulator.py
- `DroneSimulator`: 无人机仿真器
- `WeatherSimulator`: 气象事件模拟器

### utils/data_loader.py
- `DataLoader`: 数据加载工具，支持CSV、JSON格式

### utils/metrics.py
- `MetricsCalculator`: 性能指标计算
- `RobustnessAnalyzer`: 网络韧性分析

## 输入数据格式

### city_network.csv
```csv
from,to,time,capacity
H,P1,10,10
H,P2,15,8
P1,H,10,10
P2,H,15,8
```

### demand_pattern.csv
```csv
node,demand
P1,100
P2,150
```

### drone_specs.json
```json
{
    "capacity": 20,
    "speed": 60,
    "battery_time": 120
}
```

## 输出结果

运行结果保存在 `../data/results/` 目录下：
- `capacity_optimal.csv`: 容量规划最优解
- `schedule_log.csv`: 调度日志
- `figures/`: 可视化图表