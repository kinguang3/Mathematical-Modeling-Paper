import matplotlib.pyplot as plt

# ---------------------- 数据 ----------------------
years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
# 三条曲线数据
data_stagnate = [595, 595, 595, 595, 595, 595, 595]      # 问题1-滞销
data_half = [1010, 730, 1008, 735, 1008, 733, 1008]     # 问题1-半价
data_uncertain = [1008, 742, 1058, 768, 1112, 798, 1175]# 问题2-不确定性

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14,7), dpi=140)

# 绘制三条线，标记与原图匹配
ax.plot(years, data_stagnate, color='#c83c3c', marker='o', linestyle='-', label="问题1-滞销", linewidth=2, markersize=8)
ax.plot(years, data_half, color='#284b94', marker='s', linestyle='-', label="问题1-半价", linewidth=2, markersize=8)
ax.plot(years, data_uncertain, color='#3e893e', marker='^', linestyle='-', label="问题2-不确定性", linewidth=2, markersize=8)

ax.set_title("2024-2030年三种方案利润对比", fontsize=14)
ax.set_xlabel("年份", fontsize=12)
ax.set_ylabel("年利润（万元）", fontsize=12)

ax.set_xticks(years)
ax.set_ylim(570, 1200)
ax.set_yticks([600,700,800,900,1000,1100])

ax.grid(alpha=0.3)
ax.legend(loc="upper left")

plt.tight_layout()
plt.show()