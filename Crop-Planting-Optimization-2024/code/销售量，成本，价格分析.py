import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 数据 ----------------------
# 销售量灵敏度（蓝色，圆形标记）
x_sale = [0.8, 0.9, 1.0, 1.1, 1.2]
y_sale = [976, 991, 1005, 1019, 1032]

# 成本灵敏度（红色，方形标记）
x_cost = [0.8, 0.9, 1.0, 1.1, 1.2]
y_cost = [1026, 1015, 1005, 995, 986]

# 价格灵敏度（绿色，三角标记）
x_price = [0.8, 0.9, 1.0, 1.1, 1.2]
y_price = [780, 895, 1002, 1115, 1230]

# ---------------------- 子图布局，三张并排 ----------------------
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 6), dpi=120)

# 子图1：销售量灵敏度
ax1.plot(x_sale, y_sale, color='#224499', marker='o', linestyle='-', linewidth=2.2, markersize=8)
ax1.set_title("销售量灵敏度")
ax1.set_xlabel("销售量缩放比例")
ax1.set_ylabel("利润（万元）")
ax1.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
ax1.axvline(x=1.0, linestyle='--', color='#aaaaaa')
ax1.grid(alpha=0.3)

# 子图2：成本灵敏度
ax2.plot(x_cost, y_cost, color='#c83c3c', marker='s', linestyle='-', linewidth=2.2, markersize=8)
ax2.set_title("成本灵敏度")
ax2.set_xlabel("成本缩放比例")
ax2.set_ylabel("利润（万元）")
ax2.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
ax2.axvline(x=1.0, linestyle='--', color='#aaaaaa')
ax2.grid(alpha=0.3)

# 子图3：价格灵敏度
ax3.plot(x_price, y_price, color='#3e893e', marker='^', linestyle='-', linewidth=2.2, markersize=8)
ax3.set_title("价格灵敏度")
ax3.set_xlabel("价格缩放比例")
ax3.set_ylabel("利润（万元）")
ax3.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
ax3.axvline(x=1.0, linestyle='--', color='#aaaaaa')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()