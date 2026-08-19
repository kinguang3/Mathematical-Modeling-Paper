import matplotlib.pyplot as plt

# 数据
labels = [
    '梯田',
    '平旱地',
    '山坡地',
    '水浇地',
    '普通大棚',
    '智慧大棚'
]
sizes = [619, 365, 108, 109, 10, 2]
colors = [
    '#d48b29',
    '#97b382',
    '#b86c34',
    '#4472b0',
    '#b481c4',
    '#38b1bc'
]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=60,
    pctdistance=0.7,
    labeldistance=1.1
)

# 替换饼图内部文字，实现百分比+亩数
inner_text = [
    "51.0%\n619亩",
    "30.1%\n365亩",
    "8.9%\n108亩",
    "9.0%\n109亩",
    "0.8%\n10亩",
    "0.2%\n2亩"
]
for idx, txt in enumerate(autotexts):
    txt.set_text(inner_text[idx])

ax.set_title("图1 乡村各类耕地面积占比", fontsize=16, pad=20)

# 底部注释
note = "注：全村耕地总规模1213亩；露天地块34块合计1201亩，大棚20个合计12亩"
plt.figtext(0.5, 0.05, note, ha="center", fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()