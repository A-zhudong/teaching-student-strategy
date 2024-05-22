import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('/home/zd/zd/teaching_net/CrabNet/data_ratio_4props.xlsx', index_col=0)

# 创建一个2x2的子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 定义数据集和标题
# datasets = [('MP_e_form', 'mpef_ori'), ('MP_bandgap', 'mpbg_ori'), ('JV_e_form', 'jvef_ori'), ('JV_bandgap', 'jvbg_ori')]
titles = ['a', 'b', 'c', 'd']
# sub_titles = ['MP_e_form vs mpef_ori', 'MP_bandgap vs mpbg_ori', 'JV_e_form vs jvef_ori', 'JV_bandgap vs jvbg_ori']
datasets = [('MP_e_form', 'mpef_ori'), ('MP_bandgap', 'mpgp_ori'), ('JV_e_form', 'jvef_ori'), ('JV_bandgap', 'jvbg_ori')]
sub_titles = ['MP_e_form vs mpef_ori', 'MP_bandgap vs mpbg_ori', 'JV_e_form vs jvef_ori', 'JV_bandgap vs jvbg_ori']
# 条形图的宽度
bar_width = 0.35

# x轴的位置
ind = np.arange(len(df.columns))

# 根据图像读取，设置Y轴的范围
y_max_values = [0.15, 0.7, 0.25, 0.4]
y_min_values = [0.03, 0.25, 0.05, 0.1]
y_lims = [(y_min, y_max * 1.1) for y_min,y_max in zip(y_min_values,y_max_values)]  # 留出10%的空间

for i, ax in enumerate(axs.flatten()):
    dataset = datasets[i]
    
    # 读取每组数据
    y1 = df.loc[dataset[0]]
    y2 = df.loc[dataset[1]]
    
    # 绘制条形图
    ax.bar(ind - bar_width/2, y1, bar_width, label=dataset[0], color='#FF5A33', alpha=0.8)
    ax.bar(ind + bar_width/2, y2, bar_width, label=dataset[1], color='#B4CF66', alpha=0.6)
    
    # 设置标题（子图下方）
    ax.set_xlabel(sub_titles[i], labelpad=20, fontsize=16)
    
    # 添加子图标签（左上角）
    ax.text(-0.1, 1.05, titles[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    # 设置y轴的范围
    ax.set_ylim(*y_lims[i])

    # 设置x轴刻度
    ax.set_xticks(ind)
    ax.set_xticklabels(df.columns, fontsize=12, rotation=45)
    
    # 调整图框，去除上端和右端的框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('4props_barcharts_adjusted_min.png', dpi=400)
