import matplotlib.pyplot as plt
import pandas as pd

# 从Excel文件中读取数据
df = pd.read_excel('/home/zd/zd/teaching_net/CrabNet/data_ratio_4props.xlsx', index_col=0)
print(df)
# 创建一个 2x2 的子图
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 在每个子图中绘制两条线
datasets = [('MP_e_form', 'mpef_ori'), ('MP_bandgap', 'mpgp_ori'), ('JV_e_form', 'jvef_ori'), ('JV_bandgap', 'jvbg_ori')]
titles = ['MP_e_form vs mpef_ori', 'MP_bandgap vs mpbg_ori', 'JV_e_form vs jvef_ori', 'JV_bandgap vs jvbg_ori']
print(df.loc[datasets[0][0]])
print(df.loc[datasets[0][1]])
for i, ax in enumerate(axs.flatten()):
    dataset = datasets[i]
    ax.plot(df.columns, df.loc[dataset[0]], marker='o', color='#FF5A33', label=dataset[0])
    ax.plot(df.columns, df.loc[dataset[1]], marker='o', color='black', label=dataset[1])
    ax.set_title(titles[i], fontsize=18)
    ax.legend(fontsize=20)
    ax.set_xticks([0.05, 0.25, 0.5, 0.75, 1])

# 显示图表
plt.tight_layout()

# 显示图表
plt.tight_layout()
plt.savefig('4props_xticks_size20.png', dpi=400)
