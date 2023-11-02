# 数据
# x = ['5%', '25%', '50%', '75%', '100%',]
import matplotlib.pyplot as plt
import numpy as np

# 数据
x_labels = ['5%', '25%', '50%', '75%', '100%',]
x = np.arange(len(x_labels))*0.2
tops = [0.139, 0.082, 0.070, 0.062, 0.058]
bottoms = [0.093, 0.068, 0.059, 0.052, 0.048]
heights = [tops[i]-bottoms[i] for i in range(len(bottoms))]

# 颜色和透明度
colors = ['#FF5A33', '#FF5A33', '#FF5A33', '#FF5A33', '#FF5A33']
alphas = [0.9, 0.7, 0.5, 0.3, 0.2]

for i in range(len(x)):
    plt.bar(x[i], heights[i], bottom=bottoms[i], color=colors[i], alpha=alphas[i], width=0.1)
    plt.text(x[i], bottoms[i], str(bottoms[i]), ha='center', va='top', fontsize=6)
    plt.text(x[i], bottoms[i]+heights[i], str(bottoms[i]+heights[i]), ha='center', va='bottom', fontsize=6)

plt.ylim(0.04, 0.16)
plt.xlabel('percentages')
plt.ylabel('MAE')

# 设置 x 轴标签
plt.xticks(x, x_labels)

# 去掉右边和上边的框线
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)

# plt.title('The MAE changes of different ratios of training data', ha='center', va='top', fontsize=10)
# plt.text('50%', -0.1, 'Bar Chart with Different Alphas and Labeled Bottoms and Tops', ha='center')

plt.savefig('bar.png', dpi=400)
