import matplotlib.pyplot as plt

# 数据
x = [0, 5, 10, 15, 20]
y = [0.0588, 0.0491, 0.0486, 0.0476, 0.0488]

# 颜色，透明度，线宽和点的大小
color = '#FF5A33'
alpha = 0.5
linewidth = 2.0
point_size = 100  # 点的大小

plt.plot(x[0:2], y[0:2], color='blue', alpha=0.35, linewidth=linewidth)  # 将第一条线段的颜色改为蓝色
plt.plot(x[1:], y[1:], color=color, alpha=alpha, linewidth=linewidth)

plt.scatter(x[0], y[0], color='blue', alpha=alpha, s=point_size)  # 为第一个点指定不同的颜色
plt.scatter(x[1:], y[1:], color=color, alpha=alpha, s=point_size)  # 其余点使用相同的颜色

# 为每个点添加标签
for i in range(len(x)):
    plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom')

plt.ylim(0.042, 0.06)
plt.xlabel('alpha')
plt.ylabel('MAE')

plt.savefig('line_alpha_0.png', dpi=400)

