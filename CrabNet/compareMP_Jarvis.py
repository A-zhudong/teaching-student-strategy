import pandas as pd
import glob

# 创建一个空集合
values_MP = set()
values_Jarvis = set()

# 获取文件夹下的所有CSV文件
files_MP = glob.glob('/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/mp_bandgap/*.csv')
files_Jarvis = glob.glob('/home/zd/zd/teaching_net/data/jarvis/jarvis_bandgap/*.csv')

# 遍历所有文件
for file in files_MP:
    # 读取CSV文件
    df = pd.read_csv(file)
    
    # 获取第一列的值并添加到集合中
    values_MP.update(df.iloc[:, 0].values)

for file in files_Jarvis:
    # 读取CSV文件
    df = pd.read_csv(file)
    
    # 获取第一列的值并添加到集合中
    values_Jarvis.update(df.iloc[:, 0].values)

intersection = values_Jarvis & values_MP
print(len(intersection), len(values_Jarvis), len(values_MP))

df = pd.DataFrame(list(intersection))

# 保存数据框为CSV文件
df.to_csv('/home/zd/zd/teaching_net/data/intersection_MP_JARVIS.csv', index=False, header=False)