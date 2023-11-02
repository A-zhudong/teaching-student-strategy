import numpy as np
import matplotlib.pyplot as plt
# plt.rc('font',family='Times New Roman')

from sklearn.manifold import TSNE
import pickle
import glob, os
import pandas as pd
import torch

def concatemb(embeddings):
    if torch.is_tensor(embeddings[0]):
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    else:
        embeddings = np.concatenate([embedding.reshape(1,-1) for embedding in embeddings], axis=0)
    return embeddings

folder_path = '/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/mp_e_form/real_ma_e_form/mp_e_form'  
csv_files = glob.glob(folder_path + '/*.csv')

path_emb_ori = '/home/zd/zd/teaching_net/CrabNet/TSNE_visual/mp_0.05_e_form_embeddings/ori_ratio0.05_test.pkl'
with open(path_emb_ori, 'rb') as f:
    ori_embedding = pickle.load(f)

path_emb_finetune = '/home/zd/zd/teaching_net/CrabNet/TSNE_visual/mp_0.05_e_form_embeddings/pretrain_finetune_embloss_alpha6_ratio0.05_test.pkl'
with open(path_emb_finetune, 'rb') as f:
    fin_embedding = pickle.load(f)

path_emb_alignn = '/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/mp_e_form/real_ma_e_form/mp_e_form/mp_eval_formular_embedding.pkl'
with open(path_emb_alignn, 'rb') as f:
    ali_embedding = pickle.load(f)
    print('ali: ', len(ali_embedding))

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
formu_prop = pd.concat(dfs, ignore_index=True)
formu_prop = formu_prop.set_index(formu_prop.columns[0])[formu_prop.columns[1]].to_dict()

formulas = list(ori_embedding.keys())
embeddings_ori = list(ori_embedding.values())
embeddings_ori = concatemb(embeddings_ori)

embeddings_fin = list(fin_embedding.values())
embeddings_fin = concatemb(embeddings_fin)

# embeddings_ali = list(ali_embedding.values())
embeddings_ali = [ali_embedding[formula] for formula in formulas]
embeddings_ali = concatemb(embeddings_ali)
print(f'embeddings_ali shape: {embeddings_ali.shape}')
print(f'embeddings_ori shape: {embeddings_ori.shape}')

emb_fin_ori = embeddings_fin - embeddings_ori
emb_ali_ori = embeddings_ali - embeddings_ori

outputs = np.array([formu_prop[key] for key in formulas])
# duplicates = formu_prop[formu_prop.duplicated('formula')]['formula']
# print(duplicates)
print(formulas[0], len(formu_prop))
print(np.sum(outputs != None))
# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)

# for the fin and ori embeddings
# embeddings_2d_ori = tsne.fit_transform(embeddings_ori)
# embeddings_2d_fin = tsne.fit_transform(embeddings_fin)

# for the reduction of ali minus ori and fin minus ori
embeddings_2d_ori = tsne.fit_transform(emb_fin_ori)
embeddings_2d_fin = tsne.fit_transform(emb_ali_ori)

# for the comparation of the ali and ori embeddings
# embeddings_2d_ori = tsne.fit_transform(embeddings_ori)
# embeddings_2d_fin = tsne.fit_transform(embeddings_ali)

# 计算颜色
# 我们将outputs归一化到[0, 1]，然后使用它们作为颜色
# colors = (outputs - outputs.min()) / (outputs.max() - outputs.min())
colors = outputs

# 绘制结果
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='viridis')
# plt.colorbar(label='Output value')
fig, axs = plt.subplots(1, 2, figsize=(22, 10), dpi=300)

sc = axs[0].scatter(embeddings_2d_ori[:, 0], embeddings_2d_ori[:, 1], c=colors, cmap='viridis',\
                     alpha=0.5, s=10, label='Embeddings original')
axs[0].text(0.5, 0, 'Embeddings TS-ori', fontsize=24, ha='center', va='center', transform=axs[0].transAxes)
sc1 = axs[1].scatter(embeddings_2d_fin[:, 0], embeddings_2d_fin[:, 1], c=colors, cmap='viridis',\
                      alpha=0.5, s=10, label='Embeddings alignn')
# axs[1].legend(loc='upper right', fontsize=24)
axs[1].text(0.5, 0, 'Embeddings ali-ori', fontsize=24, ha='center', va='center', transform=axs[1].transAxes)
# 设置xlim和ylim，使得两个绘图框都是正方形
axs[0].set_xlim([-100, 100])
axs[0].set_ylim([-100, 100])
axs[1].set_xlim([-100, 100])
axs[1].set_ylim([-100, 100])

# 设置aspect属性，使得两个绘图框都是正方形
axs[0].set_aspect('equal', adjustable='datalim')
axs[1].set_aspect('equal', adjustable='datalim')

# 去掉边框和刻度
axs[0].axis('off')
axs[1].axis('off')

# 调整子图之间的距离
plt.subplots_adjust(wspace=0)  # wspace 控制宽度间距

cbar = plt.colorbar(sc1, ax=axs, aspect=10, shrink=0.68)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Output value', size=24)


path_figsave = '/home/zd/zd/teaching_net/CrabNet/TSNE_visual/mp_0.05_e_form_embeddings'
plt.savefig(os.path.join(path_figsave ,'aliFinOri_minus_adjust.png'))
#aliFinOri_minus_adjust.png comparation_ori_ali_adjust.png