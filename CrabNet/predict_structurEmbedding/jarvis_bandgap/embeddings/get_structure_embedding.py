import pickle
import numpy as np
import torch

path_alignn_embedding = r'/home/zd/zd/teaching_net/data/jarvis/jarvis_bandgap/jarvis_eval_formula_embedding.pkl'
path_crab_embedding = r'/home/zd/zd/teaching_net/CrabNet/predict_structurEmbedding/jarvis_bandgap/embeddings/jarvis_crabnet_oriEmbedding.pkl'

with open(path_alignn_embedding, 'rb') as af:
    alignn_embedding = pickle.load(af)

with open(path_crab_embedding, 'rb') as cf:
    crab_embedding = pickle.load(cf)

structure_embedding = {}
for key,value in crab_embedding.items():
    structure_embedding.update({key:(alignn_embedding[key].cpu().numpy() - value)})
print(len(structure_embedding))

with open('jarvis_structureEmbedding.pkl', 'wb') as f:
    pickle.dump(structure_embedding, f)