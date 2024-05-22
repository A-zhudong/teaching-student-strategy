import pandas as pd
from sklearn.metrics import r2_score

path_ts = '/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/matbench_perovskite/e_form_test_perov_output_jvpre_CPMpre_mpfinetune.csv'
path_ori = '/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/matbench_perovskite/e_form_test_perov_ori_output_jvpre_CPMpre_mpfinetune.csv'

ts = pd.read_csv(path_ts)
r2_ts = r2_score(ts['target'], ts['pred-0'])
print(f'ts r2 score: {r2_ts}')

ori = pd.read_csv(path_ori)
r2_ori = r2_score(ori['target'], ori['pred-0'])
print(f'ori r2 score: {r2_ori}')