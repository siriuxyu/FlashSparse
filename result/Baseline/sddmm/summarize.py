# Summarize Rode, Gespmm, Sputnik, Advisor, DTC, TCGNN, cuSPARSE
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
import os
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)
dimN_list = [32, 128]

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
project_dir = project_dir + '/result/Baseline/sddmm'
for dimN in dimN_list:
       #RoDe, Sputnik
       df1 = pd.read_csv(project_dir + '/rode_sddmm_f32_n' + str(dimN) + '_res.csv')
       
       # TC-GNN
       df2 = pd.read_csv(project_dir + '/base_sddmm_f32_n' + str(dimN) + '.csv')
       df_res = pd.merge(df1, df2, on='dataSet', how='inner') 
       
       df_res1= df_res[['dataSet', 'num_nodes', 'num_edges', 'sputnik',
               'rode', 'tcgnn']]
       
       df_res1.to_csv(project_dir + '/all_baseline_' + str(dimN) + '.csv', index=False) 
