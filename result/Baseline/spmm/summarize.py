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
dimN_list = [128, 256]

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
project_dir = project_dir + '/result/Baseline/spmm'
for dimN in dimN_list:
       #RoDe, cuSPARSE, Sputnik
       df1 = pd.read_csv(project_dir + '/rode_spmm_f32_n' + str(dimN) + '_res.csv')
       
       # Advisor, geSpMM, TC-GNN
       df2 = pd.read_csv(project_dir + '/base_spmm_f32_n' + str(dimN) + '.csv')
       df_res = pd.merge(df1, df2, on='dataSet', how='inner') 
       
       df_res1= df_res[['dataSet', 'num_nodes', 'num_edges', 'sputnik',
              'cusparse', 'rode', 'gespmm', 'advisor', 'tcgnn']]

       # dtc
       dtc = pd.read_csv(project_dir + '/dtc_spmm_f32_n' + str(dimN) + '.csv')
       dtc= dtc[['dataSet', 'dtc']]

       df_res2 = pd.merge(df_res1, dtc, on='dataSet', how='inner')  # 使用内连接（inner join）
       df_res2= df_res2[['dataSet', 'num_nodes', 'num_edges', 'sputnik',
              'cusparse', 'rode', 'gespmm', 'advisor', 'tcgnn', 'dtc']]
       df_res2.to_csv(project_dir + '/all_baseline_' + str(dimN) + '.csv', index=False) 
