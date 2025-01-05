import os.path as osp
import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import pandas as pd
import time
BLK_H = 16
BLK_W = 8
import DTCSpMM
from dataset import *
import sys
import time

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(current_dir))

df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
df = pd.read_csv(project_dir + '/result/ref/baseline_h100_spmm_256.csv')

dimN = int(sys.argv[1])
print('dimN: ' + str(dimN))

file_name = project_dir + '/result/Baseline/spmm/dtc_spmm_f32_n' + str(dimN) + '.csv'

head = ['dataSet', 'num_nodes', 'num_edges', 'dtc']
with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)

start_time = time.time()
for index, row in df.iterrows():
    res_temp = []
    res_temp.append(row.iloc[0])
    res_temp.append(row.iloc[1])
    res_temp.append(row.iloc[2])
    
    print(row.iloc[0])
    ## Load matrix from files.
    data = row.iloc[0]
    # Set your own path to the dataset.
    path = osp.join(project_dir, 'dataset', data + ".npz") #4090
    matrix = DTC_dataset(path)
    num_rows = matrix.num_nodes
    num_nnz = matrix.num_edges

    column_index =  matrix.column_index 
    row_pointers =  matrix.row_pointers 
    # Process data.】

    num_row_windows = (num_rows + BLK_H - 1) // BLK_H
    edgeToColumn = torch.zeros(num_nnz, dtype=torch.int)
    edgeToRow = torch.zeros(num_nnz, dtype=torch.int)
    blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
    column_index_ori  = column_index.cuda()
    row_pointers_ori = row_pointers.cuda()

    blockPartition_cuda  = blockPartition.cuda()
    edgeToColumn_cuda = edgeToColumn.cuda()
    edgeToRow_cuda  = edgeToRow.cuda()

    # Optimize GPU.
    RowWindowOffset, TCblockRowid,\
        TCblocktileId, TCblockoffset, SparseAToXindex,\
            block_count = DTCSpMM.preprocess_gpu(column_index_ori, row_pointers_ori, num_rows, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)


    X = torch.ones((num_rows, dimN)).cuda()
    # Run test.

    balance_choice = True
    # exeplan = ExecutionPlan[dset_name][dimN][1] + "_" + ExecutionPlan[dset_name][dimN][2]
    exeplan = "float4" + "_" + "split"
    if balance_choice == False:
        _, dtc_spmm = DTCSpMM.run_DTCSpMM(X, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, num_nnz, exeplan)
    else:
        _, dtc_spmm = DTCSpMM.run_DTCSpMM_balance(X, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, exeplan)

    res_temp.append(dtc_spmm.item())
    with open(file_name, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(res_temp)
    
    print(data)


end_time = time.time()
execution_time = end_time - start_time

# Record execution time.
with open("execution_time_base.txt", "a") as file:
    file.write("Baseline-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")