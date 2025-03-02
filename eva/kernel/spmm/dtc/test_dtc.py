import os.path as osp
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import pandas as pd
import time
BLK_H = 16
BLK_W = 8
import DTCSpMM
import sys
import time

import os
sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import Dataset

# print(torch.cuda.is_available())

# current_dir = os.path.dirname(__file__)
# baseline_dir = os.path.dirname(current_dir)
# project_dir = os.path.dirname(os.path.dirname(current_dir))


def kernel(inputInfo, epoches):
    num_rows = inputInfo.num_nodes
    num_nnz = inputInfo.num_edges
    
    column_index = inputInfo.column_index
    row_pointers = inputInfo.row_pointers

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


    X = inputInfo.x
    X = X.cuda()

    balance_choice = True
    # exeplan = ExecutionPlan[dset_name][dimN][1] + "_" + ExecutionPlan[dset_name][dimN][2]
    exeplan = "float4" + "_" + "split"
    if balance_choice == False:
        _, dtc_spmm = DTCSpMM.run_DTCSpMM(X, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, num_nnz, exeplan)
    else:
        _, dtc_spmm = DTCSpMM.run_DTCSpMM_balance(X, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, exeplan)

    return round(dtc_spmm.item(), 4)



def test(data, epoches, dimN, data_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = Dataset(data_path)
    inputInfo.init_embedding(dimN)
    inputInfo = inputInfo.to(device)

    execution_time = kernel(inputInfo, epoches)
    
    print(str(dimN) + '-' + data + ' dtc-' + str(execution_time))
    
    return execution_time

