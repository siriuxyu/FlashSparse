
import sys
import GESpMM_kernel
import time
# sys.path.append('eva100/kernel/gcn')
from gespmm.mdataset import *

import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import Dataset


def kernel(inputInfo:MGCN_dataset, epoches):
    # for i in range(epoches):
    X_prime, spmm_ms_avg = GESpMM_kernel.forward(inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, 
                               inputInfo.x, inputInfo.num_nodes,  inputInfo.x.size(1), inputInfo.num_edges, epoches, 10)
    return round(spmm_ms_avg.item(),4)

def test(data, epoches, dimN, data_path):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data_path)
    inputInfo.init_embedding(dimN)
    inputInfo.to(device)

    execution_time = kernel(inputInfo, epoches)

    print(str(dimN) + '-' + data + ' gespmm-' + str(execution_time))

    return execution_time

