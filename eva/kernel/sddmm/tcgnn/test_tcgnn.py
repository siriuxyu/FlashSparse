
import sys
sys.path.append('./eva100/kernel/gcn')
from tcgnn.mdataset import *
import time
import TCGNN_kernel


def kernel(inputInfo, epoches):
    X_prime, sddmm_ms_avg = TCGNN_kernel.forward_ef(inputInfo.x, inputInfo.row_pointers, inputInfo.column_index, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow, epoches)
    return round(sddmm_ms_avg.item(),4)

def test(data, epoches, dimN,data_path):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data,data_path)
    inputInfo.init_embedding(dimN)
    inputInfo = inputInfo.to(device)

    execution_time = kernel(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' tcgnn-' + str(execution_time))
    return execution_time

