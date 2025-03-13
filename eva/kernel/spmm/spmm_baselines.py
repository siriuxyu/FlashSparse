import torch
from scipy.sparse import *
import sys
# from advisor import test_advisor
from dtc import test_dtc
from tcgnn import test_tcgnn
from gespmm import test_gespmm
import subprocess

import csv
import pandas as pd
import time
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            
# '''
# GNNAdvisor
# '''
# def advisor_test(data, dimN, epoches,data_path) : 
#     spmm = test_advisor.test(data, epoches, dimN, data_path)
#     return spmm

'''
RoDe
'''
def rode_test(data, dimN, data_path, baseline_dir, result_path):
    shell_command = baseline_dir + f"/RoDe/build/eval/eval_spmm_f32_n{dimN} " + baseline_dir + "/dataset/" + data + '/' + data + ".mtx >> " + result_path
    subprocess.run(shell_command, shell=True)
                
            
'''
TCGNN
'''
def tcgnn_test(data, dimN, epoches,data_path) : 
    spmm = test_tcgnn.test(data, epoches, dimN, data_path)
    return spmm

           
''' 
GE-SpMM
'''
def gespmm_test(data, dimN, epoches,data_path) : 
    spmm = test_gespmm.test(data, epoches, dimN, data_path)
    return spmm


'''
Advisor
'''
def safe_advisor_test(data, dimN, epoches, data_path):
    try:
        result = subprocess.check_output(
            ['python3', 'advisor/test_advisor.py', data, str(dimN), str(epoches), data_path],
            stderr=subprocess.STDOUT
        )
        return float(result.decode().strip())  # 假设返回值是一个数字
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed: {e.output.decode()}")
        return 100000  # 默认值

'''
DTC
'''
def dtc_test(data, dimN, epoches, data_path):
    spmm = test_dtc.test(data, epoches, dimN, data_path)
    return spmm    
    
if __name__ == "__main__":

    gpu_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)


    dimN = int(sys.argv[1])
    print('dimN: ' + str(dimN))
    epoches = 10
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    baseline_dir = osp.join(project_dir, 'Baseline')
    
    #result path
    file_name = project_dir + '/result/Baseline/spmm/base_spmm_f32_n' + str(dimN) + '.csv'
    head = ['dataSet','num_nodes','num_edges',
            'advisor','advisor_gflops',
            'tcgnn','tcgnn_gflops',
            'gespmm','gespmm_gflops',
            'dtc','dtc_gflops',
            'src','dst','nnz',
            'cuSPARSE','cuSPARSE_gflops',
            'rode','rode_gflops']
    
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)
    
    start_time = time.time()
    
    # Traverse each dataset
    df = pd.read_csv(baseline_dir + '/dataset/data_filter.csv')
    # df = pd.read_csv(project_dir + '/result/ref/baseline_h100_spmm_256.csv')
    
    for index, row in df.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])

        #Dataset path
        data = row.iloc[0]
        data_path =  baseline_dir + '/dataset/' + data + '/' + data + '.mtx'
        
        from scipy.io import mmread
        mtx = mmread(data_path)
        nnz = mtx.nnz
        
        # advisor
        # if data not in []:
        #     spmm_advisor = advisor_test(data, dimN, epoches, data_path)
        #     res_temp.append(spmm_advisor)
        # else:
        #     spmm_advisor = advisor_test(data, dimN, epoches, data_path)
        #     res_temp.append(spmm_advisor)
        
        spmm_advisor = safe_advisor_test(data, dimN, epoches, data_path)
        print(str(dimN) + '-' + data + ' advisor-' + str(spmm_advisor))
        gflops = 2 * nnz * dimN / (spmm_advisor * 1e6)
        res_temp.append(spmm_advisor)
        res_temp.append(gflops)
        
        # tcgnn
        if row.iloc[2] < 1000000:
            spmm_tcgnn = tcgnn_test(data, dimN, epoches, data_path)
            gflops = 2 * nnz * dimN / (spmm_tcgnn * 1e6)
            res_temp.append(spmm_tcgnn)
            res_temp.append(gflops)
        else:
            res_temp.append(10000000)
            res_temp.append(10000000)
            
        # gespmm
        spmm_gespmm = gespmm_test(data, dimN, epoches, data_path)
        gflops = 2 * nnz * dimN / (spmm_gespmm * 1e6)
        res_temp.append(spmm_gespmm)
        res_temp.append(gflops)
        
        # dtc
        spmm_dtc = dtc_test(data, dimN, epoches, data_path)
        gflops = 2 * nnz * dimN / (spmm_dtc * 1e6)
        res_temp.append(spmm_dtc)
        res_temp.append(gflops)
        
            

        write_row_before_rode = ','.join(map(str, res_temp))

        with open(file_name, 'a', newline='') as csvfile:
            csvfile.write(write_row_before_rode)
            # csv_writer = csv.writer(csvfile)
            # csv_writer.writerow(res_temp)
        
        # rode
        rode_test(data, dimN, data_path, baseline_dir, file_name)
        
        print(data + ' is success')
        print()
    print('All is success')
    
    end_time = time.time()
    execution_time = end_time - start_time

    # Record execution time.
    with open("execution_time_base.txt", "a") as file:
        file.write("Baseline-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")