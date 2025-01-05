### We reuse the code from TC-GNN ()
#!/usr/bin/env python3
import torch
import numpy as np
import time

from scipy.sparse import *
torch.manual_seed(0)
class DTC_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, verbose=False):
        super(DTC_dataset, self).__init__()
        self.nodes = set()
        self.num_nodes = 0
        self.edge_index = None
        self.verbose_flag = verbose
        self.init_sparse(path)
    
    def init_sparse(self, path):
        if not path.endswith('.npz'):
            raise ValueError("graph file must be a .npz file")
        start = time.perf_counter()
        self.graph = np.load(path)
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%16 !=0 :
            self.num_nodes = self.num_nodes_ori + 16 - self.num_nodes_ori%16

        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        # print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        build_csr = time.perf_counter() - start
        if self.verbose_flag:
            print("# Build CSR (s): {:.3f}".format(build_csr))

        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        