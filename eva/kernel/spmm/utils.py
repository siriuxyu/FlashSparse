import torch
import numpy as np
import time

from scipy.io import mmread
from scipy.sparse import coo_matrix
    
torch.manual_seed(0)
class Dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, verbose=False):
        super(Dataset, self).__init__()
        self.verbose_flag = verbose
        self.init_dataset_mtx(path)
        # self.init_embedding(128)
        
    def init_dataset_mtx(self, path):
        if not path.endswith('.mtx'):
            raise ValueError("graph file must be a .mtx file")
        # start = time.perf_counter()
        
        mtx = mmread(path)
        if not isinstance(mtx, coo_matrix):
            mtx = coo_matrix(mtx)
            
        self.num_nodes_src = mtx.shape[1]
        self.num_nodes_dst = mtx.shape[0]
        
        self.num_nodes = self.num_nodes_src
        if self.num_nodes_src%16 !=0 :
            self.num_nodes = self.num_nodes_src + 16 - self.num_nodes_src%16
            
        self.num_edges = mtx.nnz
        
        self.src = mtx.col
        self.dst = mtx.row
        self.edge_index = np.stack([self.src, self.dst])
        self.avg_degree = self.num_edges / self.num_nodes
        
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        # build_csr = time.perf_counter() - start
        # if self.verbose_flag:
        #     print("# Build CSR (s): {:.3f}".format(build_csr))
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.values = torch.tensor(adj.data, dtype=torch.float32)
    
        
    def init_embedding(self, dimN):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x1 = torch.randn(self.num_nodes, dimN)
        self.x = self.x1.cuda()
        
    def to(self, device):
        self.column_index = self.column_index.cuda()
        self.row_pointers = self.row_pointers.cuda()
        self.values = self.values.cuda()
        # self.x =  self.x.to(device)
        return self