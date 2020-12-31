'''
Description: 
Date: 2020-12-16 11:06:22
LastEditors: PengyuWang
LastEditTime: 2020-12-16 19:39:34
FilePath: /sampling/scripts/trans2.py
'''
import numpy as np
import pandas as pd
import scipy.sparse as ss

def read_data_file_as_coo_matrix(filename='edges.txt'):
    "Read data file and return sparse matrix in coordinate format."
    data = pd.read_csv(filename, sep=' ', header=None, dtype=np.uint32)
    rows = data[0]  # Not a copy, just a reference.
    cols = data[1]
    ones = np.ones(len(rows), np.uint32)
    matrix = ss.coo_matrix((ones, (rows, cols)))
    return matrix

def save_csr_matrix(filename, matrix):
    """Save compressed sparse row (csr) matrix to file.

    Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    attributes = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
    }
    np.savez(filename, **attributes)
    
def tx(filename):
    "Test data file parsing and matrix serialization."
    coo_matrix = read_data_file_as_coo_matrix("/home/pywang/data/" + filename+ ".w.edge")
    csr_matrix = coo_matrix.tocsr()
    save_csr_matrix("/home/pywang/data/" + filename+ ".w.npz", csr_matrix)
    
if __name__ == '__main__':
    print("uk-2005")
    tx("uk-2005")
    # print("twitter-2010")
    # tx("twitter-2010")
    
# read_data_file_as_coo_matrix()
# read_weighted_edgelist
# G= networkit.graphio.readGraph("/home/pywang/data/lj.w.edge", networkit.Format.EdgeList, separator=" ", continuous=False)
import scipy as sp
import networkx as nx



# def save_csr_matrix(filename):
#     G=nx.read_weighted_edgelist("/home/pywang/data/" + filename+ ".w.edge")
#     S=nx.to_scipy_sparse_matrix(G)
#     sp.sparse.save_npz("/home/pywang/data/" + filename+ ".w.npz", S)

# save_csr_matrix("orkut")


# print("sk-2005")
# save_csr_matrix("sk-2005")
# print("friendster")
# save_csr_matrix("friendster")