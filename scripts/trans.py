'''
Description: 
Date: 2020-12-16 11:06:22
LastEditors: PengyuWang
LastEditTime: 2020-12-16 17:15:33
FilePath: /sampling/scripts/trans.py
'''
# import numpy as np
# import pandas as pd
# import scipy.sparse as ss

# def read_data_file_as_coo_matrix(filename='edges.txt'):
#     "Read data file and return sparse matrix in coordinate format."

#     # if the nodes are integers, use 'dtype = np.uint32'
#     data = pd.read_csv(filename, sep = '\t', encoding = 'utf-8')

#     # where 'rows' is node category one and 'cols' node category 2
#     rows = data['agn']  # Not a copy, just a reference.
#     cols = data['fct']

#     # crucial third array in python, which can be left out in r
#     ones = np.ones(len(rows), np.uint32)
#     matrix = ss.coo_matrix((ones, (rows, cols)))
#     return matrix

# def save_csr_matrix(filename, matrix):
#     """Save compressed sparse row (csr) matrix to file.

#     Based on http://stackoverflow.com/a/8980156/232571

#     """
#     assert filename.endswith('.npz')
#     attributes = {
#         'data': matrix.data,
#         'indices': matrix.indices,
#         'indptr': matrix.indptr,
#         'shape': matrix.shape,
#     }
#     np.savez(filename, **attributes)

# read_data_file_as_coo_matrix()
# read_weighted_edgelist
# G= networkit.graphio.readGraph("/home/pywang/data/lj.w.edge", networkit.Format.EdgeList, separator=" ", continuous=False)
import scipy as sp
import networkx as nx



def save_csr_matrix(filename):
    G=nx.read_weighted_edgelist("/home/pywang/data/" + filename+ ".w.edge")
    S=nx.to_scipy_sparse_matrix(G)
    sp.sparse.save_npz("/home/pywang/data/" + filename+ ".w.npz", S)

# save_csr_matrix("orkut")
print("uk-2005")
save_csr_matrix("uk-2005")
print("twitter-2010")
save_csr_matrix("twitter-2010")
# print("sk-2005")
# save_csr_matrix("sk-2005")
# print("friendster")
# save_csr_matrix("friendster")