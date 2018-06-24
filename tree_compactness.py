# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:49:44 2018

@author: MGGG
"""
import numpy as np
import random
import scipy.linalg
from scipy.sparse import csc_matrix
import scipy
from scipy import array, linalg, dot

def log_weighted_number_trees(G):
    m = nx.laplacian_matrix(G, weight = "weight")[1:,1:]
    m = csc_matrix(m)
    splumatrix = scipy.sparse.linalg.splu(m)
    diag_L = np.diag(splumatrix.L.A)
    diag_U = np.diag(splumatrix.U.A)
    S_log_L = [np.log(s) for s in diag_L]
    S_log_U = [np.log(s) for s in diag_U]
    LU_prod = np.sum(S_log_U) + np.sum(S_log_L)
    return LU_prod

import networkx as nx
m = 10
d = 10
G = nx.grid_graph([m,m])
H = nx.grid_graph([m,m])
for x in G.edges():
    a = x[0]
    b = x[1]
    G.edges[x]["weight"] = (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]))
for x in H.edges():
    a = x[0]
    b = x[1]
    H.edges[x]["weight"] = ( d*np.abs(a[0] - b[0]) + (1/d)* np.abs(a[1] - b[1]))

W_G = log_weighted_number_trees(G)
W_H = log_weighted_number_trees(H)
print("W_G", W_G)
print("W_H", W_H)