# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:49:44 2018

@author: MGGG
"""
import numpy as np
import networkx as nx
import scipy.linalg
from scipy.sparse import csc_matrix
import scipy
import random
import numpy as np
import copy
from anytree import Node, NodeMixin

class meta_tree(NodeMixin):
    def __init__(self, G):
        self.graph = G
    
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

#
#m = 3
#n = 3
#d = 2
#G = nx.grid_graph([m,n])
#H = nx.grid_graph([m,n])
#for x in G.edges():
#    a = x[0]
#    b = x[1]
#    G.edges[x]["weight"] = (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]))
#for x in H.edges():
#    a = x[0]
#    b = x[1]
#    H.edges[x]["weight"] = (d*np.abs(a[0] - b[0]) + (1/d)* np.abs(a[1] - b[1]))
#
#
#W_G = log_weighted_number_trees(G)
#W_H = log_weighted_number_trees(H)
#print("WG",W_G)
#print("WH", W_H)
#print("W_G - W_H", W_G - W_H)
#
##W_G 98.44804291763761
##W_H 209.95528522499345
#
#    
#ambient = nx.grid_graph( [100,100])
#
#G_nodes = [(50,50)]
#G = nx.induced_subgraph(ambient, G_nodes)
#neighbors = []
#for x in ambient.nodes():
#    if set(ambient.neighbors(x)).intersection(set(G.nodes())) != set():
#        neighbors.append(x)
#

'''
On how to enumerate the partitions:
One way to solve the problem is to define a tree structure on these subgraphs as follows: choose an arbitrary assignment of distinct weights to the edges of the input graph. Define the parent of a connected induced subgraph to be the graph formed by finding its minimum spanning tree, removing the leaf edge of maximum weight, and forming the induced subgraph of the remaining vertices. For a subgraph with only one vertex, define its parent to be the empty graph (which forms the root of the tree structure)

Reverse search (essentially, DFS of this tree) will then find each connected induced subgraph in time polynomial per subgraph. You can find the children of any node in the tree by trying all ways of adding one vertex and checking which of them would form the heaviest leaf of the MST of the augmented subgraph; finding all the children of a single node takes polynomial time, which is all that's required to make this work.
https://cstheory.stackexchange.com/questions/16305/enumerating-all-connected-induced-subgraphs
'''

G = nx.grid_graph([100,100])
G_edges = G.edges()
for e in G_edges:
    G.edges[e]["weight"] = random.uniform(0,1)
    
def parent(H):
    #Give it a subgraph, it returns the set of nodes of it's parent
    if len(H.nodes()) == 1:
        return set([])
    if len(H.nodes()) == 2:
        return set((random.choice(H.nodes())))
    
    M = nx.minimum_spanning_tree(H)
    M_edges = M.edges()
    leaf_edges = [f for f in M_edges if ( (M.degree(f[0]) == 1) or (M.degree(f[1]) == 1))]
    max_weight = np.max ([ M.edges[f]["weight"] for f in leaf_edges])
    for e in leaf_edges:
        #removing the leaf edge of maximum weight,
        if M.edges[e]["weight"] == max_weight:
            M.remove_edges_from([e])
            components = list(nx.connected_components(M))
            if len(components[1]) == 1:

               # print(components[0])
                return set(components[0])
            else:

                #print(components[1])
                return set(components[1])
        
def neighbors(H_nodes, G):
    #Finds all neighbor nodes of H in G
    #H is a set ofnodes
    if len(H_nodes) == 0:
        return list(G.nodes())
    neighbors_list = []
    for v in G.nodes():
        vneighbors = set(G.neighbors(v))
        if v not in H_nodes:
            if vneighbors & H_nodes != set():
                neighbors_list.append(v)
    return neighbors_list

def children(H_nodes,G):
    #returns the list of all children of subgraph H in graph G
    #In the tree described in the comments above
    N = neighbors(H_nodes,G)

    #input H as a list of nodes
    admissable_additions = []
    for n in N:
        H_nodes.add(n)
        candidate_child = nx.induced_subgraph(G, H_nodes)
        H_nodes.remove(n)
        P = parent(candidate_child)
        if P == H_nodes:
                admissable_additions.append(n)
    admissable_children = []
    for n in admissable_additions:
        H_nodes.add(n)
        admissable_children.append(copy.deepcopy(H_nodes))
        H_nodes.remove(n)
    return admissable_children

def build_tree(G):
    G = nx.Graph()
    root = meta_tree(G)
    
    
def prune(T):
    #Remove isomorphism duplicates from among the subgraphs
    
def extend(T):
    
    