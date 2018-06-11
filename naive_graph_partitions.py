# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:28:05 2018

@author: pgirardet
"""

import networkx as nx 
import itertools 

# Enumerates the set partitions of an input list. If the length of the input 
# list is n, this should generate B_n (the nth Bell number) partitions. 
# Input: a_list - a Python list
def list_partitions(a_list): 
    a_list = list(a_list)
    if len(a_list) < 1: 
        yield []
    elif len(a_list) == 1:
        yield [a_list]
    else:
        for k in range(len(a_list)):      
            for subset in itertools.combinations(a_list[1:], k):
                first_subset = list(subset)
                first_subset.append(a_list[0])
                remainder_list = [elt for elt in a_list if elt not in first_subset]
                if len(remainder_list) == 0:
                    yield [first_subset]
                else:
                    for recursive_partition in list_partitions(remainder_list):
                        partition = [first_subset]
                        partition.extend(recursive_partition)
                        yield partition

def graph_partitions(G): 
    vertex_partitions = list_partitions(nx.nodes(G))
    for vertex_partition in vertex_partitions: 
        graph_partition = []
        for subset in vertex_partition:
            subgraph = G.subgraph(subset)
            graph_partition.append(subgraph)
        yield graph_partition
    
def connected_graph_partitions(G):
    for partition in graph_partitions(G):
        valid = True
        for subgraph in partition:
            if not nx.is_connected(subgraph):
                valid = False
                break
        if not valid:
            continue
        else:
            yield partition
            
    
