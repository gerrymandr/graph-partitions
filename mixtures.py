# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:07:44 2018

@author: MGGG
"""
#Goal is to check whether the uniform distribution on partitions is a mixture of the D_T
#where D_T is the distirbution on all partitions obtained by removing uniformly an edge from the tree T
import naive_graph_partitions as ngp
import networkx as nx
import MHonSpanningTrees as ST_tools
G = nx.complete_graph(4)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#Code below taken from  here: https://stackoverflow.com/questions/40155070/networkx-all-spanning-trees-and-their-associated-total-weight
def _expand(G, explored_nodes, explored_edges):
    """
    Expand existing solution by a process akin to BFS.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    explored_nodes: set of ints
        nodes visited

    explored_edges: set of 2-tuples
        edges visited

    Returns:
    --------
    solutions: list, where each entry in turns contains two sets corresponding to explored_nodes and explored_edges
        all possible expansions of explored_nodes and explored_edges

    """
    frontier_nodes = list()
    frontier_edges = list()
    for v in explored_nodes:
        for u in nx.neighbors(G,v):
            if not (u in explored_nodes):
                frontier_nodes.append(u)
                frontier_edges.append([(u,v), (v,u)])

    return zip([explored_nodes | frozenset([v]) for v in frontier_nodes], [explored_edges | frozenset(e) for e in frontier_edges])

def find_all_spanning_trees(G, root=0):
    """
    Find all spanning trees of a Graph.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    Returns:
    ST: list of networkx.Graph() instances
        list of all spanning trees

    """

    # initialise solution
    explored_nodes = frozenset([root])
    explored_edges = frozenset([])
    solutions = [(explored_nodes, explored_edges)]
    # we need to expand solutions number_of_nodes-1 times
    for ii in range(G.number_of_nodes()-1):
        # get all new solutions
        solutions = [_expand(G, nodes, edges) for (nodes, edges) in solutions]
        # flatten nested structure and get unique expansions
        solutions = set([item for sublist in solutions for item in sublist])

    return [nx.from_edgelist(edges) for (nodes, edges) in solutions]
##### end https://stackoverflow.com/questions/40155070/networkx-all-spanning-trees-and-their-associated-total-weight stuff
    
ST = find_all_spanning_trees(G)
partitions = ngp.k_connected_graph_partitions(G,2)
found_partitions = []
for T in ST:
    for e in T.edges():
        partitions = ST_tools.R(G,T,e)
        found_partitions.append(partitions)

histogram = ST_tools.make_histogram(partitions, found_partitions)
