# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:20:40 2018

@author: MGGG
"""
#Here are some partition generation tools based on deleting edges until you get two partitions
# or adding random edges until you get a spanning tree, then removing a random edge... 
#implemented by sorting the edges and then adding them greedily. (This is the same as finding a minimal spanning tree by greedy for random edge weights.)
#This implementation is equivalent -- instead of drawing the edges randomly, we can just imagine that we have a list representing all samples, then we leave only the first 
#occurence of each edge, and this give us our random sort. But its clear that the sample of edges produces the same tree as when we leave only the 
#first time each edge appears, because if the edge is not incldued then, that is because there is a cycle that it forms, so it will never be included 
#again afterwards, and if it is included, we we never need to consider samples that draw that edge again...
#none of them work, at least without modification
m = 3
import networkx as nx
partitions = []
for i in range(10000):
    G = nx.grid_graph([m,m])
    while len(list(nx.connected_components(G))) == 1:
        e = random.choice(list(G.edges()))
        G.remove_edges_from([e])
    components = list(nx.connected_components(G))
    G_A = nx.subgraph(G, components[0])
    G_B = nx.subgraph(G, components[1])
    partitions.append([G_A, G_B])
G = nx.grid_graph([m,m])
A = k_connected_graph_partitions(G,2)
list_of_partitions = list(A)

print("now making histogram")
histogram = make_histogram(list_of_partitions, partitions)
total_variation = 0
for k in histogram.keys():
    total_variation += np.abs( histogram[k] - 1 / len(list_of_partitions))
print(total_variation)

#doesn't work:
#0.760441509433962 for # trials = 100000

m = 3
import networkx as nx
trees = []
G = nx.grid_graph([m,m])
n = len(G.nodes())
sorted_edges = list(G.edges())
for i in range(5000):
    edge_list = []
    T = nx.Graph()
    random.shuffle(sorted_edges)
    i = 0
    while len(T.edges()) < n - 1:
        e = sorted_edges[i]
        T.add_edges_from([e])
        if nx.is_forest(T) == False:
            T.remove_edges_from([e])
        i += 1
    trees.append(T)
    
partitions = []
for tree in trees:
    for e in tree.edges():
        partitions.append(R(G,tree,e))

print("now making histogram")
histogram = make_histogram(list_of_partitions, partitions)
total_variation = 0
for k in histogram.keys():
    total_variation += np.abs( histogram[k] - 1 / len(list_of_partitions))
print(total_variation)

#0.655114780047320

      
partitions = []
for tree in trees:
    e = random.choice(list(tree.edges()))
    partitions.append(R(G,tree,e))

print("now making histogram")
histogram = make_histogram(list_of_partitions, partitions)
total_variation = 0
for k in histogram.keys():
    total_variation += np.abs( histogram[k] - 1 / len(list_of_partitions))
print(total_variation, "for # trials =")
#0.65004150943396
