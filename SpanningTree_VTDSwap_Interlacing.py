# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:02:09 2018

@author: MGGG
"""
#cd "Documents/GitHub/graph-partitions/"

import MHonSpanningTrees
import MCMC_partitions
import naive_graph_partitions
import numpy as np
import random
import networkx as nx

m = 5
mixture_parameter = 0
tree_steps = int(m**2 * np.log(m)) + 1
district_paths = []
print("when m is", m)
G = nx.grid_2d_graph(m, m)
num_districts = 2

A = naive_graph_partitions.k_connected_graph_partitions(G,2)
list_of_partitions = list(A)

for mixture_parameter in [0,.05,.1]:
    print("mixture_parameter:",mixture_parameter)
    for sample_size in [1000,10000]:
        M = MCMC_partitions.maps(G, num_districts, m)
        #number of steps = (1 - mixture_parameter)*sample_size + mixture_parameter*samplesize*tree_steps
        # = sample_size*( 1 - mixutre_parameter + mixture_parameter*tree_steps)
        samples = sample_size / (1 - mixture_parameter + mixture_parameter*tree_steps)
        #This corrects for the extra steps we're taking...
        M.district_list = M.district_maker() 
        T = MHonSpanningTrees.random_spanning_tree(G)
        e = MHonSpanningTrees.best_edge_for_equipartition(G,T)[0]
        partition = MHonSpanningTrees.R(G,T,e)
        M.district_list = partition
        M.districts = M.district_list
        M.set_node_district_flags()
        M.initialize_boundary()
        M.set_boundary()
        found_districts = []
        
        rejections = 0
        for i in range(sample_size):
            current = M.district_list
            coin = random.uniform(0,1)
            if coin < mixture_parameter:
                partition = M.district_list 
                T_e_pair = MHonSpanningTrees.random_lift(G, partition[0], partition[1])
                T = T_e_pair[0]
                e = T_e_pair[1]
                for i in range(tree_steps):
                    MH_returns = MHonSpanningTrees.MH_step(G,T,e,False, True)
    #                if (T == MH_returns[0]) and (e == MH_returns[1]):
    #                    rejections += 1
                    T = MH_returns[0]
                    e = MH_returns[1]
                    partition = MHonSpanningTrees.R(G,T,e)
                    found_districts.append(partition)
                    M.district_list = partition
                    M.districts = M.district_list
                    M.set_node_district_flags()
                    M.set_boundary()
                M.set_boundary()
            else:
                M.metropolis_hastings()
            found_districts.append(M.district_list)
            if (current[0].nodes() == M.district_list[0].nodes()) or (current[0].nodes() == M.district_list[1].nodes()):
                rejections += 1
                
        print("now making histogram")
        histogram = MCMC_partitions.make_histogram(list_of_partitions, found_districts)
        total_variation = 0
        for k in histogram.keys():
            total_variation += np.abs( histogram[k] - 1 / len(list_of_partitions))
        print(total_variation, "for # trials =", sample_size)
        district_paths.append(found_districts)