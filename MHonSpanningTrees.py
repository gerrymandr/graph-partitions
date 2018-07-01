# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 19:24:59 2018

@author: MGGG
"""

import networkx as nx
import numpy as np
import random
import scipy.linalg
from scipy.sparse import csc_matrix
import scipy
from scipy import array, linalg, dot
from naive_graph_partitions import k_connected_graph_partitions

def log_number_trees(G):
    #Kirkoffs is the determinant of the minor..
    #at some point this should be replaced with a Cholesky decomposition based algorithm, which is supposedly faster. 
    m = nx.laplacian_matrix(G)[1:,1:]
    m = csc_matrix(m)
    splumatrix = scipy.sparse.linalg.splu(m)
    diag_L = np.diag(splumatrix.L.A)
    diag_U = np.diag(splumatrix.U.A)
    S_log_L = [np.log(s) for s in diag_L]
    #Seems like the upper diagonal of L is always 1.... so this may be unnecessary.
    S_log_U = [np.log(s) for s in diag_U]
    #LU_prod = np.sum(S_log_U)
    LU_prod = np.sum(S_log_U) + np.sum(S_log_L)
    return LU_prod

def true_number_trees(G):
    m = nx.laplacian_matrix(G)[1:,1:]
    return  np.linalg.det(m.A)

def number_trees_via_spectrum(G):
    n = len(G.nodes())
    S = nx.laplacian_spectrum(G)[1:]
    logdet = np.sum ( [np.log(s) for s in S]) - np.log(n)
    return logdet
 
##This makes a random spanning tree:    

def srw(G,a):
    wet = set([a])
    trip = [a]
    while len(wet) < len(G.nodes()):
        b = random.choice(list(G.neighbors(a)))
        wet.add(b)
        trip.append(b)
        a = b
    return trip

def forward_tree(G,a):
    walk = srw(G,a)
    edges = []
    for x in G.nodes():
        if (x != walk[0]):
            t = walk.index(x)
            edges.append( [walk[t], walk[t-1]])
    return edges

def random_spanning_tree(G):
    T_edges = forward_tree(G, random.choice(list(G.nodes())))
    T = nx.Graph()
    T.add_nodes_from(list(G.nodes()))
    T.add_edges_from(T_edges)
    return T

###/spanning tree

##May be faster to do this all with matrices / F_2... since cycles are linearly depenent sets, etc

def propose_step(G,T, cut_only = False):
    T_edges = list(T.edges())
    T_edges_t = [ tuple((e[1], e[0])) for e in T_edges]
    e = list(T.edges())[0]
    if cut_only == False:
        while (e in list(T.edges())) or (tuple([e[1], e[0]]) in list(T.edges())):
            e = random.choice(list(G.edges()))
            ##This is stupid!
    if cut_only == True:
        #If we want to only add edges that would change a best partition...
        cuts = cut_edges(G,T, best_edge_for_equipartition(G,T)[0])
        A = [e for e in cuts if e not in T_edges and e not in T_edges_t]
        e = random.choice(cuts)       
    T.add_edges_from([e])
    C = nx.find_cycle(T, orientation = 'ignore')
    w = random.choice(C)
    U = nx.Graph()
    U.add_edges_from(list(T.edges()))
    U.remove_edges_from([w])
    T.remove_edges_from([e])
    return U

def cut_edges(G,T,e):
    T.remove_edges_from([e])
    components = list(nx.connected_components(T))
    T.add_edges_from([e])
    A = components[0]
    B = components[1]
    G_A = nx.induced_subgraph(G, A)
    G_B = nx.induced_subgraph(G, B)
    edges_of_G = list(G.edges())
    for x in list(G_A.edges()):
        if x in edges_of_G:
            edges_of_G.remove(x)
        if (x[1], x[0]) in edges_of_G:
            edges_of_G.remove( (x[1], x[0]))
    for x in list(G_B.edges()):
        if x in edges_of_G:
            edges_of_G.remove(x)
        if (x[1], x[0]) in edges_of_G:
            edges_of_G.remove( (x[1], x[0]) )
    return edges_of_G

def R(G,T,e):
    T.remove_edges_from([e])
    components = list(nx.connected_components(T))
    T.add_edges_from([e])
    A = components[0]
    B = components[1]
    G_A = nx.induced_subgraph(G, A)
    G_B = nx.induced_subgraph(G, B)
    return [G_A, G_B]

def random_lift(G,G_A, G_B):
    T_A = random_spanning_tree(G_A)
    T_B = random_spanning_tree(G_B)
    cross_edges = []
    for a in G_A.nodes():
        for b in G_B.nodes():
            if G.has_edge(a,b):
                cross_edges.append([a,b])
    #This is stupid!
    e = random.choice(cross_edges)
    T = nx.Graph()
    T.add_nodes_from(list(G.nodes()))
    T.add_edges_from(list(T_A.edges()))
    T.add_edges_from(list(T_B.edges()))
    T.add_edges_from([e])
    e = random.choice(list(T.edges()))
    return [T,e]

def bounce(G,T,e):
    partition = R(G,T,e)
    G_A = partition[0]
    G_B = partition[1]
    lift = random_lift(G, G_A, G_B)
    return lift

def score_tree_edge_pair(G,T,e):
    partition = R(G,T,e)
    G_A = partition[0]
    G_B = partition[1]
    cut_size = nx.cut_size(G,G_A,G_B)
    return -1 * (log_number_trees(G_A) + log_number_trees(G_B) +  np.log(cut_size))
    
def score_tree(G, T):
    #returns the sum of the negative log likelihoods of all the partitions that can be
    #obtained from T be removing one edge. not sure if this is a meaningful thing to do.
    scores = []
    for e in T.edges():
        scores.append(score_tree_edge_pair(G,T,e))
    return  np.sum(scores)

def best_edge_for_equipartition(G,T):
    list_of_edges = list(T.edges())
    best = 0
    candidate = 0
    for e in list_of_edges:
        score = equi_score_tree_edge_pair(G,T,e)
        if score > best:
            #print(best)
            best = score
            candidate = e
    return [candidate, best]

def equi_score_tree_edge_pair(G,T,e):
    T.remove_edges_from([e])
    components = list(nx.connected_components(T))
    T.add_edges_from([e])
    A = len(components[0])
    B = len(components[1])
    x =  np.min([A / (A + B), B / (A + B)])
    return x

def MH_step(G, T,e, equi = False, cut_edges = False, use_MH = True):
        U = propose_step(G,T, cut_edges)
        if use_MH == True:
            current_score = score_tree_edge_pair(G,T,e)
        if equi == False:
            e2 = random.choice(list(U.edges()))
        if equi == True:
            e2 = best_edge_for_equipartition(G,U)[0]
        if use_MH == True:
            new_score = score_tree_edge_pair(G, U, e2)
        #A = np.min(1, current_score - new_score)
            if new_score > current_score:
                return [U,e2]
            else:
               p = np.exp(new_score - current_score)
               a = np.random.uniform(0,1)
               if a < p:
                   return [U,e2]
               else:
                   return [T,e]
        if use_MH == False:
            return [U,e2]
    
def print_summary_statistics(G,partitions):
    sizes = []
    cuts = []
    interior = []
    TATBcut = []
    for x in partitions:
        sizes += [len(y.nodes()) for y in x]
        cut_size = nx.cut_size(G,x[0],x[1])
        cuts.append(cut_size)
        #boundary = nx.node_boundary(G,x[0],x[1])
        interior_count = 0
        for a in x[0].nodes():
            y = 1
            for z in G.neighbors(a):
                if z in x[1]:
                    y = 0
            interior_count += y
        for a in x[1].nodes():
            y = 1
            for z in G.neighbors(a):
                if z in x[0]:
                    y = 0
            interior_count += y
        
        interior.append(interior_count)
        logTA = log_number_trees(x[0])
        logTB = log_number_trees(x[1])
        TATBcut.append([logTA + logTB + np.log(cut_size)])
    print([np.mean(TATBcut), np.var(TATBcut)], [np.mean(cuts), np.var(cuts)], 
           [np.mean(interior), np.var(interior)])
    
def count(x, sample):

    x_lens = np.sort([len(k) for k in x])
    count = 0
    for i in sample:
        sample_nodes = set([frozenset(g.nodes()) for g in i])
        sample_lens = np.sort([len(k) for k in sample_nodes])
        if (x_lens == sample_lens).all():
            if x == sample_nodes:
                count += 1
    return count

def make_histogram(A, sample):
    A_node_lists = [ set([frozenset(g.nodes()) for g in x]) for x in A]
    dictionary = {}
    for x in A_node_lists:
        dictionary[str(x)] = count(x,sample) / len(sample)
    return dictionary



rejections_list = []
for m in range(3,4):
    m = 3
    #print("when m is", m)
    G = nx.grid_graph([3,2])
    list_of_partitions = list(k_connected_graph_partitions(G,2))
    desired_samples = [10**k for k in range(4,5)]
    desired_samples = [1000]
    for sample_size in desired_samples:
        samples = sample_size
        T = random_spanning_tree(G)
        e = best_edge_for_equipartition(G,T)[0]
        trees = []
        scores = []
        partitions = []
        rejections = 0
        for i in range(samples):
            previous_T = T
            previous_e = e
            MH_returns = MH_step(G,T,e,False, True)
            if (T == MH_returns[0]) and (e == MH_returns[1]):
                rejections += 1
            T = MH_returns[0]
            e = MH_returns[1]
            partitions.append(R(G,T,e))
            trees.append(T)
            scores.append(score_tree_edge_pair(G,T,e))        
        print_summary_statistics(G,partitions)
        #print(rejections / samples , "at: ", m)
        rejections_list.append(rejections)
        
            
    #    inv_rejections = [1 / (1 - x/samples) for x in rejections_list]
        #print("now making histogram")
        histogram = make_histogram(list_of_partitions, partitions)
        total_variation = 0
        for k in histogram.keys():
            total_variation += np.abs( histogram[k] - 1 / len(list_of_partitions))
        print("total variation", total_variation, "for # trials =", sample_size)
#              
#    
#for sample_size in [100000]:     
#    partitions_subsampled = []
#    for i in range(sample_size):
#        partitions_subsampled.append( random.choice(partitions))
#    print("now making histogram")
#    histogram = make_histogram(list_of_partitions, partitions_subsampled)
#    print("now computing TV")
#    total_variation = 0
#    length = len(list_of_partitions)
#    for k in histogram.keys():
#        total_variation += np.abs( histogram[k] - (1 / length))
#    print(total_variation, "for # trials =", sample_size)
#      
#    print_summary_statistics(G,A)
#    
#raw_scores = []
#partitions = []
#for i in range(10000):
#    T = random_spanning_tree(G)
#    e = random.choice(list(T.edges()))
#    partitions.append(R(G,T,e))
#    raw_scores.append(score_tree_edge_pair(G,T,e))
#
#print_summary_statistics(G,partitions)
#
#

###Notes:
#
#1) It's easy to compute the minimum distance between spanning trees: For any trees T and T', to get from T 
#        to T', take any edge in T' \ T, and add it to T. Then consider the cycle in T union this edge; there must
#        be some edge in that cycle that is not T'. Remove that edge. 
#2) We can potentially parallelize this algorithm: 
#        For any starting state, consider random lifts to the tree edge space. After walking a while,
#        use the torsor structure on spanning trees: pick three spanning trees we've arrived at, and add
#        the difference between two of them to the third
#3) Maybe there is a way to use the bounce idea as a new proposal step.
#4) the rapid mixing on spanning trees doesn't care about the edge we pick. So we can probably say things about the optimal split e.
    #this is also nice because the TATBcut(A,B) statistic will stay optimized for compactness on roughly equal sized regions... I think.
    #So when we use a score function that weights for compactness, we can hope that will cancel out somewhat with the TATBcut(A,B) term, and make 
    #fewer rejections happen.


#################Spanning tree results:
              
#Result sfor m = 3 for total variation...
        
#0.9875471698113206 for # trials = 100
#0.4563396226415093 for # trials = 1000
#0.12113207547169812 for # trials = 10000
#0.04591622641509434 for # trials = 100000

###############################
# Total Variation Distance from Uniform (4x4 grid graph)
#Number of Steps
#0.84352
#(True: 0.19970)
#10,000
#
#0.30092
#(True: 0.061953)
#100,000
# 
#1,000,000
#(when subsampling 10000 from the tree MH...) : 0.7392822966507208 (compare .19)
#(when subsampling 100000 from the treeMH...): 0.3874779425837311 (compare .06)
    ############################
    
###################MH VTD results:
    

#when m is 4
#now making histogram
#1.7480063795853424 for # trials = 100
#now making histogram
#0.8291355661881985 for # trials = 1000
#now making histogram
#0.3643438596491226 for # trials = 10000
##actually, this isn't suprising, just because the ratio between degree of different vertices
    #is not nearly as much as the ratio between TATBcut(A,B)

#Summary statistics: log( TATBcut(A,B)), cut size, number of interior points (mean, variance)
#5x5 grid
#True values:
#[9.014163302779068, 9.019158494862038] [12.11867020292358, 6.134810246689387] [8.929130944303953, 9.059370348177712]

#MH on spanning tree (scored so that stationary on partitions is uniform) (10,000 steps -  fast)
#[8.564053489416544, 10.6287110507783] [12.483, 7.229911] [8.7962, 9.967665559999999]
#100,000 steps:
#[9.291270317050051, 9.080127914784288] [11.9001, 6.162719989999999] [9.19792, 8.5542076736]
#MH on vertex flipping MCMC (scored so that stationary on partitions is uniform) (10,000 steps -  fast):
#[14.077806914778312, 3.1979242018157756] [7.8702, 2.4757519599999998] [14.2048, 1.2706569599999997] (was this working right???)
#Uniform spanning tree distribution:
#[17.228877329514262, 4.031003964483184] [4.7813, 4.1446703099999995] [17.9483, 9.079027109999998]

                                                                
#Summary statistics: log( TATBcut(A,B)), cut size, number of interior points (mean, variance)
#4x4 grid
#True values:
#[5.855501658111644, 4.170027711908105] [7.08133971291866, 3.1114061796509542] [6.220095693779904, 4.780903978083529]
#MH on spanning tree (scored so that stationary on partitions is uniform) (10,000 steps -  fast)
#[5.368163391113894, 4.3792031659171515] [7.4887, 3.2148723100000005] [5.731, 5.060839]
#MH on vertex flipping MCMC (scored so that stationary on partitions is uniform) (10,000 steps -  fast):
#[6.58424847312279, 0.3179187539754784] [6.4841, 0.24974719000000004] [6.4937, 1.25176031]
#Uniform spanning tree distribution:
#[9.299556053055793, 1.965646645451391] [3.9294, 2.01701564] [10.0926, 4.54262524]