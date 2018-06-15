'''The point of this file is to provide a framework which we can use to compute
the fewest number of VTD flips we must perform to move between partitions. Our
idea to do so is as follows:

1) Compute the number of VTDs which must be flipped in order to have one district
absorb another. This includes shortest path movement if not overlapping
2) Use LP from http://www.cse.iitd.ernet.in/~naveen/courses/CSL851/lec4.pdf with 
gurobi to find matching of districts
3) Actually perform a set of flips to move between districts
'''

import numpy as np
import networkx as nx

''' Returns true if there is a set overlap and false otherwise
'''

def overlap(set1,set2):
    for vertex in set1:
        if vertex in set2:
            return True
        
''' Returns true if there is a perfect matching with all overlapping districts and false otherwise
'''

def overlapping_matching(partition1,partition2):
    #Check to make sure we are accepting partitions of equal size
    k = len(partition1)
    if k != len(partition2):
        return "Partition sizes are not equal"
    
    #Initialize adjacency matrix
    adjacency_matrix = np.zeros((k,k), int)
    index_p1 = 0
    index_p2 = 0
    for district_p2 in partition2:
        for district_p1 in patition1:
            if overlap(district_p1,district_p2):
                adjacency_matrix[index_p1][index_p2] = 1
            index_p1 += 1
        index_p2 += 1
        
    #Create Bipartite Graph
    G = nx.Graph()
    for i in range(k):
        district = str(i)+'a'
        G.add_node(district)
    for i in range(k):
        district = str(i)+'b'
        G.add_node(district)
    for row in len(adjacency_matrix):
        for col in len(adjacency_matrix[row]):
            if adjacency_matrix[row][col] == 1:
                G.add_edge(str(row)+'a',str(col)+'b')
                    
    #Check max matching (Best Matching is equal to 2k because max matching records from both sides)
    if list(nx.bipartite.maximum_matching(G)) == 2*k:
        return True
    else:
        return False
    
                