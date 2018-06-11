'''This script is meant to perform random walk partitioning from a boundary

Authors: Bryce McLaughlin and Amara Jaeger

The general idea is that we begin at the boundary of a state (represented by a super node)
and perform a random walk until we reintersect our path, which may be on the boundary.
At the point of reintersection we create a cycle, which will be used to partition the graph
into multiple parts.

Implementation (for k=2):
-Augment Shapefile to include an outerface/district which touches all boundary districts
-Create [Adjacency Matrix for] dual graph (later: weighted probability of edges to adjust probability distribution)
-Create a random walk of districts and assign them all to partition group 2
-Randomly selected a node which is not one of these and use a BFS to create partition 1 around this district
-Assign all other districts to partition 2 (this guarantees that we will have two connected partitions)
'''

import networkx as nx #for graph datatypes and such
import geopandas as gpd #for shapfile reading
import matplotlib.pyplot as plt #general visualizations
import random

###Augment Shapefile Code



###Create Adjacency Matrix

#Temporary graph for testing
G = nx.grid_graph(dim = [10,10])
G.add_node("boundary")
for i in [0:9]:
    for j in [0:9]:
        if i == 0 or j == 0:
            G.add_edge('boundary', (i,j))
        if i == 9 or j == 9:
            G.add_edge('boundary', (i,j))

#Preset all districts to one color
for n in G.nodes:
    G.nodes[n]['color'] = 'red'
            


###Perform Random Walk

visited = []
current = 'boundary'
while current not in visited:
    visited.append(current)
    G.nodes[current]['color'] = 'blue'
    neighbors = list(G.neighbors(current))
    index = np.random.randrange(0,len(neighbors))
    current = neighbors[index]
    nx.draw(G)
    
    


###Create Partition 1



###Allocate Partition 2 and output LoL for partitioning.