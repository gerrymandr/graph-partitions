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
import queue as q

#Temporary graph for testing
G = nx.grid_graph(dim = [10,10])
G.add_node("boundary")
for i in range(0,9):
    for j in range(0,9):
        if i == 0 or j == 0:
            G.add_edge('boundary', (i,j))
        if i == 9 or j == 9:
            G.add_edge('boundary', (i,j))

visited = []
current = 'boundary'
color_map = []
last = None
while current not in visited:
    plt.close
    visited.append(current)
    neighbors = list(G.neighbors(current))
    newnode = last
    while newnode == last:
        index = random.randrange(0,len(neighbors))
        newnode = neighbors[index]
    last = current
    current = newnode 
    if current == 'boundary':
        break
for node in G:
    if node in visited:
        color_map.append('purple')
    else:
        color_map.append('green')

start = 'boundary'
while start in visited:
   start = random.choice(list(G.nodes))
partition_1 = []
bfs_queue = q.Queue()
bfs_queue.put(start)
bfs_color = []

#Populate the first partition using a BFS model
while not bfs_queue.empty():
   node = bfs_queue.get()
   partition_1.append(node)
   for nbr in list(G.neighbors(node)):
       if nbr not in partition_1 and nbr not in visited and nbr not in list(bfs_queue.queue):
           bfs_queue.put(nbr)

###Allocate Partition 2 and output LoL for partitioning.
partition_2 = []
for other in list(G.nodes):
   if other not in partition_1 and other != 'boundary':
       partition_2.append(other)
print('[DISTRICT 1:', partition_1, '\n', 'DISTRICT 2',partition_2)

G.remove_node('boundary')

for node in G:
    if node in partition_1:
        bfs_color.append('orange')
    else:
        bfs_color.append('red')
        
pos = dict([(n,n) for n in G.nodes])
nx.draw(G, node_color = bfs_color,with_labels = True,pos=pos)
