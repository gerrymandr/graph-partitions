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

###Required Libraries

import networkx as nx #for graph datatypes and such
import geopandas as gpd #for shapfile reading
import matplotlib.pyplot as plt #for general visualizations
import random #for random choices
import queue as q #for BFS searching
import copy

###Temp Stuff

#Temporary graph for testing
G = nx.grid_graph(dim = [10,10])
G.add_node("boundary")
for i in range(0,10):
    for j in range(0,10):
        if i == 0 or j == 0:
            G.add_edge('boundary', (i,j))
        if i == 9 or j == 9:
            G.add_edge('boundary', (i,j))

''' Takes in a networkx graph object and returns a list of list of nodes representing the partitions
'''
def rand_walk_partition(G, step_count, fVisualize = False):
    ###Random Walk Cut
    
    #Needed for visualization step
    graph = copy.deepcopy(G)
    
    #Begin Walk at the boundary
    rand_walk = []
    current = 'boundary'
    last = None
    count = 0

    #Populate the Walk
    while True:
        rand_walk.append(current)
        neighbors = list(graph.neighbors(current))
    
        #This section prevents doubling back in the random walk
        newnode = last
        while (newnode == last) or (newnode == 'boundary' and count < step_count): #The Boundary Bouncer
            index = random.randrange(0,len(neighbors))
            newnode = neighbors[index]
        
        last = current
        current = newnode
    
        #We only end the walk if we reach boundary and have taken a certain number of steps
        if current == 'boundary':
            break
            
        count+=1

    #Choose a node we know not to be in our cut and initialize our queue with it
    start = 'boundary'
    while start in rand_walk:
       start = random.choice(list(graph.nodes))
    partition_1 = []
    bfs_queue = q.Queue()
    bfs_queue.put(start)
    dist_color = []

    ###Populate the first partition using a BFS model

    while not bfs_queue.empty():
       node = bfs_queue.get()
       partition_1.append(node)
       for nbr in list(graph.neighbors(node)):
           if nbr not in partition_1 and nbr not in rand_walk and nbr not in list(bfs_queue.queue):
               bfs_queue.put(nbr)

    ###Allocate Partition 2 and output LoL for partitioning.

    partition_2 = []
    for other in list(graph.nodes):
       if other not in partition_1 and other != 'boundary':
           partition_2.append(other)
    if fVisualize == False:
        return [partition_1,partition_2]
    
    graph.remove_node('boundary')

    for node in graph:
        if node in partition_1:
            dist_color.append('orange')
        else:
            dist_color.append('red')
    
    for node in partition_2:
        for nbr in list(graph.neighbors(node)):
            if nbr in partition_1:
                graph.remove_edge(node,nbr)
    
    pos = dict([(n,n) for n in graph.nodes])
    nx.draw(graph, node_color = dist_color,pos=pos)

''' Creates a histogram comparing the size of the partitions created by the rand walk
'''
def equality_histogram(graph, step_count, num_runs):
    hist_list = []
    for i in range(num_runs):
        partitioning = rand_walk_partition(graph, step_count)
        score = len(partitioning[0]) - len(partitioning[1])
        hist_list.append(score)
    return plt.hist(hist_list)

''' Creates a histogram looking up how often each specific partition occurs (run on small graphs)
'''
def uniformity_histogram(graph, step_count, num_runs):
    hist_list = []
    for i in range(num_runs):
        partitioning = rand_walk_partition(graph, step_count)
        hist_list.append(partitioning)
    return plt.hist(hist_list)

'''
Current State of Random Walk: We ran histograms on 10 by 10 grid graphs with different step counts. It appears that requiring a walk of length 45 or 50 in this case produced the most even results. We were not able to get a spike around equal partitioning while only requiring minimum walk length. The major issues for low walk counts in that they would return to the boundary too quickly and therefore there would be a very large partition 1 in most cases and small partition 2. Large values had the opposite problem in which the walk would create many small areas and the random sampling would often create very small partition 1s. Upon talking to Justin it appears like a good next step would be to look at making the random walk respond to some sort of global potential we update on every iteration. Lorenzo suggested using a Brownian Bridge model to implement this.
'''