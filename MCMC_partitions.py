"""
Created on Mon Jun 11 10:37:28 2018

@author: MGGG
"""

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


# Check if removing the proposed node from its current graph would disconnect the graph.
# Input: A subgraph and a node from said graph.
def removal_remains_connected(H, v):
    if len(H.nodes()) == 1:
        # Because Network X has connectivity as undefined for the empty graph...
        return False
    W = H.copy()
    W.remove_node(v)
    return nx.is_connected(W)


# Make the initial districts using agglomeration.
def district_maker():
    graphs = M.generate_districts()
    for i in graphs:
        # Reject maps where any districts are small.
        if i.number_of_nodes() <= 5:
            return district_maker()
    return graphs


class maps:
    def __init__(self, K, n_districts, m):
        self.G = K
        self.district_list = []
        self.number_districts = n_districts
        self.graph_width = m
        self.partitions = []
        self.boundary_nodes_list = []
        self.districts = {}
        self.boundary_nodes = set([])

    # Pick a random node from the node boundary of the graph.
    # Input: a graph
    def random_neighbor(self, graph):
        return random.sample(nx.node_boundary(self.G, graph.nodes()), 1)

    # Generate the specified number of districts using agglomeration.
    def generate_districts(self):
        graphs = []
        for i in self.G.nodes():
            g = nx.Graph()
            g.add_node(i)
            graphs.append(g)
        while len(graphs) != self.number_districts:
            random_index = np.random.randint(0, high=len(graphs))
            random_neighbor = self.random_neighbor(graphs[random_index])
            target_graph = nx.Graph()
            for graph in graphs:
                if graph.has_node(random_neighbor[0]):
                    target_graph = graph
                    graphs[random_index] = nx.compose(graphs[random_index], target_graph)
                    break
            graphs.remove(target_graph)
        return graphs

    # Check if all districts are connected.
    def components_connected(self):
        for g in self.district_list:
            if not nx.is_connected(g):
                return False
        return True

    # Computes the number of nodes that could possibly be flipped with the current partition.
    def number_options(self):
        count = 0
        for v in self.G.nodes():
            if self.check_proposal(v):
                count += 1
        return count

    # Check if the proposal violates any restrictions.
    # Input: a proposed node to change district assignment
    def check_proposal(self, v):
        k = self.G.node[v]['district']
        H = self.districts[k]
        return removal_remains_connected(H, v)

    # Creates the district graphs.
    def make_district_list(self):
        self.district_list = []
        for k in range(self.number_districts):
            nbunch = [i for i in self.G.nodes() if self.G.node[i]['district'] == k]
            H = self.G.subgraph(nbunch)
            self.district_list.append(H)
            self.districts[k] = H

    # Establish the entire set of district edge nodes.
    def initialize_boundary(self):
        for i in range(self.number_districts):
            X = list(nx.node_boundary(K, self.districts[i].nodes()))
            self.boundary_nodes = self.boundary_nodes.union(X)
            self.boundary_nodes_list.append(X)

    # Randomly select a node at the edge of a district.
    def pick_boundary_node(self):
        while True:
            n = random.sample(self.boundary_nodes, 1)[0]
            n_graph = self.districts[(self.G.node[n]["district"])]
            if removal_remains_connected(n_graph, n):
                possible_swaps = []
                for i in range(len(self.boundary_nodes_list)):
                    for node in self.boundary_nodes_list[i]:
                        if node == n:
                            possible_swaps.append(i)
                f = random.sample(possible_swaps, 1)[0]
                return n, f

    # The case to deal with when a district is empty.
    def empty_district(self):
        return True

    # Use the Metropolis-Hastings method to accept random proposed nodes to change.
    def metropolis_hastings(self):
        if 0 in [len(g.nodes()) for g in self.district_list]:
            # Original code had this as a separate case.
            M.MH_empty_district_case()
            return
        node, new_district = self.pick_boundary_node()
        old_district = self.G.node[node]["district"]
        old_deg = self.number_options()
        self.update(node, old_district, new_district)
        self.make_district_list()
        new_deg = self.number_options()
        a = np.min([1, old_deg / new_deg])
        coin = np.random.rand(1)
        if coin > a:
            self.update(node, new_district, old_district)
            self.make_district_list()

    # Update all the districts and their respective nodes.
    # Input: a node, two districts
    # Mostly adapted from the MCMC code in the Github.
    def update(self, node, old_district, new_district):
        other_districts_nodes = nx.Graph()
        for i in range(self.number_districts):
            if i is not old_district:
                other_districts_nodes.add_nodes_from(self.districts[i].nodes())
        other_districts = self.G.subgraph(other_districts_nodes)
        old_boundary_nodes = nx.node_boundary(self.G, [node], other_districts)
        for n in old_boundary_nodes:
            if n in self.boundary_nodes:
                self.boundary_nodes.remove(n)
            if n in self.boundary_nodes_list:
                self.boundary_nodes_list[old_district].remove(n)
        updated_old_district = set(self.district_list[old_district].nodes())
        updated_old_district.remove(node)
        updated_new_district = set(self.district_list[new_district].nodes())
        updated_new_district.add(node)
        self.G.node[node]["district"] = new_district
        self.district_list[old_district] = self.G.subgraph(updated_old_district)
        self.district_list[new_district] = self.G.subgraph(updated_new_district)

    # Plots the graph and it's districts by color.
    # Input: A graoh
    def visual(self, graph):
        plt.figure(figsize=(6, 6))
        # Need to change the position dictionary if not using a grid graph.
        pos = dict((n, n) for n in graph.nodes())
        districts = []
        for node in self.G.nodes():
            districts.append(self.G.node[node]["district"])
        nx.draw_networkx(graph,
                         pos=pos,
                         with_labels=True,
                         font_size=8,
                         node_size=700,
                         node_color=districts,
                         cmap='tab10')
        plt.axis('off')


# For making a grid graph.
m = 10
K = nx.grid_2d_graph(m, m)
# Initialize the entire graph.
num_districts = 4
M = maps(K, num_districts, m)
graphs = district_maker()
# Store what district a node belongs to in the node attributes.
for i in range(len(graphs)):
    for node in K.nodes():
        if graphs[i].has_node(node):
            K.node[node]['district'] = i
M.make_district_list()
M.initialize_boundary()
# Run the visual and the MH Algorithm, pausing occasionally to show changes in the graph.
M.visual(K)
plt.pause(0.2)
for i in range(100):
    M.metropolis_hastings()
    if i % 2 == 0:
        M.visual(K)
        plt.pause(0.1)
        plt.show()

