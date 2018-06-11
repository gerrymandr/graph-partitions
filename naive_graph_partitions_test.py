# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:18:38 2018

@author: pgirardet
"""
import networkx as nx
import naive_graph_partitions as gp 
import unittest

bell_numbers = {0:1, 1:1, 2:2, 3:5, 4:15, 5:52, 6:203, 7:877, 8:4140, 9:21147, 
                10:115975, 11:678570, 12:4213597, 13:27644437}

# The size of lists/graphs being used for all tests. Shouldn't go too far above 
# 10 to run in a reasonable amount of time - TEST_SIZE = 12 took me 116 seconds.
TEST_SIZE = 10

class TestGraphPartitiosn(unittest.TestCase):

    # Number of list partitions should equal nth Bell number
    def test_list_partitions(self):
        for i in range(TEST_SIZE):
            test_list = list(range(i))
            test_partitions = list(gp.list_partitions(test_list))
            self.assertEqual(len(test_partitions), bell_numbers[i])
        print("Correctly enumerated list partitions")

    # Number of partitions of a path graph of length n should be the same as 
    # the number of list partitions for a length n list. 
    def test_graph_partitions(self):
        for i in range(TEST_SIZE):
            G = nx.path_graph(i)
            g_partitions = list(gp.graph_partitions(G))
            self.assertEqual(len(g_partitions), bell_numbers[i])
        print("Correctly enumerated graph partitions")

    # Number of connected partitions of a path graph of length n should be 
    # 2^(n-1) by a stars and bars argument. 
    def test_connected_graph_partitions(self):
        for i in range(1, TEST_SIZE):        
            G = nx.path_graph(i)
            conn_partitions = list(gp.connected_graph_partitions(G))
            self.assertEqual(len(conn_partitions), 2**(i-1))
        print("Correctly enumerated connected graph partitions")
        
if __name__ == '__main__':
    unittest.main()