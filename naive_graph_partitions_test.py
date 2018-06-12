# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:18:38 2018

@author: Patrick Girardet 
"""
import networkx as nx
import naive_graph_partitions as gp 
import unittest
import scipy.special as sp
import time 

bell_numbers = {0:1, 1:1, 2:2, 3:5, 4:15, 5:52, 6:203, 7:877, 8:4140, 9:21147, 
                10:115975, 11:678570, 12:4213597, 13:27644437}

stirling_numbers = {(5,2):15, (5,3):25, (5,4):10, (6,2):31, (6,3):90, (6,4):65, 
                    (6,5):15, (7,2):63, (7,3):301, (7,4):350, (7,5):140, (7,6):21,
                    (8,2):127, (8,3):966, (8,4):1701, (8,5):1050, (8,6):266, (8,7):28,
                    (9,2):255, (9,3):3025, (9,4):7770, (9,5):6951, (9,6):2646, 
                    (9,7):462, (9,8):36, (10,2):511, (10,3):9330, (10,4):34105, 
                    (10,5):42525, (10,6):22827, (10,7):5880, (10,8):750, (10,9):45}


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
    
    # Number of k-partitions of a n-element set should be the Stirling number 
    # of the second kind S(n,k) by definition.    
    def test_k_list_partitions(self):
        for i in range(5, TEST_SIZE):
            for k in range(2, i): 
                test_list = list(range(i))
                test_partitions = list(gp.k_list_partitions(test_list, k))
                self.assertEqual(len(test_partitions), stirling_numbers[(i,k)])
        print("Correctly enumerated k-list partitions")
        
    # Number of k-partitions of the vertices of a path graph of length n should 
    # again be S(n,k).
    def test_k_graph_partitions(self):
        for i in range(5, TEST_SIZE):
            for k in range(2, i): 
                G = nx.path_graph(i)
                g_partitions = list(gp.k_graph_partitions(G, k))
                self.assertEqual(len(g_partitions), stirling_numbers[(i,k)])
        print("Correctly enumerated k-graph partitions")
    
    # Number of connected k-partitions of the vertices of a path graph of length 
    # n should be n-1 choose k-1 by a stars and bars argument. 
    def test_connected_k_graph_partitions(self):
        for i in range(5, TEST_SIZE):
            for k in range(2, i): 
                G = nx.path_graph(i)
                conn_partitions = list(gp.k_connected_graph_partitions(G, k))
                self.assertEqual(len(conn_partitions), sp.binom(i-1, k-1))
        print("Correctly enumerated k-graph partitions")
       
    # Compares the performance of the two different implementations of k-list 
    # partitioning. 
    def test_k_partition_profiling(self):
        test_list = list(range(10))
        eps = 10 ** (-6)
        for k in range(11):
            start = time.time()
            naive_val = len(list(gp.k_list_partitions(test_list, k)))
            naive_time = time.time() - start
            start = time.time()
            dp_val = len(list(gp.dynamic_k_list(test_list, k)))
            dp_time = time.time() - start
            self.assertEqual(dp_val, naive_val)
            print("Naive time: {}".format(naive_time))
            print("DP time: {}".format(dp_time))
            print("Naive / DP time: {}".format((naive_time + eps) / (dp_time + eps)))
            print("==================")
            
        
if __name__ == '__main__':
    unittest.main()