""" Tests for the implemented functions
"""
import unittest

from clustering_functions import *
from dsm_helper_classes import *
from plotting_functions import *

# Test case for DSMMatrix.reorder_by_cluster
if False:    
    d_mat = np.identity(3)
    d_list = ["1", "2", "3"]
    d = DSMMatrix(d_mat, d_list)
    print(d.mat)

    c_mat = np.array([[1, 1, 1], [0, 1, 1]])
    c = ClusterMatrix.from_mat(c_mat)
    print(c.mat)

    new_d_mat = DSMMatrix.reorder_by_cluster(d, c)
    print(new_d_mat.mat)
    print(new_d_mat.act_labels)


# Test case for ClusterMatrix.reorder_by_cluster
if True:
    c_mat = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]])
    c = ClusterMatrix.from_mat(c_mat)
    new_c_mat = ClusterMatrix.reorder_by_cluster(c)
    print(new_c_mat.mat)
    print(new_c_mat.cluster_size)

    # Checks that a new cluster matrix (once reordered) is invariant under reordering
    newer_c_mat = ClusterMatrix.reorder_by_cluster(new_c_mat)
    print(newer_c_mat.mat)


# Tests for dsm_helper_classes
class ClusterParametersTestCase(unittest.TestCase):
    raise NotImplementedError

class DSMMatrixTestCase(unittest.TestCase):
    raise NotImplementedError

class ClusterMatrixTestCase(unittest.TestCase):
    raise NotImplementedError

# Tests for clustering_functions
class BidTestCase(unittest.TestCase):
    raise NotImplementedError

class ClusterTestCase(unittest.TestCase):
    raise NotImplementedError

if __name__ == "__main__":
    unittest.main()