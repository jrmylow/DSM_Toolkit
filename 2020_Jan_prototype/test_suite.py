""" Tests for the implemented functions
"""
import unittest

from clustering_functions import *
from dsm_helper_classes import *
from plotting_functions import *

# Test case for reorder_by_cluster
if True:    
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