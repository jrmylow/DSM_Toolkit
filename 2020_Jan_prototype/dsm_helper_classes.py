""" Contains helper classes for containing the relevant DSM data
    1. ClusterParameters is for defining the behaviour of a specific optimisation algorithm
    2. DSMMatrix
    3. ClusterMatrix
"""
from collections import Counter
import copy
import numpy as np

# ----- Cluster Parameters -----------------------------------------------------
class ClusterParameters(object):
    """ Constants that define the behaviour of the optimisation algorithm
    """
    def __init__(self, pow_cc, pow_bid, pow_dep, max_cluster_size, rand_accept,              rand_bid, times, stable_limit, max_repeat):
        # Penalty functions for the clusters
        self.pow_cc = pow_cc
        self.pow_bid = pow_bid
        self.pow_dep = pow_dep

        # Parameters dictating the prob of accepting a non-better solution
        self.rand_accept = rand_accept
        self.rand_bid = rand_bid

        # Sets limits for the number of calculations/passes carried out
        self.max_cluster_size = max_cluster_size
        self.max_repeat = max_repeat
        self.times = times

        # Dictates the limit for which the algorithm is considered converged
        self.stable_limit = stable_limit

def set_default_cluster_parameters(max_size):
    """ Sets the cluster_parameters following the guidelines by R Thebeau (2000)
    """
    return ClusterParameters(
        pow_cc = 1,
        pow_bid = 1,
        pow_dep = 4,
        max_cluster_size = max_size,
        rand_accept = 1 / (2 * max_size),
        rand_bid = 1 / (2 * max_size),
        times = 2,
        stable_limit = 2,
        max_repeat = 10
    )
# ------------------------------------------------------------------------------

class DSMMatrix(object):
    """ Lists activities in act_labels and dependencies in mat (2-d matrix)
    """
    def __init__(self, mat, act_labels):
        """ Default constructor is to accept a matrix and list of labels. Checks size of matrix and labels for correctness
        """
        assert len(mat.shape) == 2, "Input matrix must be two-dimensional"
        assert mat.shape[0] == mat.shape[1], "Input matrix must be square"
        assert len(act_labels) == mat.shape[0], "Labels must match matrix"

        self.mat = mat
        self.act_labels = act_labels
    
    def __len__(self):
        """ Built in len() method gives characteristic dimension of container
        """
        return len(self.act_labels)

    @classmethod
    def from_size(cls, size):
        """ Alternate constructor for DSM matrix creating a blank matrix of given size and act_labels with all "None" values
        """
        mat = np.zeros([size, size])
        act_labels = [None for _ in range(size)]
        return cls(mat, act_labels)

    @staticmethod
    def reorder_by_cluster(dsm_matrix, cluster_matrix):
        """ Returns a new dsm_matrix object expanding out the elements in multiple clusters, also recalculating the size of this matrix
        """
        assert cluster_matrix.num_activities == len(dsm_matrix), "Number of activities in cluster_matrix should be same as the dsm_matrix"

        cl_mat = cluster_matrix.mat
        ds_mat = dsm_matrix.mat

        num_ele = cl_mat[cl_mat > 0].size
        temp_mat = np.zeros([num_ele, len(dsm_matrix)])
        new_ds_mat = DSMMatrix.from_size(num_ele)

        ii = 0
        for cluster in cl_mat:
            for ind, flag in enumerate(cluster):
                if flag:
                    temp_mat[ii,:] = ds_mat[ind,:]
                    new_ds_mat.act_labels[ii] = ind
                    ii += 1

        for ind, col in enumerate(new_ds_mat.act_labels):
            new_ds_mat.mat[:,ind] = temp_mat[:,col]
            new_ds_mat.act_labels[ind] = dsm_matrix.act_labels[col]

        return new_ds_mat

class ClusterMatrix(object):
    def __init__(self, n_clus):
        """ Default constructor is to create a symmetric matrix and vector to track the cluster sizes
        """
        self.mat = np.zeros((n_clus, n_clus))
        self._num_activities = n_clus
        self.update_cluster_size()
        self.total_coord_cost = 0

    @classmethod
    def from_mat(cls, mat):
        """ Alternate constructor from a given input matrix
        """
        num_act = mat.shape[1]
        cluster_mat = ClusterMatrix(num_act)
        cluster_mat.mat = mat
        cluster_mat.update_cluster_size()

        return cluster_mat

    @property
    def num_clusters(self):
        return len(self.cluster_size)

    @property
    def num_activities(self):
        return self._num_activities

    def update_cluster_size(self):
        """ Updates the cluster_size array to reflect changes in mat
        """
        self.cluster_size = np.array([row[row > 0].size for row in self.mat])

    def update_mat(self, element, cluster_list):
        """ Updates the matrix to reflect a new bid in input cluster_list
        """
        assert len(cluster_list) == self.num_activities, "Number of elements in cluster_list must match the number of activities in this cluster matrix"
        self.mat[:,element] = \
            np.logical_or(self.mat[:,element], cluster_list)

    @staticmethod
    def reorder_by_cluster(cluster_matrix):
        """ Reorders the clusters in order as follows:

            The matrix:
            [[1, 0],                
             [1, 1]]

            becomes:
            [[1, 0, 0],
             [0, 1, 1]]

            This function guarantees that an ordered cluster will be reordered to itself
        """
        new_clu_mat = ClusterMatrix.from_mat(np.zeros(\
            [cluster_matrix.num_clusters, sum(cluster_matrix.cluster_size)]))

        start = 0
        for ind, row in enumerate(cluster_matrix.mat):
            fin = start + row[row != 0].size
            new_clu_mat.mat[ind, start:fin] = 1
            start = fin

        new_clu_mat.update_cluster_size()
        return new_clu_mat