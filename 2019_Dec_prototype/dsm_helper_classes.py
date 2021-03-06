import copy
import numpy as np

class DsmMatrix(object):
    """
    Contains activities (act_list) and dependencies (mat)
    
    Inside:
    1. A matrix (n x n) of dependencies, 1 means dependency exists, 0 otherwise
    2. An activity list (n) listing the activities that make up the DSM
    """
    def __init__(self, mat, act_labels):
        self.mat = mat
        self.act_labels = act_labels
        
    def __len__(self):
        """Allows use of python built-in len() function"""
        return len(self.act_labels)

    def Reorder_DSM_by_Cluster(self, dsm_cluster):
        """Returns a new dsm_matrix object expanding out the elements that are in multiple clusters. Also recalculates the size of this matrix"""

        # return DSM_matrix(new_mat, new_labels)
        cl_mat = dsm_cluster.cluster_mat
        ds_mat = self.mat

        num_ele = cl_mat[cl_mat > 0].size
        temp_mat = np.zeros((num_ele, len(self)))
        new_dsm_mat = DsmMatrix(np.zeros((num_ele,num_ele)), np.zeros(num_ele))

        ii = 0
        for cluster in cl_mat:
            for ind, flag in enumerate(cluster):
                if flag:
                    temp_mat[ii,:] = ds_mat[ind,:]
                    new_dsm_mat.act_labels[ii] = ind
                    ii += 1

        for ind, col in enumerate(new_dsm_mat.act_labels):
            new_dsm_mat.mat[:,ind] = temp_mat[:,col]

        return new_dsm_mat

class DsmCluster(object):
    """
    Tracks clusters (cluster_mat), size of clusters, and total coordination cost
    
    1. A matrix (n x n) capturing the clusters that exist
    2. An array (n) capturing the size of the various clusters
    3. A value (1) representing the total coordination cost
    """
    def __init__(self, n_clus):
        # Setup logic depending only on the dimension
        self.cluster_mat = np.zeros((n_clus, n_clus))
        self.cluster_size = np.zeros(n_clus)
        self.total_coord_cost = 0

    def deepcopy(self):
        """Preferred copying method using python std lib deepcopy"""
        return copy.deepcopy(self)

    def calc_coord_cost(self, dsm_matrix, cluster_parameters):
        """Calculates and updates the coordination cost of the current matrix"""
        coord_cost_list = []

        # Gets a new DSM matrix with duplicate rows depending on clustering
        new_dsm_mat = dsm_matrix.Reorder_DSM_by_Cluster(self)

        # Builds the new cluster matrix according to the dsm structure
        new_clu_mat = np.zeros((len(new_dsm_mat), len(self.cluster_size)))
        start = 0
        for ind, row in enumerate(self.cluster_mat):
            fin = start + row[row != 0].size
            new_clu_mat[ind, start:fin] = 1
            start = fin

        for ii in range(len(new_dsm_mat)):
            coord_cost_list.append(0)
            for jj in range(ii + 1, len(new_dsm_mat)):
                if (new_dsm_mat[ii, jj] > 0) or (new_dsm_mat[jj, ii] > 0):
                    cost_tot = 0

                    for cluster in new_clu_mat:
                        if (cluster[ii]) and (cluster[jj]):
                            cost_tot += (new_dsm_mat[ii,jj] \
                                        + new_dsm_mat[jj,ii]) \
                                        * (cluster[cluster != 0].size \
                                        ** cluster_parameters.pow_cc)

                    if cost_tot > 0:
                        coord_cost_list[ii] += cost_tot
                    else:
                        coord_cost_list[ii] += (new_dsm_mat[ii,jj] \
                                                + new_dsm_mat[jj,ii]) \
                                                * (len(new_dsm_mat) \
                                                ** cluster_parameters.pow_cc)

        self.total_coord_cost = sum(coord_cost_list)

    def update_cluster_mat(self, element, cluster_list):
        """Updates the matrix to reflect a new bid in cluster_list"""
        self.cluster_mat[:,element] = \
            np.logical_or(self.cluster_mat[:,element], cluster_list)

    def update_cluster_size(self):
        """Updates the cluster_size array to reflect changes in cluster_mat"""
        self.cluster_size = np.array([sum(row) for row in self.cluster_mat])
        # for entry in cluster_mat, sum row/col (need to check structure)

    def delete_redundant_clusters(self):
        """Deletes any duplicate or empty clusters"""
        raise NotImplementedError

class ClusterParameters(object):
    """Constants that define the behaviour of the optimisation algorithm"""
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