import copy
import numpy as np
import random as rd

class DSM_matrix(object):
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


class DSM_cluster(object):
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
        new_dsm_mat = Reorder_DSM_by_Cluster(dsm_matrix, self)
        new_clu_mat = np.zeros((len(new_dsm_mat), len(new_dsm_mat)))
        self.total_coord_cost = sum(coord_cost_list)
        raise NotImplementedError

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

class cluster_parameters(object):
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

def Bid(element, dsm_matrix, dsm_cluster, cluster_parameters):
    """
    Calculates the bid of every element for the cluster, with penalties
    Returns a numpy array of same length as number of rows in dsm_cluster
    """
    rows, cols = dsm_cluster.cluster_mat.size
    bid_list = []
    for row in range(rows):
        in_var = 0
        out_var = 0

        for col in range(cols):
            if (dsm_cluster.cluster_mat[row,col] == 1) and (col != element):
                # Checks whether the given element has input/output interactions
                if (dsm_matrix.mat[col, element] > 0):
                    in_var += dsm_matrix.mat[col, element]
                if (dsm_matrix.mat[element, col] > 0):
                    out_var += dsm_matrix.mat[element, col]
        
        if (in_var > 0) or (out_var > 0):
            if (dsm_cluster.cluster_size[row] == cluster_parameters.max_cluster_size):
                bid_list.append(0)
            else:
                bid_list.append(
                    (in_var + out_var)**cluster_parameters.pow_dep / (dsm_cluster.cluster_size[row]**cluster_parameters.pow_bid)
                )
    return np.array(bid_list)

def Cluster(dsm_cluster, dsm_matrix, cluster_parameters):
    """Returns clustered DSM from matrix and parameters, and history data"""
    # Setup of the clusters with the appropriate size 
    # i.e. the number of items in the DSM work breakdown structure.
    n_clus = len(dsm_matrix)
    curr_cluster = DSM_cluster(n_clus)
    new_cluster = DSM_cluster(n_clus)

    # Update the current cluster and set benchmark_cost as benchmark
    curr_cluster.cluster_mat = np.identity(n_clus)
    curr_cluster.update_cluster_size()
    curr_cluster.calc_coord_cost(dsm_matrix, cluster_parameters)
    benchmark_cost = curr_cluster.total_coord_cost

    cost_history = []
    data_history = []
    best_cluster = curr_cluster.deepcopy()

    # Setup of flags for clustering routine
    first_run = True
    counter = 0

    # Continues looping until one of two stop flags are hit. Either:
    #   1. Current cluster total coord cost is equal to or better than best
    #   2. Maximum number of repeats have been passed through
    while ((curr_cluster.total_coord_cost > benchmark_cost) \
            and (counter <= cluster_parameters.max_repeat)) \
            or first_run:
        
        first_run = False

        # Tracks the current history
        data_history.append(curr_cluster)
        cost_history.append(curr_cluster.total_coord_cost)

        # Resets the run to the best cluster
        curr_cluster = best_cluster.deepcopy()
        counter += 1

        stable = 0
        change = True
        accept = False

        # Continues looping until the solution is stable
        while stable < cluster_parameters.stable_limit:
            for _ in range(n_clus * cluster_parameters.times):
                # Bidding process
                element = rd.randint(0, len(dsm_matrix) - 1)
                cluster_bid = Bid(element, dsm_matrix, dsm_cluster,                               cluster_parameters)
                bid_1st = max(cluster_bid)
                bid_2nd = max(cluster_bid[cluster_bid != bid_1st])

                # Accepts the second best bid with a prob 1/rand_bid
                if rd.random() < 1/cluster_parameters.rand_bid:
                    bid_1st = bid_2nd
                
                # Checks to see whether the bid produces an improvement
                if bid_1st > 0:
                    # If there is no existing relation for an element and the element's bid is equal to bid_1st, update cluster_list to track which clusters are affected
                    cluster_list = np.logical_and(
                        cluster_bid[cluster_bid == bid_1st],
                        np.equal(curr_cluster.cluster_mat[:,element], 0)
                    )

                    # Updates the cluster matrix and cluster size
                    new_cluster.update_cluster_mat(element, cluster_list)
                    new_cluster.update_cluster_size()
                    new_cluster.delete_redundant_clusters()
                    new_cluster.calc_coord_cost(dsm_matrix, cluster_parameters)

                    # Accepts a worse total cost with a prob 1/rand_accept
                    if (new_cluster.total_coord_cost <= curr_cluster.total_coord_cost):
                        accept = True

                    elif rd.random() < 1/cluster_parameters.rand_accept:
                        accept = True
                        if (curr_cluster.total_coord_cost < best_cluster.total_coord_cost):
                            best_cluster = curr_cluster.deepcopy()
                
                if accept:
                    accept = False
                    curr_cluster = new_cluster.deepcopy()
                    cost_history.append(curr_cluster.total_coord_cost)

                    if (curr_cluster.total_coord_cost < benchmark_cost):
                        benchmark_cost = curr_cluster.total_coord_cost
                        change = True

            if change:
                stable = 0
                change = False
            else:
                stable += 1
    else:
        return curr_cluster, cost_history, data_history

def Reorder_DSM_by_Cluster(dsm_matrix, dsm_cluster):
    """Returns a new dsm_matrix object expanding out the elements that are in multiple clusters. Also recalculates the size of this matrix"""

    # return DSM_matrix(new_mat, new_labels)
    cl_mat = dsm_cluster.cluster_mat
    ds_mat = dsm_matrix.mat

    num_ele = cl_mat[cl_mat > 0].size
    temp_mat = np.zeros((num_ele, len(dsm_matrix)))
    new_dsm_mat = DSM_matrix(np.zeros((num_ele,num_ele)), np.zeros(num_ele))

    ii = 0
    for cluster in cl_mat:
        for flag, ind in enumerate(cluster):
            if flag:
                temp_mat[ii,:] = ds_mat[ind,:]
                new_dsm_mat.act_labels[ii] = ind
                ii += 1

    for ind, col in enumerate(new_dsm_mat.act_labels):
        new_dsm_mat.mat[:,ind] = temp_mat[:,col]

    return new_dsm_mat
    