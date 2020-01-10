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