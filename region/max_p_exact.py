from math import floor, log10
import numbers

import libpysal as ps
import networkx as nx
import numpy as np
import pulp
from geopandas import GeoDataFrame
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum

from region.exact_algorithms import _get_solver_instance, ClusterExact
from region.util import dissim_measure


class MaxPExact:
    """
    A class for solving the p-regions problem by transforming it into a
    mixed-integer-programming problem as described in [1]_.

    Parameters
    ----------
    num_regions : `int`
        The number of regions the areas are clustered into.

    References
    ----------
    .. [1] J. C. Duque, L. Anselin, S. Rey (2012): "The Max-p-Regions Problem" in Journal of Regional Science, Vol. 52, No. 3, pp. 397-419
    """
    # todo: docstring (add labels_, method_, solver_, and fit&fit_from_...-methods)
    def __init__(self):

        self.labels_ = None
        self.solver_ = None

    def fit_from_dict(self, areas, attr, spatially_extensive_attr, threshold,
                      solver="cbc"):
        """\
        Parameters
        ----------
        areas : dict
            Each key represents an area and each value is an iterable of
            neighbors of this area.
        attr : dict
            A dict with the same keys as `areas` and values representing the
            attributes for calculating homo-/heterogeneity. A value can be
            scalar (e.g. `float` or `int`) or an `numpy.ndarray`.
        spatially_extensive_attr : dict
            A dict with the same keys as `areas` and values representing the
            spatially extensive attribute. In the Max-p-Regions problem each
            region's sum of spatially extensive attributes must be greater than
            a specified threshold.
        threshold : float
            The threshold for a region's sum of spatially extensive attributes.
        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            The solver to use. Unless the default solver is used, the user has
            to make sure that the specified solver is installed.
            * "cbc" - the Cbc (Coin-or branch and cut) solver
            * "cplex" - the CPLEX solver
            * "glpk" - the GLPK (GNU Linear Programming Kit) solver
            * "gurobi" - the Gurobi Optimizer
        """
        ClusterExact._check_solver(solver)  # todo: move this static method out of ClusterExact

        if not isinstance(areas, dict):
            raise ValueError("The areas argument must be dict.")
        neighbor_dict = areas

        # todo: arg checks
        # if len(neighbors_dict) < self.num_regions:
        #     raise ValueError("The number of regions must be less than the "
        #                      "number of areas.")
        #
        # if not isinstance(data, dict) or data.keys() != areas.keys():
        #     raise ValueError("The data argument has to be of type dict with "
        #                      "the same keys as areas.")

        prob = LpProblem("Max-p-Regions", LpMinimize)

        # Parameters of the optimization problem
        I = [area for area in areas]  # index for areas
        II = [(i, j) for i in I
                     for j in I]
        II_upper_triangle = [(i, j) for i, j in II if i < j]
        n = len(areas)
        K = range(n)  # index of potential regions, called k in Duque et al.
        O = range(n)  # index of contiguity order, called c in Duque et al.
        d = {(i, j): dissim_measure(attr[i], attr[j])
             for i, j in II_upper_triangle}
        h = 1 + floor(log10(sum(d[(i, j)] for i, j in II_upper_triangle)))

        # Decision variables
        t = LpVariable.dicts(
            "t",
            ((i, j) for i, j in II_upper_triangle),
            lowBound=0, upBound=1, cat=LpInteger)
        x = LpVariable.dicts(
            "x",
            ((i, k, o) for i in I for k in K for o in O),
            lowBound=0, upBound=1, cat=LpInteger)

        # Objective function
        # (1) in Duque et al. (2012): "The Max-p-Regions Problem"
        prob += -10**h * lpSum(x[i, k, 0] for k in K for i in I) \
            + lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

        # Constraints
        # (2) in Duque et al. (2012): "The Max-p-Regions Problem"
        for k in K:
            prob += lpSum(x[i, k, 0] for i in I) <= 1
        # (3) in Duque et al. (2012): "The Max-p-Regions Problem"
        for i in I:
            prob += lpSum(x[i, k, o] for k in K for o in O) == 1
        # (4) in Duque et al. (2012): "The Max-p-Regions Problem"
        for i in I:
            for k in K:
                for o in range(1, len(O)):
                    prob += x[i, k, o] <= lpSum(x[j, k, o-1]
                                                for j in neighbor_dict[i])
        # (5) in Duque et al. (2012): "The Max-p-Regions Problem"
        for k in K:
            lhs = lpSum(x[i, k, o] * spatially_extensive_attr[i]
                        for i in I for o in O)
            prob += lhs >= threshold * lpSum(x[i, k, 0] for i in I)
        # (6) in Duque et al. (2012): "The Max-p-Regions Problem"
        for i, j in II_upper_triangle:
            for k in K:
                prob += t[i, j] >= \
                        lpSum(x[i, k, o] + x[j, k, o] for o in O) - 1
        # (7) in Duque et al. (2012): "The Max-p-Regions Problem"
        # already in LpVariable-definition
        # (8) in Duque et al. (2012): "The Max-p-Regions Problem"
        # already in LpVariable-definition

        # additional constraint for speedup (p. 405 in [1]_)
        for o in O:
            prob += x[I[0], K[0], o] == (1 if o == 0 else 0)

        # Solve the optimization problem
        solver = _get_solver_instance(solver)  # todo: move this function to file not specific to the p-regions-problem
        print("start solving with", solver)
        # prob.writeLP("max-p-regions")  # todo: rm
        prob.solve(solver)
        print("solved")
        result_dict = {}
        for i in I:
            for k in K:
                for o in O:
                    if x[i, k, o].varValue == 1:
                        result_dict[i] = k
        self.labels_ = result_dict
        self.solver_ = solver
