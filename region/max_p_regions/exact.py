import collections
import numbers
from math import floor, log10

from geopandas import GeoDataFrame
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum

from region import fit_functions
from region.fit_functions import check_solver, get_solver_instance, \
    graph_attr_to_dict
from region.util import copy_func, dataframe_to_dict, \
    raise_distance_metric_not_set, set_distance_metric


class MaxPExact:
    """
    A class for solving the max-p-regions problem by transforming it into a
    mixed-integer-programming problem (MIP) as described in [DAR2012]_.

    Parameters
    ----------

    Attributes
    ----------
    labels_ : dict
        Each key is an area and each value the region it has been assigned to.
    solver_ : str
        The solver used in the last call of a fit-method.
    """
    def __init__(self):

        self.labels_ = None
        self.solver_ = None
        self.distance_metric = raise_distance_metric_not_set

    def fit_from_dict(self, neighbors_dict, attr, spatially_extensive_attr,
                      threshold, solver="cbc", distance_metric="euclidean"):
        """\
        Parameters
        ----------
        neighbors_dict : dict
            Each key represents an area and each value is an iterable of
            neighbors of this area.
        attr : dict
            A dict with the same keys as `neighbors_dict` and values
            representing the attributes for calculating homo-/heterogeneity. A
            value can be scalar (e.g. `float` or `int`) or a
            :class:`numpy.ndarray`.
        spatially_extensive_attr : dict
            A dict with the same keys as `neighbors_dict` and values
            representing the spatially extensive attribute (scalar or iterable
            of scalars). In the Max-p-Regions problem each region's sum of
            spatially extensive attributes must be greater than a specified
            threshold. In case of iterables of scalars as dict-values all
            elements of the iterable have to fulfill the condition.
        threshold : numbers.Real or iterable of numbers.Real
            The threshold for a region's sum of spatially extensive attributes.
            The argument's type is numbers.Real if the values of
            ``spatially_extensive_attr`` are scalar, otherwise the argument
            must be an iterable of scalars.
        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            The solver to use. Unless the default solver is used, the user has
            to make sure that the specified solver is installed.

            * "cbc" - the Cbc (Coin-or branch and cut) solver
            * "cplex" - the CPLEX solver
            * "glpk" - the GLPK (GNU Linear Programming Kit) solver
            * "gurobi" - the Gurobi Optimizer

        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        set_distance_metric(self, distance_metric)
        check_solver(solver)

        if not isinstance(neighbors_dict, dict):
            raise ValueError("The neighbors_dict argument must be dict.")

        if not isinstance(attr, dict) or attr.keys() != neighbors_dict.keys():
            raise ValueError("The attr argument has to be of type dict with "
                             "the same keys as neighbors_dict.")

        prob = LpProblem("Max-p-Regions", LpMinimize)

        # Parameters of the optimization problem
        I = [area for area in neighbors_dict]  # index for neighbors_dict
        II = [(i, j) for i in I
                     for j in I]
        II_upper_triangle = [(i, j) for i, j in II if i < j]
        n = len(neighbors_dict)
        K = range(n)  # index of potential regions, called k in Duque et al.
        O = range(n)  # index of contiguity order, called c in Duque et al.
        d = {(i, j): self.distance_metric(attr[i], attr[j])
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
                                                for j in neighbors_dict[i])
        # (5) in Duque et al. (2012): "The Max-p-Regions Problem"
        if isinstance(spatially_extensive_attr[I[0]], numbers.Real):
            for k in K:
                lhs = lpSum(x[i, k, o] * spatially_extensive_attr[i]
                            for i in I for o in O)
                prob += lhs >= threshold * lpSum(x[i, k, 0] for i in I)
        elif isinstance(spatially_extensive_attr[I[0]], collections.Iterable):
            for el in range(len(spatially_extensive_attr[I[0]])):
                for k in K:
                    lhs = lpSum(x[i, k, o] * spatially_extensive_attr[i][el]
                                for i in I for o in O)
                    if isinstance(threshold, numbers.Real):
                        rhs = threshold * lpSum(x[i, k, 0] for i in I)
                        prob += lhs >= rhs
                    elif isinstance(threshold, numbers.Real):
                        rhs = threshold[el] * lpSum(x[i, k, 0] for i in I)
                        prob += lhs >= rhs
        # (6) in Duque et al. (2012): "The Max-p-Regions Problem"
        for i, j in II_upper_triangle:
            for k in K:
                prob += t[i, j] >= \
                        lpSum(x[i, k, o] + x[j, k, o] for o in O) - 1
        # (7) in Duque et al. (2012): "The Max-p-Regions Problem"
        # already in LpVariable-definition
        # (8) in Duque et al. (2012): "The Max-p-Regions Problem"
        # already in LpVariable-definition

        # additional constraint for speedup (p. 405 in [DAR2012]_)
        for o in O:
            prob += x[I[0], K[0], o] == (1 if o == 0 else 0)

        # Solve the optimization problem
        solver = get_solver_instance(solver)
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

    fit = copy_func(fit_from_dict)
    fit.__doc__ = "Alias for :meth:`fit_from_dict`.\n\n" \
                  + fit_from_dict.__doc__

    def fit_from_geodataframe(self, areas, attr, spatially_extensive_attr,
                              threshold, solver="cbc",
                              distance_metric="euclidean", contiguity="rook"):
        """
        Parameters
        ----------
        areas : GeoDataFrame

        attr : str or list
            The clustering criteria (columns of the GeoDataFrame `areas`) are
            specified as string (for one column) or list of strings (for
            multiple columns).
        spatially_extensive_attr :

        threshold :
            See the corresponding argument in :meth:`fit_from_dict`.
        solver : str
            See the corresponding argument in :meth:`fit_from_dict`.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        contiguity : {"rook", "queen"}, default: "rook"
            Defines the contiguity relationship between areas. Possible
            contiguity definitions are:

            * "rook" - Rook contiguity.
            * "queen" - Queen contiguity.
        """
        if isinstance(spatially_extensive_attr, str):
            spatially_extensive_attr = [spatially_extensive_attr]
        else:  # isinstance(data, collections.Sequence)
            spatially_extensive_attr = list(spatially_extensive_attr)
        spatially_extensive_attr = dataframe_to_dict(areas,
                                                     spatially_extensive_attr)

        fit_functions.fit_from_geodataframe(self, areas, attr,
                                            spatially_extensive_attr,
                                            threshold, solver,
                                            distance_metric=distance_metric,
                                            contiguity=contiguity)

    def fit_from_networkx(self, areas, attr, spatially_extensive_attr,
                          threshold, solver="cbc",
                          distance_metric="euclidean"):
        """
        Parameters
        ----------
        areas : `networkx.Graph`

        attr : str, list or dict
            If the clustering criteria are present in the networkx.Graph
            `areas` as node attributes, then they can be specified as a string
            (for one criterion) or as a list of strings (for multiple
            criteria).
            Alternatively, a dict can be used with each key being a node of the
            networkx.Graph `areas` and each value being the corresponding
            clustering criterion (a scalar (e.g. `float` or `int`) or a
            :class:`numpy.ndarray`).
            If there are no clustering criteria are present in the
            networkx.Graph `areas` as node attributes, then a dictionary must
            be used for this argument.
        spatially_extensive_attr : str, list or dict
            If the spatially_extensive_attr is present in the networkx.Graph
            `areas` as node attributes, then they can be specified as a string
            (for one element (scalar or iterable of scalars)) or as a list of
            strings (for multiple elements).
            Alternatively, a dict can be used with each key being a node of the
            networkx.Graph `areas` and each value being the corresponding
            spatially_extensive_attr (a scalar (e.g. `float` or `int`), a
            :class:`numpy.ndarray` or an iterable of scalars).
            If there are no clustering criteria are present in the
            networkx.Graph `areas` as node attributes, then a dictionary must
            be used for this argument. See the corresponding argument in
            :meth:`fit_from_dict` for more details about the expected the
            expected dict.

        threshold :
            See the corresponding argument in :meth:`fit_from_dict`.
        solver : str
            See the corresponding argument in :meth:`fit_from_dict`.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        sp_ext_attr_dict = graph_attr_to_dict(areas, spatially_extensive_attr)
        fit_functions.fit_from_networkx(self, areas, attr, sp_ext_attr_dict,
                                        threshold, solver,
                                        distance_metric=distance_metric)

    def fit_from_w(self, areas, attr, spatially_extensive_attr, threshold,
                   solver="cbc", distance_metric="euclidean"):
        """
        Parameters
        ----------
        areas : libpysal.weights.W

        attr : dict
            See the corresponding argument in :meth:`fit_from_dict`.
        spatially_extensive_attr : dict
            See the corresponding argument in :meth:`fit_from_dict`.
        threshold :
            See the corresponding argument in :meth:`fit_from_dict`.
        solver : str
            See the corresponding argument in :meth:`fit_from_dict`.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        fit_functions.fit_from_w(self, areas, attr, spatially_extensive_attr,
                                 threshold, solver,
                                 distance_metric=distance_metric)
