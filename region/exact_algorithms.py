import numbers

from geopandas import GeoDataFrame
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum

from region import fit_functions
from region.fit_functions import check_solver, get_solver_instance
from region.util import dissim_measure, find_sublist_containing, copy_func


class ClusterExact:
    """
    A class for solving the p-regions problem by transforming it into a
    mixed-integer-programming problem (MIP) as described in [DCM2011]_.

    Parameters
    ----------
    n_regions : int
        The number of regions the areas are clustered into.

    Attributes
    ----------
    labels_ : dict
        Each key is an area and each value the region it has been assigned to.
    method_ : str
        The method used in the last call of a fit-method for translating the
        p-regions problem into an MIP.
    solver_ : str
        The solver used in the last call of a fit-method.
    """
    def __init__(self, n_regions):
        if not isinstance(n_regions, numbers.Integral) or n_regions <= 0:
            raise ValueError("The n_regions argument must be a positive "
                             "integer.")
        self.n_regions = n_regions
        self.labels_ = None
        self.method_ = None
        self.solver_ = None

    def fit_from_dict(self, neighbors_dict, data, method="flow", solver="cbc"):
        """\
        Parameters
        ----------
        neighbors_dict : dict
            Each key represents an area and each value is an iterable of
            neighbors of this area.
        data : dict
            A dict with the same keys as `neighbors_dict` and values
            representing the clustering criteria. A value can be scalar (e.g.
            float or int) or a :class:`numpy.ndarray`.
        method : {"flow", "order", "tree"}, default: "flow"
            The method to translate the clustering problem into an exact
            optimization model.

            * "flow" - Flow model on p. 112-113 in [DCM2011]_
            * "order" - Order model on p. 110-112 in [DCM2011]_
            * "tree" - Tree model on p. 108-110 in [DCM2011]_

        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            The solver to use. Unless the default solver is used, the user has
            to make sure that the specified solver is installed.

            * "cbc" - the Cbc (Coin-or branch and cut) solver
            * "cplex" - the CPLEX solver
            * "glpk" - the GLPK (GNU Linear Programming Kit) solver
            * "gurobi" - the Gurobi Optimizer
        """
        self._check_method(method)
        check_solver(solver)

        if not isinstance(neighbors_dict, dict):
            raise ValueError("The neighbors_dict argument must be dict.")
        neighbors_dict = neighbors_dict

        if len(neighbors_dict) < self.n_regions:
            raise ValueError("The number of regions must be less than the "
                             "number of neighbors_dict.")

        if not isinstance(data, dict) or data.keys() != neighbors_dict.keys():
            raise ValueError("The data argument has to be of type dict with "
                             "the same keys as neighbors_dict.")
        values_dict = data

        opt_func = {"flow": _flow,
                    "order": _order,
                    "tree": _tree}[method.lower()]
        result_dict = opt_func(neighbors_dict, values_dict, self.n_regions,
                               solver)
        self.labels_ = result_dict
        self.method_ = method
        self.solver_ = solver

    fit = copy_func(fit_from_dict)
    fit.__doc__ = "Alias for :meth:`fit_from_dict`.\n\n" \
                  + fit_from_dict.__doc__

    def fit_from_geodataframe(self, areas, data, method="flow", solver="cbc",
                              contiguity="rook"):
        """

        Parameters
        ----------
        areas : GeoDataFrame

        data : str or list
            The clustering criteria (columns of the GeoDataFrame `areas`) are
            specified as string (for one column) or list of strings (for
            multiple columns).
        method : str
            See the corresponding argument in :meth:`fit_from_dict`.
        solver : str
            See the corresponding argument in :meth:`fit_from_dict`.
        contiguity : str
            See the corresponding argument in
            :func:`region.fit_functions.fit_from_geodataframe`.
        """
        fit_functions.fit_from_geodataframe(self, areas, data, method, solver,
                                            contiguity=contiguity)

    def fit_from_networkx(self, areas, data, method="flow", solver="cbc"):
        """

        Parameters
        ----------
        areas : `networkx.Graph`

        data : str, list or dict
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
            be used for this argument. See the corresponding argument in
            :meth:`fit_from_dict` for more details about the expected the
            expected dict.
        method : str
            See the corresponding argument in :meth:`fit_from_dict`.
        solver : str
            See the corresponding argument in :meth:`fit_from_dict`.
        """
        fit_functions.fit_from_networkx(self, areas, data, method, solver)

    def fit_from_w(self, areas, data, method="flow", solver="cbc"):
        """

        Parameters
        ----------
        areas : libpysal.weights.W

        data : dict
            See the corresponding argument in :meth:`fit_from_dict`.
        method : str
            See the corresponding argument in :meth:`fit_from_dict`.
        solver : str
            See the corresponding argument in :meth:`fit_from_dict`.
        """
        fit_functions.fit_from_w(self, areas, data, method, solver)

    @staticmethod
    def _check_method(method):
        if not isinstance(method, str) \
                or method.lower() not in ["flow", "order", "tree"]:
            raise ValueError("The method argument must be one of the following"
                             ' strings: "flow", "order", or "tree".')


def _flow(neighbor_dict, data, n_regions, solver):
    """
    Parameters
    ----------
    neighbor_dict : dict
        The keys represent the areas. The values represent the corresponding
        neighbors.
    data : dict
        See the corresponding argument in :meth:`fit_from_dict`.
    n_regions : int
        The number of regions the areas are clustered into.
    solver : str
        See the corresponding argument in :meth:`ClusterExact.fit_from_dict`.

    Returns
    -------
    result : dict
        The keys represent the areas. Each value specifies the region an area
        has been assigned to.
    """
    print("running FLOW algorithm")  # TODO: rm
    prob = LpProblem("Flow", LpMinimize)

    # Parameters of the optimization problem
    n = len(data)
    I = list(data.keys())  # index for areas
    II = [(i, j)
          for i in I
          for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    K = range(n_regions)  # index for regions
    d = {(i, j): dissim_measure(data[i], data[j])
         for i, j in II_upper_triangle}

    # Decision variables
    t = LpVariable.dicts(
        "t",
        ((i, j) for i, j in II_upper_triangle),
        lowBound=0, upBound=1, cat=LpInteger)
    f = LpVariable.dicts(           # The amount of flow (non-negative integer)
        "f",                        # from area i to j in region k.
        ((i, j, k) for i in I for j in neighbor_dict[i] for k in K),
        lowBound=0, cat=LpInteger)
    y = LpVariable.dicts(  # 1 if area i is assigned to region k. 0 otherwise.
        "y",
        ((i, k) for i in I for k in K),
        lowBound=0, upBound=1, cat=LpInteger)
    w = LpVariable.dicts(  # 1 if area i is chosen as a sink. 0 otherwise.
        "w",
        ((i, k) for i in I for k in K),
        lowBound=0, upBound=1, cat=LpInteger)

    # Objective function
    # (20) in Duque et al. (2011): "The p-Regions Problem"
    prob += lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

    # Constraints
    # (21) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += sum(y[i, k] for k in K) == 1
    # (22) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for k in K:
            prob += w[i, k] <= y[i, k]
    # (23) in Duque et al. (2011): "The p-Regions Problem"
    for k in K:
        prob += sum(w[i, k] for i in I) == 1
    # (24) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbor_dict[i]:
            for k in K:
                prob += f[i, j, k] <= y[i, k] * (n - n_regions)
    # (25) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbor_dict[i]:
            for k in K:
                prob += f[i, j, k] <= y[j, k] * (n - n_regions)
    # (26) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for k in K:
            lhs = sum(f[i, j, k] - f[j, i, k] for j in neighbor_dict[i])
            prob += lhs >= y[i, k] - (n - n_regions) * w[i, k]
    # (27) in Duque et al. (2011): "The p-Regions Problem"
    for i, j in II_upper_triangle:
        for k in K:
            prob += t[i, j] >= y[i, k] + y[j, k] - 1
    # (28) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (29) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (30) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (31) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition

    # Solve the optimization problem
    solver = get_solver_instance(solver)
    # prob.writeLP("flow")  # todo: rm
    prob.solve(solver)
    result = {}
    for i in I:
        for k in K:
            if y[i, k].varValue == 1:
                result[i] = k
    return result


def _order(neighbor_dict, data, n_regions, solver):
    """
    Parameters
    ----------
    neighbor_dict : dict
        The keys represent the areas. The values represent the corresponding
        neighbors.
    data : dict
        See the corresponding argument in :meth:`fit_from_dict`.
    n_regions : int
        The number of regions the areas are clustered into.
    solver : str
        See the corresponding argument in :meth:`ClusterExact.fit_from_dict`.

    Returns
    -------
    result : dict
        The keys represent the areas. Each value specifies the region an area
        has been assigned to.
    """
    print("running ORDER algorithm")  # TODO: rm
    prob = LpProblem("Order", LpMinimize)

    # Parameters of the optimization problem
    n = len(data)
    I = list(data.keys())  # index for areas
    II = [(i, j)
          for i in I
          for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    K = range(n_regions)  # index for regions
    O = range(n - n_regions)  # index for orders
    d = {(i, j): dissim_measure(data[i], data[j])
         for i, j in II_upper_triangle}

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
    # (13) in Duque et al. (2011): "The p-Regions Problem"
    prob += lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

    # Constraints
    # (14) in Duque et al. (2011): "The p-Regions Problem"
    for k in K:
        prob += sum(x[i, k, 0] for i in I) == 1
    # (15) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += sum(x[i, k, o] for k in K for o in O) == 1
    # (16) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for k in K:
            for o in range(1, len(O)):
                    prob += x[i, k, o] <= \
                            sum(x[j, k, o-1] for j in neighbor_dict[i])
    # (17) in Duque et al. (2011): "The p-Regions Problem"
    for i, j in II_upper_triangle:
        for k in K:
            summ = sum(x[i, k, o] + x[j, k, o] for o in O) - 1
            prob += t[i, j] >= summ
    # (18) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (19) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition

    # Solve the optimization problem
    solver = get_solver_instance(solver)
    prob.solve(solver)
    result = {}
    for i in I:
        for k in K:
            for o in O:
                if x[i, k, o].varValue == 1:
                    result[i] = k
    return result


def _tree(neighbor_dict, data, n_regions, solver):
    """
    Parameters
    ----------
    neighbor_dict : dict
        The keys represent the areas. The values represent the corresponding
        neighbors.
    data : dict
        See the corresponding argument in :meth:`fit_from_dict`.
    n_regions : int
        The number of regions the areas are clustered into.
    solver : str
        See the corresponding argument in :meth:`ClusterExact.fit_from_dict`.

    Returns
    -------
    result : dict
        The keys represent the areas. Each value specifies the region an area
        has been assigned to.
    """
    print("running TREE algorithm")  # TODO: rm
    prob = LpProblem("Tree", LpMinimize)

    # Parameters of the optimization problem
    n = len(data)
    I = list(data.keys())
    II = [(i, j)
          for i in I
          for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    d = {(i, j): dissim_measure(data[i], data[j])
         for i, j in II}
    # Decision variables
    t = LpVariable.dicts(
        "t",
        ((i, j) for i, j in II),
        lowBound=0, upBound=1, cat=LpInteger)
    x = LpVariable.dicts(
        "x",
        ((i, j) for i, j in II),
        lowBound=0, upBound=1, cat=LpInteger)
    u = LpVariable.dicts(
        "u",
        (i for i in I),
        lowBound=0, cat=LpInteger)

    # Objective function
    # (3) in Duque et al. (2011): "The p-Regions Problem"
    prob += lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

    # Constraints
    # (4) in Duque et al. (2011): "The p-Regions Problem"
    lhs = lpSum(x[i, j] for i in I for j in neighbor_dict[i])
    prob += lhs == n - n_regions
    # (5) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += lpSum(x[i, j] for j in neighbor_dict[i]) <= 1
    # (6) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in I:
            for m in I:
                if i != j and i != m and j != m:
                    prob += t[i, j] + t[i, m] - t[j, m] <= 1
    # (7) in Duque et al. (2011): "The p-Regions Problem"
    for i, j in II:
        prob += t[i, j] - t[j, i] == 0
    # (8) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbor_dict[i]:
            prob += x[i, j] <= t[i, j]
    # (9) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbor_dict[i]:
            prob += u[i] - u[j] + (n - n_regions) * x[i, j] \
                    + (n - n_regions - 2) * x[j, i] <= n - n_regions - 1
    # (10) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += u[i] <= n - n_regions
        prob += u[i] >= 1
    # (11) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (12) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition

    # Solve the optimization problem
    solver = get_solver_instance(solver)
    prob.solve(solver)
    result = {}

    # build a list of regions like [[0, 1, 2, 5], [3, 4, 6, 7, 8]]
    idx_copy = set(I)
    regions = [[] for _ in range(n_regions)]
    for i in range(n_regions):
        area = idx_copy.pop()
        regions[i].append(area)

        for other_area in idx_copy:
            if t[area, other_area].varValue == 1:
                regions[i].append(other_area)

        idx_copy.difference_update(regions[i])
    for i in I:
        result[i] = find_sublist_containing(i, regions, index=True)
    return result
