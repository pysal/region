import libpysal as ps
from pandas import Series
from geopandas import GeoDataFrame
import numpy as np
import pulp
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum


def cluster_exact(areas, data, num_regions, method="flow", solver="cbc",
                  contiguity=None):
    """
    Parameters
    ----------
    areas : GeoDataFrame or dict
        In case of dict, each key represents an area and each value is a
        sequence of neighbors of this area.
    data : str, list or dict
        If `areas` is of type GeoDataFrame, the clustering criteria (columns)
        is specified as string (for one column) or list of strings (for
        multiple columns).
        If `areas` is of type dict, then `data` is also a
        dict with the same keys as `areas` and values representing the
        clustering criteria. A value can be a scalar or sequence of scalars.
    num_regions : int
        The number of regions the areas are clustered into.
    method : {"flow", "order", "tree"}, default: "flow"
        The method to translate the clustering problem into an exact
        optimization model.
        * "flow" - Flow model on p. 112-113 in [1]_
        * "order" - Order model on p. 110-112 in [1]_
        * "tree" - Tree model on p. 108-110 in [1]_

    solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
        The solver to use. Unless the default solver is used, the user has to
        make sure that the specified solver is installed.
        * "cbc" - the Cbc (Coin-or branch and cut) solver
        * "cplex" - the CPLEX solver
        * "glpk" - the GLPK (GNU Linear Programming Kit) solver
        * "gurobi" - the Gurobi Optimizer

    contiguity : None or str, default: None
        If `areas` is of type dict, this argument is ignored. If `areas` is of
        type GeoDataFrame, this argument defines the contiguity relationship
        between areas and will be treated as "rook" if not provided. Possible
        contiguity definitions are:
        * "rook" - Rook contiguity.
        * "queen" - Queen contiguity.

    Returns
    -------
    result : Series or dict
        If `areas` is of type GeoDataFrame, return a pandas.Series with the
        same index as the GeoDataFrame and specifying to which region (cluster)
        each area belongs.
        If `areas` is of type dict, a dict with the same keys is returned. The
        values of the dict specify to which region (cluster) each area belongs.

    References
    ----------
    .. [1] Duque, Church, Middleton (2011): "The p-Regions Problem"
    """
    if not isinstance(areas, (GeoDataFrame, dict)):
        raise ValueError("The areas argument must be one of the following"
                         "types: GeoDataFrame or dict.")

    if type(num_regions) is not int or num_regions <= 0:
        raise ValueError("The num_regions argument must be positive integer.")

    if not isinstance(method, str) \
            or method.lower() not in ["flow", "order", "tree"]:
        raise ValueError("The method argument must be one of the following"
                         'strings: "flow", "order", or "tree".')

    if not isinstance(solver, str) \
            or solver.lower() not in ["cbc", "cplex", "glpk", "gurobi"]:
        raise ValueError("The solver argument must be one of the following"
                         'strings: "cbc", "cplex", "glpk", or "gurobi".')

    if isinstance(areas, GeoDataFrame):
        if contiguity is not None:
            if not isinstance(contiguity, str) or \
                    contiguity.lower() not in ["rook", "queen"]:
                raise ValueError("The contiguity argument must be either None "
                                 "or one of the following strings: "
                                 '"rook" or"queen".')

        if contiguity is None or contiguity.lower() == "rook":
            weights = ps.weights.Contiguity.Rook.from_dataframe(areas)
        elif contiguity.lower() == "queen":
            weights = ps.weights.Contiguity.Queen.from_dataframe(areas)
        neighbors_dict = weights.neighbors

        if not isinstance(data, (str, list, tuple)):
            raise ValueError("In case the areas argument is of type "
                             "GeoDataFrame, data has to be of one of the "
                             "following types: str or list.")
        if isinstance(data, str):
            data = [data]
        else:
            data = list(data)
        values_dict = dict(zip(areas.index, np.array(areas[data])))

    elif isinstance(areas, dict):
        neighbors_dict = areas
        if not isinstance(data, dict) or data.keys() != areas.keys():
            raise ValueError("In case the areas argument is of type dict, "
                             "data has to be of type dict too with the keys "
                             "as areas.")
        values_dict = data

    opt_func = {"flow": _flow,
                "order": _order,
                "tree": _tree}[method.lower()]
    result_dict = opt_func(neighbors_dict, values_dict, num_regions,
                           solver)
    if isinstance(areas, GeoDataFrame):
        return Series(result_dict)
    elif isinstance(areas, dict):
        return result_dict


def _get_solver_instance(solver_string):
    solver = {"cbc": pulp.solvers.COIN_CMD,
              "cplex": pulp.solvers.CPLEX,
              "glpk": pulp.solvers.GLPK,
              "gurobi": pulp.solvers.GUROBI}[solver_string.lower()]
    return solver()


def _dissim_measure(v1, v2):
    """
    Parameters
    ----------
    v1 : float or ndarray
    v2 : float or ndarray

    Returns
    -------
    result : numpy.float64
        The dissimilarity between the values v1 and v2.
    """
    return np.linalg.norm(v1 - v2)


def _flow(neighbor_dict, value_dict, num_regions, solver):
    """
    Parameters
    ----------
    neighbor_dict : dict
        The keys represent the areas. The values represent the corresponding
        neighbors.
    value_dict : dict
        The keys represent the areas. The values are the corresponding values
        (e.g. of type float or ndarray).
    num_regions : int
        The number of regions the areas are clustered into.
    solver : {"cbc", "cplex", "glpk", "gurobi"}
        The solver to use for solving the mixed-integer program.

    Returns
    -------
    result : dict
        The keys represent the areas. Each value specifies the region an area
        has been assigned to.
    """
    print("running FLOW algorithm")  # TODO: rm
    prob = LpProblem("Flow", LpMinimize)

    # Parameters of the optimization problem
    n = len(value_dict)
    I = list(value_dict.keys())  # index for areas
    II = [(i, j) for i in I
                 for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    K = range(num_regions)  # index for regions
    d = {(i, j): _dissim_measure(value_dict[i], value_dict[j])
         for i, j in II_upper_triangle}

    # Decision variables
    t = LpVariable.dicts(
        "t",
        ((i, j) for i, j in II_upper_triangle),
        lowBound=0, upBound=1, cat=LpInteger)
    f = LpVariable.dicts(           # The amount of flow (non-negative integer)
        "f",                        # from area i to j in region k.
        ((i, j, k) for i, j in II for k in K),
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
                prob += f[i, j, k] <= y[i, k] * (n-num_regions)
    # (25) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbor_dict[i]:
            for k in K:
                prob += f[i, j, k] <= y[j, k] * (n-num_regions)
    # (26) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for k in K:
            lhs = sum(f[i, j, k] - f[j, i, k] for j in neighbor_dict[i])
            prob += lhs >= y[i, k] - (n-num_regions) * w[i, k]
    # (27) in Duque et al. (2011): "The p-Regions Problem"
    for i, j in II_upper_triangle:
        for k in K:
            prob += t[i, j] >= y[i, k] + y[j, k] - 1

    # Solve the optimization problem
    solver = _get_solver_instance(solver)
    prob.solve(solver)
    result = {}
    for i in I:
        for k in K:
            if y[i, k].varValue == 1:
                result[i] = k
    print(result)
    return result


def _order(neighbor_dict, value_dict, num_regions, solver):
    """
    Parameters
    ----------
    neighbor_dict : dict
        The keys represent the areas. The values represent the corresponding
        neighbors.
    value_dict : dict
        The keys represent the areas. The values are the corresponding values
        (e.g. of type float or ndarray).
    num_regions : int
        The number of regions the areas are clustered into.
    solver : {"cbc", "cplex", "glpk", "gurobi"}
        The solver to use for solving the mixed-integer program.

    Returns
    -------
    result : dict
        The keys represent the areas. Each value specifies the region an area
        has been assigned to.
    """
    print("running ORDER algorithm")  # TODO: rm
    prob = LpProblem("Order", LpMinimize)

    # Parameters of the optimization problem
    n = len(value_dict)
    I = list(value_dict.keys())  # index for areas
    II = [(i, j) for i in I
                 for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    K = range(num_regions)  # index for regions
    O = range(n - num_regions)  # index for orders
    d = {(i, j): _dissim_measure(value_dict[i], value_dict[j])
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

    # Solve the optimization problem
    solver = _get_solver_instance(solver)
    prob.solve(solver)
    result = {}
    for i in I:
        for k in K:
            for o in O:
                if x[i, k, o].varValue == 1:
                    result[i] = k
    return result


def _tree(neighbor_dict, value_dict, num_regions, solver):
    """
    Parameters
    ----------
    neighbor_dict : dict
        The keys represent the areas. The values represent the corresponding
        neighbors.
    value_dict : dict
        The keys represent the areas. The values are the corresponding values
        (e.g. of type float or ndarray).
    num_regions : int
        The number of regions the areas are clustered into.
    solver : {"cbc", "cplex", "glpk", "gurobi"}
        The solver to use for solving the mixed-integer program.

    Returns
    -------
    result : dict
        The keys represent the areas. Each value specifies the region an area
        has been assigned to.
    """
    print("running TREE algorithm")  # TODO: rm
    prob = LpProblem("Tree", LpMinimize)

    # Parameters of the optimization problem
    n = len(value_dict)
    I = list(value_dict.keys())
    II = [(i, j) for i in I
                 for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    d = {(i, j): _dissim_measure(value_dict[i], value_dict[j])
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
    prob += lhs == n - num_regions
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
            prob += u[i] - u[j] + (n-num_regions) * x[i, j] \
                    + (n-num_regions-2) * x[j, i] <= n - num_regions - 1
    # (10) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += u[i] <= n - num_regions
        prob += u[i] >= 1
    # (11) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (12) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition

    # Solve the optimization problem
    solver = _get_solver_instance(solver)
    prob.solve(solver)
    result = {}

    def find_index_of_sublist(el, lst):
        for idx, sublst in enumerate(lst):
            if el in sublst:
                return idx
        raise LookupError(
                "{} not found in any of the sublists of {}".format(el, lst))
    # build a list of regions like [[0, 1, 2, 5], [3, 4, 6, 7, 8]]
    idx_copy = set(I)
    regions = [[] for i in range(num_regions)]
    for i in range(num_regions):
        area = idx_copy.pop()
        regions[i].append(area)

        for other_area in idx_copy:
            if t[area, other_area].varValue == 1:
                regions[i].append(other_area)

        idx_copy.difference_update(regions[i])
    for i in I:
        result[i] = find_index_of_sublist(i, regions)
    return result

