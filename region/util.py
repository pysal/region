import collections
import functools
import itertools
import random
import types

import scipy.sparse.csgraph as csg
import numpy as np
import networkx as nx
from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics.pairwise import distance_metrics


Move = collections.namedtuple("move", "area old_region new_region")


def array_from_dict_values(dct, sorted_keys=None, dtype=np.float):
    """
    Return values of the dictionary passed as `dct` argument as an numpy array.
    The values in the returned array are sorted by the keys of `dct`.

    Parameters
    ----------
    dct : dict

    sorted_keys : iterable, optional
        If passed, then the elements of the returned array will be sorted by
        this argument. Thus, this argument can be passed to suppress the
        sorting, or for getting a subset of the dictionary's values or to get
        repeated values.
    dtype : default: np.float64
        The `dtype` of the returned array.

    Returns
    -------
    array : :class:`numpy.ndarray`
    """
    if sorted_keys is None:
        sorted_keys = sorted(dct)
    return np.fromiter((dct[key] for key in sorted_keys),
                       dtype=dtype)


def array_from_region_list(region_list):  # todo: remove after refactoring (use sparse matrices and arrays instead of graphs and region_lists)
    return array_from_dict_values(region_list_to_dict(region_list))


def dataframe_to_dict(df, cols):
    """

    Parameters
    ----------
    df : Union[:class:`pandas.DataFrame`, :class:`geopandas.GeoDataFrame`]

    cols : Union[`str`,  `list`]
        If `str`, then it is the name of a column of `df`.
        If `list`, then it is a list of strings. Each string is the name of a
        column of `df`.

    Returns
    -------
    result : dict
        The keys are the elements of the DataFrame's index.
        Each value is a :class:`numpy.ndarray` holding the corresponding values
        in the columns specified by `cols`.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"data": [100, 120, 115]})
    >>> result = dataframe_to_dict(df, "data")
    >>> result == {0: 100, 1: 120, 2: 115}
    True
    >>> import numpy as np
    >>> df = pd.DataFrame({"data": [100, 120],
    ...                    "other": [1, 2]})
    >>> actual = dataframe_to_dict(df, ["data", "other"])
    >>> desired = {0: np.array([100, 1]), 1: np.array([120, 2])}
    >>> all(np.array_equal(actual[i], desired[i]) for i in desired)
    True
    """
    return dict(zip(df.index, np.array(df[cols])))


def find_sublist_containing(el, lst, index=False):
    """

    Parameters
    ----------
    el :
        The element to search for in the sublists of `lst`.
    lst : collections.Sequence
        A sequence of sequences or sets.
    index : bool, default: False
        If False (default), the subsequence or subset containing `el` is
        returned.
        If True, the index of the subsequence or subset in `lst` is returned.

    Returns
    -------
    result : collections.Sequence, collections.Set or int
        See the `index` argument for more information.

    Raises
    ------
    exc : LookupError
        If `el` is not in any of the elements of `lst`.

    Examples
    --------
    >>> lst = [{0, 1}, {2}]
    >>> find_sublist_containing(0, lst, index=False) == {0, 1}
    True
    >>> find_sublist_containing(0, lst, index=True) == 0
    True
    >>> find_sublist_containing(2, lst, index=False) == {2}
    True
    >>> find_sublist_containing(2, lst, index=True) == 1
    True
    """
    for idx, sublst in enumerate(lst):
        if el in sublst:
            return idx if index else sublst
    raise LookupError(
            "{} not found in any of the sublists of {}".format(el, lst))


def set_distance_metric(instance, metric="euclidean"):  # todo: move to classes (AZP, MaxPHeu,...) or to a new base class
    """
    Save the distance metric function specified by the `metric` argument as
    `distance_metric` attribute in the object passed as `instance` argument.

    Parameters
    ----------
    instance : object

    metric : str or function, default: "euclidean"
        If str, then this string specifies the distance metric (from
        scikit-learn) to use for calculating the objective function.
        Possible values are:

        * "cityblock" for sklearn.metrics.pairwise.manhattan_distances
        * "cosine" for sklearn.metrics.pairwise.cosine_distances
        * "euclidean" for sklearn.metrics.pairwise.euclidean_distances
        * "l1" for sklearn.metrics.pairwise.manhattan_distances
        * "l2" for sklearn.metrics.pairwise.euclidean_distances
        * "manhattan" for sklearn.metrics.pairwise.manhattan_distances

        If function, then this function should take two arguments and
        return a scalar value. Furthermore, the following conditions
        have to be fulfilled:

        1. d(a, b) >= 0, for all a and b
        2. d(a, b) == 0, if and only if a = b, positive definiteness
        3. d(a, b) == d(b, a), symmetry
        4. d(a, c) <= d(a, b) + d(b, c), the triangle inequality

    Examples
    --------
    >>> from region.p_regions.azp import AZP
    >>> from sklearn.metrics.pairwise import manhattan_distances
    >>> azp = AZP()
    >>> set_distance_metric(azp, "manhattan")
    >>> azp.distance_metric == manhattan_distances
    True
    """
    metric = get_distance_metric_function(metric)
    instance.distance_metric = metric


def get_distance_metric_function(metric="euclidean"):
    """

    Parameters
    ----------
    metric : str or function, default: "euclidean"
        If str, then this string specifies the distance metric (from
        scikit-learn) to use for calculating the objective function.
        Possible values are:

        * "cityblock" for sklearn.metrics.pairwise.manhattan_distances
        * "cosine" for sklearn.metrics.pairwise.cosine_distances
        * "euclidean" for sklearn.metrics.pairwise.euclidean_distances
        * "l1" for sklearn.metrics.pairwise.manhattan_distances
        * "l2" for sklearn.metrics.pairwise.euclidean_distances
        * "manhattan" for sklearn.metrics.pairwise.manhattan_distances

        If function, then this function should take two arguments and
        return a scalar value. Furthermore, the following conditions
        have to be fulfilled:

        1. d(a, b) >= 0, for all a and b
        2. d(a, b) == 0, if and only if a = b, positive definiteness
        3. d(a, b) == d(b, a), symmetry
        4. d(a, c) <= d(a, b) + d(b, c), the triangle inequality

    Returns
    -------
    If the `metric` argument is a function, it is returned.
    If the `metric` argument is a string, then the corresponding distance
    metric function from `sklearn.metrics.pairwise`.
    """
    if isinstance(metric, str):
        try:
            return distance_metrics()[metric]
        except KeyError:
            raise ValueError(
                "{} is not a known metric. Please use rather one of the "
                "following metrics: {}".format(tuple(name for name in
                                               distance_metrics().keys()
                                               if name != "precomputed")))
    elif callable(metric):
        return metric
    else:
        raise ValueError("A {} was passed as `metric` argument. "
                         "Please pass a string or a function "
                         "instead.".format(type(metric)))


class MissingMetric(RuntimeError):
    """Raised when a distance metric is required but was not set."""


def raise_distance_metric_not_set(x, y):
    raise MissingMetric("distance metric not set!")


def distribute_regions_among_components_nx(n_regions, graph):  # todo: rm if not needed
    """

    Parameters
    ----------
    n_regions : `int`
        The overall number of regions.
    graph : `networkx.Graph`
        An undirected graph whose number of connected components is not greater
        than `n_regions`.

    Returns
    -------
    result_dict : `dict`
        Each key (of type `networkx.Graph`) is a connected component of
        `graph`.
        Each value is an `int` defining the number of regions in the key
        component.
    """
    # print("distribute_regions_among_components_nx got a ", type(graph))
    if len(graph) < 1:
        raise ValueError("There must be at least one area.")
    if len(graph) < n_regions:
        raise ValueError("The number of regions must be "
                         "less than or equal to the number of areas.")
    comps = list(nx.connected_component_subgraphs(graph, copy=False))
    n_regions_to_distribute = n_regions
    result_dict = {}
    comps_multiplied = []
    # make sure each connected component has at least one region assigned to it
    for comp in comps:
        comps_multiplied += [comp] * (len(comp)-1)
        result_dict[comp] = 1
        n_regions_to_distribute -= 1
    # distribute the rest of the regions to random components with bigger
    # components being likely to get more regions assigned to them
    while n_regions_to_distribute > 0:
        position = random.randrange(len(comps_multiplied))
        picked_comp = comps_multiplied.pop(position)
        result_dict[picked_comp] += 1
        n_regions_to_distribute -= 1
    return result_dict


def make_move(moving_area, new_label, labels):
    """
    Modify the `labels` argument in place (no return value!) such that the
    area `moving_area` has the new region label `new_label`.

    Parameters
    ----------
    moving_area :
        The area to be moved (assigned to a new region).
    new_label : `int`
        The new region label of area `moving_area`.
    labels : :class:`numpy.ndarray`
        Each element is a region label of the area corresponding array index.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array([0, 0, 0, 0, 1, 1])
    >>> make_move(3, 1, labels)
    >>> (labels == np.array([0, 0, 0, 1, 1, 1])).all()
    True
    """
    labels[moving_area] = new_label


def objective_func(distance_metric, region_list, graph, attr="data"):
    return sum(distance_metric(graph.node[list(region_list[r])[i]][attr],
                               graph.node[list(region_list[r])[j]][attr])
               for r in range(len(region_list))
               for i in range(len(region_list[r]))
               for j in range(len(region_list[r]))
               if i < j)


def objective_func_arr(distance_metric, labels_arr, attr,
                       region_restriction=None):
    """
    Parameters
    ----------
    distance_metric : function
        A function taking two arguments and returning a scalar >= 0.
        Furthermore, the function must fulfill the properties described in the
        docstring of :meth:`get_distance_metric_function`.
    labels_arr : :class:`numpy.ndarray`
        Region labels.
    attr : :class:`numpy.ndarray`

    region_restriction : iterable
        Each element is a (distinct) region label. The calculation will be
        restricted to region labels present in this iterable.

    Returns
    -------
    obj_val : float
        The objective value attained with the clustering defined by
        `labels_arr`.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import distance_metrics
    >>> metric = distance_metrics()["manhattan"]
    >>> labels = np.array([0, 0, 0, 0, 1, 1])
    >>> attr = np.arange(len(labels))
    >>> int(objective_func_arr(metric, labels, attr))
    11
    >>> labels = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    >>> attr = np.arange(len(labels))
    >>> int(objective_func_arr(metric, labels, attr, region_restriction={0,1}))
    11
    """
    if region_restriction is not None:
        regions_set = set(region_restriction)
    else:
        regions_set = set(labels_arr)
    obj_val = sum(distance_metric(attr[i].reshape(1, -1),
                                  attr[j].reshape(1, -1))
                  for r in regions_set
                  for i, j in
                  itertools.combinations(np.where(labels_arr == r)[0], 2))
    return obj_val


def objective_func_diff(distance_metric, labels, attr, area, new_region):
    """
    Parameters
    ----------
    distance_metric : function
    labels : :class:`numpy.ndarray`
    attr : :class:`numpy.ndarray`
    area : int
    new_region : int

    Returns
    -------
    diff : tuple
        The tuple's first entry is the difference in the objective function
        in the donor region caused by removing area `area` from it.
        The tuple's second entry is the difference in the objective function
        in the recipient region caused by adding area `area` to it.
    """
    donor_region = labels[area]

    attr_donor = attr[labels == donor_region]
    donor_diff = sum(distance_metric(attr_donor,
                                     attr[area].reshape(1, -1)))

    attr_recipient = attr[labels == new_region]
    recipient_diff = sum(distance_metric(attr_recipient,
                                         attr[area].reshape(1, -1)))
    return -donor_diff, recipient_diff


def objective_func_dict(distance_metric, regions, attr):
    """
    Parameters
    ----------
    distance_metric : str or function, default: "euclidean"
        See the `metric` argument in :func:`region.util.set_distance_metric`.
    regions : `dict`
        Each key is an area. Each value is the region it is assigned to.
    attr : `dict`
        Each key is an area. Each value is the corresponding attribute.

    Returns
    -------
    obj_val : float
        The objective value is the total heterogeneity (sum of each region's
        heterogeneity).
    """
    return objective_func_list(distance_metric, dict_to_region_list(regions),
                               attr)


def objective_func_list(distance_metric, regions, attr):
    """
    Parameters
    ----------
    distance_metric : str or function, default: "euclidean"
        See the `metric` argument in :func:`region.util.set_distance_metric`.
    regions : `list`
        Each list element is an iterable of a region's areas.
    attr : `dict`
        Each key is an area. Each value is the corresponding attribute.

    Returns
    -------
    obj_val : float
        The objective value is the total heterogeneity (sum of each region's
        heterogeneity).
    """
    # print("regions in objective_func_list:", regions)
    obj_val = sum(distance_metric(attr[i], attr[j])
                  for r in regions
                  for i, j in itertools.combinations(r, 2))
    return obj_val


def distribute_regions_among_components(component_labels, n_regions):
    r"""

    Parameters
    ----------
    component_labels : list
        Each element specifies to which connected component an area belongs.
        An example would be [0, 0, 1, 0, 0, 1] for the following two islands:
        island one      island two
        .-------.         .---.
        | 0 | 1 |         | 2 |
        |-------|         |---|
        | 3 | 4 |         | 5 |
        `-------'         `---'

    n_regions : int

    Returns
    -------
    result_dict : Dict[int, int]
        Each key is a label of a connected component. Each value specifies into
        how many regions the component is to be clustered.
    """
    # copy list to avoid manipulating callers list instance
    component_labels = list(component_labels)
    n_regions_to_distribute = n_regions
    components = set(component_labels)
    if len(components) == 1:
        return {0: n_regions}
    result_dict = {}
    # make sure each connected component has at least one region assigned to it
    for comp in components:
        component_labels.remove(comp)
        result_dict[comp] = 1
        n_regions_to_distribute -= 1
    # distribute the rest of the regions to random components with bigger
    # components being likely to get more regions assigned to them
    while n_regions_to_distribute > 0:
        position = random.randrange(len(component_labels))
        picked_comp = component_labels.pop(position)
        result_dict[picked_comp] += 1
        n_regions_to_distribute -= 1
    return result_dict


def generate_initial_sol(adj, n_regions):
    """
    Generate a random initial clustering.

    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`

    n_regions : int

    Yields
    ------
    region_labels : :class:`numpy.ndarray`
        An array with -1 for areas which are not part of the yielded
        component and an integer >= 0 specifying the region of areas within the
        yielded component.
    """
    # check args
    n_areas = adj.shape[0]
    if n_areas == 0:
        raise ValueError("There must be at least one area.")
    if n_areas < n_regions:
        raise ValueError("The number of regions ({}) must be "
                         "less than or equal to the number of areas "
                         "({}).".format(n_regions, n_areas))
    if n_regions == 1:
        yield {area: 0 for area in range(n_areas)}
        return

    n_comps, comp_labels = csg.connected_components(adj)
    if n_comps > n_regions:
            raise ValueError("The number of regions ({}) must not be "
                             "less than the number of connected components "
                             "({}).".format(n_regions, n_comps))
    n_regions_per_comp = distribute_regions_among_components(comp_labels,
                                                             n_regions)

    print("n_regions_per_comp", n_regions_per_comp)
    regions_built = 0
    for comp_label, n_regions_in_comp in n_regions_per_comp.items():
        print("comp_label", comp_label)
        print("n_regions_in_comp", n_regions_in_comp)
        region_labels = -np.ones(len(comp_labels), dtype=np.int32)
        in_comp = comp_labels == comp_label
        comp_adj = adj[in_comp]
        comp_adj = comp_adj[:, in_comp]
        region_labels_comp = _randomly_divide_connected_graph(
                comp_adj, n_regions_in_comp) + regions_built
        regions_built += n_regions_in_comp
        print("Regions in comp:", set(region_labels_comp))
        region_labels[in_comp] = region_labels_comp
        yield region_labels


def _randomly_divide_connected_graph(adj, n_regions):
    """
    Divide the provided connected graph into `n_regions` regions.

    Parameters
    ----------
    csgraph : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix.
    n_regions : int
        The desired number of clusters. Must be > 0 and <= number of nodes.

    Returns
    -------
    labels : :class:`numpy.ndarray`
        Each element (an integer in {0, ..., `n_regions` - 1}) specifies the
        region an area (defined by the index in the array) belongs to.

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> n_nodes = 10
    >>> adj_diagonal = [1] * (n_nodes-1)
    >>> # 10x10 adjacency matrix representing the path 0-1-2-...-9-10
    >>> adj = diags([adj_diagonal, adj_diagonal], offsets=[-1, 1])
    >>> n_regions_desired = 4
    >>> labels = _randomly_divide_connected_graph(adj, n_regions_desired)
    >>> n_regions_obtained = len(set(labels))
    >>> n_regions_desired == n_regions_obtained
    True
    """
    if not n_regions > 0:
        msg = "n_regions is {} but must be positive.".format(n_regions)
        raise ValueError(msg)
    n_areas = adj.shape[0]
    if not n_regions <= n_areas:
        msg = "n_regions is {} but must less than or equal to " + \
              "the number of nodes which is {}".format(n_regions, n_areas)
        raise ValueError(msg)
    mst = csg.minimum_spanning_tree(adj)
    for _ in range(n_regions - 1):
        # try different links to cut and pick the one leading to the most
        # balanced solution
        best_link = None
        max_region_size = float("inf")
        for __ in range(5):
            mst_copy = mst.copy()
            nonzero_i, nonzero_j = mst_copy.nonzero()
            random_position = random.randrange(len(nonzero_i))
            i, j = nonzero_i[random_position], nonzero_j[random_position]
            mst_copy[i, j] = 0
            mst_copy.eliminate_zeros()
            labels = csg.connected_components(mst_copy, directed=False)[1]
            max_size = max(np.unique(labels, return_counts=True)[1])
            if max_size < max_region_size:
                best_link = (i, j)
                max_region_size = max_size
        mst[best_link[0], best_link[1]] = 0
        mst.eliminate_zeros()
    return csg.connected_components(mst)[1]


def generate_initial_sol_kmeans(areas, graph, n_regions, random_state):  # todo: rm if not needed
    """

    Parameters
    ----------
    areas : :class:`geopandas.GeoDataFrame`

    graph : networkx.Graph

    n_regions : int
        Number of regions to divide the graph into.
    random_state : int or None
        Random seed for the K-Means algorithm.

    Yields
    ------
    result : `dict`
        Each key must be an area and each value must be the corresponding
        region-ID. The dict's keys are a connected component of the graph
        provided to the function.
    """
    n_regions_per_comp = distribute_regions_among_components_nx(
            n_regions, graph)
    # print("step 1")
    # step 1: generate a random zoning system of n_regions regions
    #         from num_areas areas
    # print(n_regions_per_comp)
    comp_clusterings_dicts = []
    for comp, n_regions_in_comp in n_regions_per_comp.items():
        comp_gdf = areas[areas.index.isin(comp.nodes())]
        polys = comp_gdf["geometry"]
        cents = polys.centroid
        geometry_arr = np.array([[cent.x, cent.y]
                                 for cent in cents])
        k_means = KMeans(n_regions_in_comp, random_state=random_state)
        comp_clustering = {area: region for area, region in zip(
                           comp.nodes(), k_means.fit(geometry_arr).labels_)}
        # check feasibility because K-Means can produce solutions violating the
        # spatial contiguity condition.
        try:
            assert_feasible(comp_clustering, graph)
        except ValueError:
            regions_list = dict_to_region_list(comp_clustering)
            for region in regions_list:
                region_graph = comp.subgraph(region)
                if not nx.is_connected(region_graph):
                    # print("Region", region, "produced by K-Means disconnected")
                    parts = list(nx.connected_components(region_graph))
                    parts.sort(key=len)
                    # assign region's smallest parts to neighboring regions
                    for part in parts[:-1]:
                        # find neighboring region
                        neighs = [neigh
                                  for area in part
                                  for neigh in nx.neighbors(graph, area)
                                  if neigh not in part]
                        neigh = pop_randomly_from(neighs)
                        neigh_region = comp_clustering[neigh]
                        # move part to neighboring region
                        for area in part:
                            comp_clustering[area] = neigh_region
        comp_clusterings_dicts.append(comp_clustering)
    return (c for c in comp_clusterings_dicts)


def copy_func(f):
    """
    Return a copy of a function. This is useful e.g. to create aliases (whose
    docstrings can be changed without affecting the original function).
    The implementation is taken from https://stackoverflow.com/a/13503277.
    """
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def assert_feasible(solution, adj, n_regions=None):
    """

    Parameters
    ----------
    solution : :class:`numpy.ndarray`
        Array of region labels.
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix representing the contiguity relation.
    n_regions : `int` or `None`
        An `int` represents the desired number of regions.
        If `None`, then the number of regions is not checked.

    Raises
    ------
    exc : `ValueError`
        A `ValueError` is raised if clustering is not spatially contiguous.
        Given the `n_regions` argument is not `None`, a `ValueError` is raised
        also if the number of regions is not equal to the `n_regions` argument.
    """
    if n_regions is not None:
        if len(set(solution)) != n_regions:
            raise ValueError("The number of regions is {} but "
                             "should be {}".format(len(solution), n_regions))
    for region_label in set(solution):
        _, comp_labels = csg.connected_components(adj)
        # check whether equal region_label implies equal comp_label
        comp_labels_in_region = comp_labels[solution == region_label]
        if not all_elements_equal(comp_labels_in_region):
            raise ValueError("Region {} is not spatially "
                             "contiguous.".format(region_label))


def all_elements_equal(array):
    return np.max(array) == np.min(array)


def separate_components(adj, solution):
    """

    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix representing the contiguity relation.
    solution : :class:`numpy.ndarray`

    Yields
    ------
    comp_dict : :class:`numpy.ndarray`
        Each yielded dict represents one connected component of the graph
        specified by the `adj` argument. In a yielded dict, each key is an area
        and each value is the corresponding region-ID.

    Examples
    --------
    >>> edges_island1 = [(0, 1), (1, 2),          # 0 | 1 | 2
    ...                  (0, 3), (1, 4), (2, 5),  # ---------
    ...                  (3, 4), (4,5)]           # 3 | 4 | 5
    >>>
    >>> edges_island2 = [(6, 7),                  # 6 | 7
    ...                  (6, 8), (7, 9),          # -----
    ...                  (8, 9)]                  # 8 | 9
    >>>
    >>> graph = nx.Graph(edges_island1 + edges_island2)
    >>> adj = nx.to_scipy_sparse_matrix(graph)
    >>>
    >>> # island 1: island divided into regions 0, 1, and 2
    >>> sol_island1 = [area%3 for area in range(6)]
    >>> # island 2: all areas are in region 3
    >>> sol_island2 = [3 for area in range(6, 10)]
    >>> solution = np.array(sol_island1 + sol_island2)
    >>>
    >>> yielded = list(separate_components(adj, solution))
    >>> yielded.sort(key=lambda arr: arr[0], reverse=True)
    >>> (yielded[0] == np.array([0, 1, 2, 0, 1, 2, -1, -1, -1, -1])).all()
    True
    >>> (yielded[1] == np.array([-1, -1, -1, -1, -1, -1, 3, 3, 3, 3])).all()
    True
    """
    n_comps, comp_labels = csg.connected_components(adj)
    for comp in set(comp_labels):
        region_labels = -np.ones(len(comp_labels), dtype=np.int32)
        in_comp = comp_labels == comp
        region_labels[in_comp] = solution[in_comp]
        yield region_labels


def random_element_from(lst):
    random_position = random.randrange(len(lst))
    return lst[random_position]


def pop_randomly_from(lst):
    random_position = random.randrange(len(lst))
    return lst.pop(random_position)


def count(arr, el):
    """
    Parameters
    ----------
    arr : :class:`numpy.ndarray`

    el : object

    Returns
    -------
    result : :class:`numpy.ndarray`
        The number of occurences of `el` in `arr`.

    Examples
    --------
    >>> arr = np.array([0, 0, 0, 1, 1])
    >>> count(arr, 0)
    3
    >>> count(arr, 1)
    2
    >>> count(arr, 2)
    0
    """
    unique, counts = np.unique(arr, return_counts=True)
    idx = np.where(unique == el)[0]
    if len(idx) > 0:
        return int(counts[idx])
    return 0


def region_list_to_dict(region_list):
    """

    Parameters
    ----------
    region_list : `list`
        Each list element is an iterable of a region's areas.

    Returns
    -------
    result_dict : `dict`
        Each key is an area, each value is the corresponding region.

    Examples
    --------
    >>> result_dict = region_list_to_dict([{0, 1, 2, 5}, {3, 4, 6, 7, 8}])
    >>> result_dict == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1}
    True

    """
    result_dict = {}
    for region_idx, region in enumerate(region_list):
        for area in region:
            result_dict[area] = region_idx
    return result_dict


def dict_to_region_list(region_dict):
    """
    Inverse operation of :func:`region_list_to_dict`.

    Parameters
    ----------
    region_dict : dict

    Returns
    -------
    region_list : `list`

    Examples
    --------
    >>> region_list = dict_to_region_list({0: 0, 1: 0, 2: 0,
    ...                                    3: 1, 4: 1, 5: 0,
    ...                                    6: 1, 7: 1, 8: 1})
    >>> region_list == [{0, 1, 2, 5}, {3, 4, 6, 7, 8}]
    True
    """
    region_list = [set() for _ in range(max(region_dict.values()) + 1)]
    for area in region_dict:
        region_list[region_dict[area]].add(area)
    region_list = [region for region in region_list if region]  # rm empty sets
    return region_list
