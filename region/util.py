import collections
import functools
import itertools
import random
import types

import scipy.sparse.csgraph as cg
import numpy as np
import networkx as nx
from sklearn.cluster.k_means_ import KMeans


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


def dissim_measure(v1, v2):
    """
    Parameters
    ----------
    v1 : Union[`float`, :class:`ndarray`]
    v2 : Union[`float`, :class:`ndarray`]

    Returns
    -------
    result : `float`
        The dissimilarity between the values v1 and v2.
    """
    return np.linalg.norm(v1 - v2)


def distribute_regions_among_components(n_regions, graph):
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
    print("distribute_regions_among_components got a ", type(graph))
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


def make_move(area, from_idx, to_idx, region_list):
    """
    Modify the `region_list` argument in place (no return value!) such that the
    area `area` appears in the set with index `to_idx` instead of `from_idx`.
    This means that area `area` is moved to a new region.

    Parameters
    ----------
    area :
        The area to be moved (assigned to a new region).
    from_idx : `int`
        The index of `area`'s current region in the list `region_list`.
    to_idx : `int`
        The index of `area`'s new region in the list `region_list`.
    region_list : `list`
        List of sets where each set represents one region.

    Examples
    --------
    >>> region0 = {0, 1, 2}
    >>> region1 = {3}
    >>> region_list = [region0, region1]
    >>> make_move(2, from_idx=0, to_idx=1, region_list=region_list)
    >>> region_list == [{0, 1}, {2, 3}]
    True
    """
    # print("  move", area,
    #       "  from", region_list[from_idx],
    #       "  to", region_list[to_idx])
    region_list[from_idx].remove(area)
    region_list[to_idx].add(area)


def objective_func(region_list, graph, attr="data"):
    return sum(dissim_measure(graph.node[list(region_list[r])[i]][attr],
                              graph.node[list(region_list[r])[j]][attr])
               for r in range(len(region_list))
               for i in range(len(region_list[r]))
               for j in range(len(region_list[r]))
               if i < j)


def objective_func_dict(regions, attr):
    """

    Parameters
    ----------
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
    return objective_func_list(dict_to_region_list(regions), attr)


def objective_func_list(regions, attr):
    """

    Parameters
    ----------
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
    obj_val = sum(dissim_measure(attr[i], attr[j])
                  for r in regions
                  for i, j in itertools.combinations(r, 2))
    return obj_val


def generate_initial_sol(w, n_regions):
    """
    Generate a random initial clustering.

    Parameters
    ----------
    w : :class:`libpysal.weights.weights.W`

    n_regions : int

    Yields
    ------
    result : `dict`
        Each key must be an area and each value must be the corresponding
        region-ID. The dict's keys are a connected component of the graph
        provided to the function.
    """
    graph = w.to_networkx()
    n_regions_per_comp = distribute_regions_among_components(
            n_regions, graph)
    for comp, n_regions_in_comp in n_regions_per_comp.items():
        comp_adj = nx.to_scipy_sparse_matrix(comp)
        labels = randomly_divide_connected_graph(comp_adj, n_regions_in_comp)
        yield {area: region for area, region in zip(comp.nodes(), labels)}


def randomly_divide_connected_graph(adj, n_regions):
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
    labels : `list`
        Each element of the list specifies the region an area belongs to.

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> n_nodes = 10
    >>> adj_diagonal = [1] * (n_nodes-1)
    >>> # 10x10 adjacency matrix representing the path 0-1-2-...-9-10
    >>> adj = diags([adj_diagonal, adj_diagonal], offsets=[-1, 1])
    >>> n_regions_desired = 4
    >>> labels = randomly_divide_connected_graph(adj, n_regions_desired)
    >>> n_regions_obtained = len(set(labels))
    >>> n_regions_desired == n_regions_obtained
    True
    """
    if not n_regions > 0:
        msg = "n_regions is {} but must be positive.".format(n_regions)
        raise ValueError(msg)
    if not n_regions <= adj.shape[0]:
        msg = "n_regions is {} but must less than or equal to " + \
              "the number of nodes which is {}".format(n_regions, adj.shape[0])
        raise ValueError(msg)
    mst = cg.minimum_spanning_tree(adj)
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
            labels = cg.connected_components(mst_copy, directed=False)[1]
            max_size = max(np.unique(labels, return_counts=True)[1])
            print(max_size)
            if max_size < max_region_size:
                best_link = (i, j)
                max_region_size = max_size
        mst[best_link[0], best_link[1]] = 0
        mst.eliminate_zeros()
    return cg.connected_components(mst)[1]


def generate_initial_sol_kmeans(areas, graph, n_regions, random_state):
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
    n_regions_per_comp = distribute_regions_among_components(
            n_regions, graph)
    print("step 1")
    # step 1: generate a random zoning system of n_regions regions
    #         from num_areas areas
    print(n_regions_per_comp)
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
                    print("Region", region, "produced by K-Means disconnected")
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


def assert_feasible(regions, graph, n_regions=None):
    """

    Parameters
    ----------
    regions : `dict` or `list`
        If `dict`, then each key is an area and each value the corresponding
        region.
        If `list`, then each list element is a `set` of areas representing one
        region.
    graph : :class:`networkx.Graph`
        A :class:`networkx.Graph` representing areas as nodes. Bordering areas
        are connected by a an edge in the graph.
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
    if isinstance(regions, dict):
        regions_list = dict_to_region_list(regions)
    else:
        regions_list = regions

    if n_regions is not None:
        if len(regions_list) != n_regions:
            raise ValueError("The number of regions is " +
                             str(len(regions_list)) +
                             " but should be " + str(n_regions))
    for region in regions_list:
        if not nx.is_connected(graph.subgraph(region)):
            raise ValueError("Region " + str(region) + " is not spatially "
                             "contiguous.")


def separate_components(region_dict, graph):
    """

    Parameters
    ----------
    region_dict : `dict`

    graph : :class:`networkx.Graph`


    Yields
    ------
    comp_dict : `dict`
        Dictionary representing the clustering of a connected component in the
        graph passed as argument.

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
    >>>
    >>> # island 1: island divided into regions 0, 1, and 2
    >>> regions_dict = {area: area%3 for area in range(6)}
    >>> # island 2: all areas are in region 3
    >>> regions_dict.update({area: 3 for area in range(6, 10)})
    >>>
    >>> yielded = list(separate_components(regions_dict, graph))
    >>> yielded == [{0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2},
    ...             {8: 3, 9: 3, 6: 3, 7: 3}]
    True

    """
    for comp in nx.connected_component_subgraphs(graph):
        yield {area: region_dict[area] for area in comp.nodes()}


def random_element_from(lst):
    random_position = random.randrange(len(lst))
    return lst[random_position]


def pop_randomly_from(lst):
    random_position = random.randrange(len(lst))
    return lst.pop(random_position)


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
