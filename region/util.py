import collections
import functools
import random
import types

import numpy as np
import networkx as nx
from sklearn.cluster.k_means_ import KMeans


def dataframe_to_dict(df, cols):
    """

    Parameters
    ----------
    df : `pandas.DataFrame` or `geopandas.GeoDataFrame`
    cols : list
        A list of strings. Each string is the name of a column of `df`.

    Returns
    -------
    result : dict
        The keys are the elements of the DataFrame's index.
        Each value is a :class:`numpy.ndarray` holding the corresponding values
        in the columns specified by `cols`.

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
    v1 : float or ndarray
    v2 : float or ndarray

    Returns
    -------
    result : numpy.float64
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
    print("  move", area,
          "  from", region_list[from_idx],
          "  to", region_list[to_idx])
    region_list[from_idx].remove(area)
    region_list[to_idx].add(area)


def objective_func(region_list, graph, attr="data"):
    return sum(dissim_measure(graph.node[list(region_list[r])[i]][attr],
                              graph.node[list(region_list[r])[j]][attr])
               for r in range(len(region_list))
               for i in range(len(region_list[r]))
               for j in range(len(region_list[r]))
               if i < j)


def generate_initial_sol(areas, graph, n_regions, random_state):
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
        if not feasible(comp_clustering, graph):
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


def regionalized_components(initial_sol, graph):
    """

    Parameters
    ----------
    initial_sol : dict
        If `initial_sol` is a dict then the each key must be an area and each
        value must be the corresponding region-ID in the initial clustering.
    graph : networkx.Graph
        The graph with areas as nodes and links between bordering areas.

    Yields
    ------
    comp : networkx Graph
        The yielded value represents a connected component of graph but with
        links removed between regions.
    """
    graph_copy = graph.copy()
    for comp in nx.connected_component_subgraphs(graph_copy):
        # cut edges between regions
        for n1, n2 in comp.edges():
            if initial_sol[n1] != initial_sol[n2]:
                comp.remove_edge(n1, n2)
        yield comp


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


def region_list_to_dict(region_list):
    """

    Parameters
    ----------
    region_list : `list`
        A list of sets. Each set consists of areas belonging to the same
        region. An example would be [{0, 1, 2, 5}, {3, 4, 6, 7, 8}].

    Returns
    -------
    result_dict : `dict`
        Each key is an area, each value is the corresponding region. An example
        would be {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1}.

    """
    result_dict = {}
    for region_idx, region in enumerate(region_list):
        for area in region:
            result_dict[area] = region_idx
    return result_dict


def dict_to_region_list(region_dict):
    """
    Inverse operation of `region_list_to_dict`.

    Parameters
    ----------
    region_dict : `dict`

    Returns
    -------
    region_list : `list`
        Each list element is a set of areas representing one region.
    """
    region_list = [set() for _ in region_dict.values()]
    for area in region_dict:
        region_list[region_dict[area]].add(area)
    region_list = [region for region in region_list if region]  # rm empty sets
    return region_list


def feasible(regions, graph, n_regions=None):
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
            raise ValueError("The number of regions is " + str(len(regions_list)) +
                             " but should be " + str(n_regions))
    for region in regions_list:
        if not nx.is_connected(graph.subgraph(region)):
            raise ValueError("Region " + str(region) + " is not spatially "
                             "contiguous.")


def separate_components(region_dict, graph):
    for comp in nx.connected_component_subgraphs(graph):
        yield {area: region_dict[area] for area in comp.nodes()}


def random_element_from(lst):
    random_position = random.randrange(len(lst))
    return lst[random_position]


def pop_randomly_from(lst):
    random_position = random.randrange(len(lst))
    return lst.pop(random_position)
