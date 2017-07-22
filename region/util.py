import collections
import random

import numpy as np
import networkx as nx


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
        Each value is an `numpy.ndarray` holding the corresponding values in
        the columns specified by `cols`.

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


def distribute_regions_among_components(num_regions, nx_graph):
    """

    Parameters
    ----------
    num_regions : `int`
        The overall number of regions.
    nx_graph : `networkx.Graph`
        An undirected graph whose number of connected components is not greater
        than `num_regions`.

    Returns
    -------
    result : `dict`
        Each key (of type `networkx.Graph`) is a connected component of
        `nx_graph`.
        Each value is an `int` defining the number of regions in the key
        component.
    """
    comps = list(nx.connected_component_subgraphs(nx_graph, copy=False))
    num_regions_to_distribute = num_regions
    result = {}
    comps_multiplied = []
    # make sure each connected component has at least one region assigned to it
    for comp in comps:
        comps_multiplied += [comp] * (len(comp)-1)
        result[comp] = 1
        num_regions_to_distribute -= 1
    # distribute the rest of the regions to random components with bigger
    # components being likely to get more regions assigned to them
    while num_regions_to_distribute > 0:
        position = random.randrange(len(comps_multiplied))
        picked_comp = comps_multiplied.pop(position)
        result[picked_comp] += 1
        num_regions_to_distribute -= 1
    return result


def make_move(area, from_region, to_region, region_list_copy):
    print("  move", area, "from", from_region, "to", to_region)
    from_index = region_list_copy.index(from_region)
    to_index = region_list_copy.index(to_region)
    from_region.remove(area)
    to_region.add(area)
    region_list_copy[from_index] = from_region
    region_list_copy[to_index] = to_region


def objective_func(region_list, graph, attr="data"):
    return sum(dissim_measure(graph.node[list(region_list[r])[i]][attr],
                              graph.node[list(region_list[r])[j]][attr])
               for r in range(len(region_list))
               for i in range(len(region_list[r]))
               for j in range(len(region_list[r]))
               if i < j)