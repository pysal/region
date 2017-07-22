import random

import libpysal as ps
from geopandas import GeoDataFrame
import networkx as nx

from region.util import dataframe_to_dict, \
                        distribute_regions_among_components,\
                        find_sublist_containing, make_move, objective_func


def azp(areas, data, num_regions, contiguity=None, initial_sol=None):
    """

    Parameters
    ----------
    areas : GeoDataFrame

    data : str or list
        A string to select one column or a list of strings to select multiple
        columns.
    num_regions : int
        The number of regions the areas are clustered into.
    contiguity : {"rook", "queen"}
        This argument defines the contiguity relationship between areas.
    initial_sol : None or list, default: None
        If None, a starting solution will be computed.
        If `initial_sol` is a list then the starting solution of the algorithm
        will be determined by that list. An example of this list would be
        [0, 0, 0, 1, 1, 0, 1, 1, 1] for two regions.

    Returns
    -------
    result_dict : dict
        Each key is an area. Each value is the ID of the region (integer) an
        area belongs to.
    """
    num_areas = len(areas)

    if isinstance(data, str):
        data = [data]
    else:
        data = list(data)
    # todo: check if all elements of data correspond to a col in areas
    # todo: check if all elements of data are different

    if num_regions >= num_areas:
        raise ValueError("The num_regions argument must be "
                         "less than the number of areas.")
    if contiguity is None or contiguity.lower() == "rook":
        weights = ps.weights.Contiguity.Rook.from_dataframe(areas)
    elif contiguity.lower() == "queen":
        weights = ps.weights.Contiguity.Queen.from_dataframe(areas)
    else:
        raise ValueError("The contiguity argument must be one of the "
                         'following strings: "rook" or"queen".')
    weights.remap_ids(areas.index)
    graph = weights.to_networkx()
    nx.set_node_attributes(graph, "data", dataframe_to_dict(areas, data))

    if initial_sol is not None:
        # TODO: translate inial_sol to components with cut edges.
        raise NotImplementedError("initial_sol currently not handled.")
    else:
        print("step 1")
        num_comp = nx.number_connected_components(graph)
        if num_comp > num_regions:
            raise ValueError("The num_regions argument must not be less than "
                             "the number of connected components.")
        if num_comp > 1:
            num_regions_per_comp = distribute_regions_among_components(num_regions,
                                                                       graph)
        else:
            num_regions_per_comp = {graph.copy(): num_regions}

        # step 1: generate a random zoning system of num_regions regions
        #         from num_areas areas
        for comp, num_regions_in_comp in num_regions_per_comp.items():
            # cut edges until we have the desired number of regions in comp
            while nx.number_connected_components(comp) < num_regions_in_comp:
                num_edges = len(comp.edges())
                position = random.randrange(num_edges)
                edge_to_rm = comp.edges()[position]
                comp.remove_edge(*edge_to_rm)

    # do steps 2-7 for each component separately:
    region_list = []
    for comp, num_regions_in_comp in num_regions_per_comp.items():
        region_list_component = _azp_connected_component(
            graph, list(nx.connected_components(comp)), num_regions_in_comp)
        region_list += region_list_component
    result_dict = {}
    for n in graph.nodes():
        result_dict[n] = find_sublist_containing(n, region_list,
                                                 index=True)
    return result_dict


def _azp_connected_component(graph, initial_clustering, num_regions):
    """

    Parameters
    ----------
    graph : `networkx.Graph`
        A graph containing all areas in `initial_clustering` as nodes.
    initial_clustering : `list`
        Each list element is a `set` containing the areas of a region.
    num_regions : int
        The number of regions the areas in `initial_clustering` shall be
        divided into.
    """
    #  step 2: make a list of the M regions
    region_list = initial_clustering
    region_list_copy = region_list.copy()

    # todo: rm print-statements
    print("Init with: ", initial_clustering)
    obj_val_start = float("inf")  # since Python 3.5 math.inf is also possible
    obj_val_end = objective_func(region_list, graph)
    # step 7: Repeat until no further improving moves are made
    while obj_val_end < obj_val_start:  # improvement
        print("=" * 45)
        # print("step 7")
        obj_val_start = obj_val_end
        print("step 2")
        region_list = region_list_copy.copy()
        print("obj_value:", obj_val_end)
        print(region_list)
        # step 6: when the list for region K is exhausted return to step 3 and
        # select another region and repeat steps 4-6
        print("-" * 35)
        # print("step 6")
        while region_list:
            # step 3: select and remove any region K at random from this list
            print("step 3")
            random_position = random.randrange(len(region_list))
            region = region_list.pop(random_position)
            while True:
                # step 4: identify a set of zones bordering on members of
                # region K that could be moved into region K without
                # destroying the internal contiguity of the donor region(s)
                print("step 4")
                neighbors_of_region = [neigh for r in region
                                       for neigh in graph.neighbors(r)
                                       if neigh not in region]

                candidates = {}
                for neigh in neighbors_of_region:
                    region_index_of_neigh = find_sublist_containing(
                            neigh, region_list_copy, index=True)
                    region_of_neigh = region_list_copy[region_index_of_neigh]
                    try:
                        if nx.is_connected(
                                graph.subgraph(region_of_neigh - {neigh})):
                            candidates[neigh] = region_index_of_neigh
                    except nx.NetworkXPointlessConcept:
                        # if area is the only one in region than it has to stay
                        pass
                # step 5: randomly select zones from this list until either
                # there is a local improvement in the current value of the
                # objective function or a move that is equivalently as good
                # as the current best. Then make the move, update the list
                # of candidate zones, and return to step 4 or else repeat
                # step 5 until the list is exhausted.
                print("step 5")
                while candidates:
                    print("step 5 loop")
                    cand = random.choice(list(candidates))
                    region_of_cand = region_list_copy[candidates[cand]]
                    del candidates[cand]
                    # before move
                    obj_val = objective_func([region_of_cand, region], graph)
                    # after move
                    region_of_cand_after = region_of_cand.copy()
                    region_of_cand_after.remove(cand)
                    obj_val_after = objective_func(
                        [region_of_cand_after, region.union({cand})], graph)
                    if obj_val_after <= obj_val:
                        make_move(cand, region_of_cand, region,
                                  region_list_copy)
                        break
                else:
                    break

        obj_val_end = objective_func(region_list_copy, graph)
    return region_list_copy
