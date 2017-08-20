from collections import namedtuple

import libpysal as ps

from region.p_regions.azp import AZP
from region.p_regions.azp_util import AllowMoveAZPMaxPRegions
from region.util import dissim_measure, find_sublist_containing, \
    random_element_from, pop_randomly_from, objective_func_dict, \
    dataframe_to_dict, region_list_to_dict

Move = namedtuple("move", "area from_idx to_idx")


def max_p_regions(areas, attr, spatially_extensive_attr, threshold, max_it=10,
                  local_search=None, contiguity="rook"):
    """

    Parameters
    ----------
    areas : :class:`geopandas.GeoDataFrame`
        See corresponding argument in :meth:`region.azp.AZP.fit`.
    attr : str
        A string to select a column of the :class:`geopandas.GeoDataFrame`
        `areas`. The selected data is used for calculating the objective
        function.
    spatially_extensive_attr : str
        A string to select a column of the :class:`geopandas.GeoDataFrame`
        `areas`. The selected data is used to ensure that the spatially
        extensive attribute in each region adds up to a threshold defined by
        the `threshold` argument.
    threshold : float
        Lower bound for the sum of `spatially_extensive_attr` within a region.
    max_it : int, default: 10
        The maximum number of partitions produced in the algorithm's
        construction phase.
    local_search : Union[:class:`AZP`, :class:`AZPSimulatedAnnealing`, `None`]
        Algorithm used in the local search phase.
    contiguity : {"rook", "queen"}, default: "rook"
        See corresponding argument in
        :func:`region.fit_functions.fit_from_geodataframe`.

    Returns
    -------
    region_dict : dict
        Each key represents an area. Each value represents the corresponding
        region.
    """
    if contiguity == "rook":
        weights = ps.weights.Contiguity.Rook.from_dataframe(areas)
    elif contiguity == "queen":
        weights = ps.weights.Contiguity.Queen.from_dataframe(areas)
    else:
        raise ValueError("The contiguity argument must be either "
                         '"rook" or "queen".')
    areas_dict = weights.neighbors
    attr_dict = dataframe_to_dict(areas, attr)
    spatially_extensive_attr_dict = dataframe_to_dict(areas,
                                                      spatially_extensive_attr)
    d = {(a1, a2): dissim_measure(attr_dict[a1], attr_dict[a2])
         for a1 in areas_dict for a2 in areas_dict}

    best_partition = None
    best_obj_value = float("inf")
    feasible_partitions = []
    partitions_before_enclaves_assignment = []
    max_p = 0  # maximum number of regions

    # construction phase
    print("constructing")
    for _ in range(max_it):
        print(" ", _)
        partition, enclaves = grow_regions(
                areas_dict, d, spatially_extensive_attr_dict, threshold)
        n_regions = len(partition)
        if n_regions > max_p:
            partitions_before_enclaves_assignment = [(partition, enclaves)]
            max_p = n_regions
        elif n_regions == max_p:
            partitions_before_enclaves_assignment.append((partition, enclaves))

    print("\n" + "assigning enclaves")
    for partition, enclaves in partitions_before_enclaves_assignment:
        print("  cleaning up in partition", partition)
        feasible_partitions.append(assign_enclaves(partition, enclaves,
                                                   areas_dict, d))
    # local search phase
    if local_search is None:
        local_search = AZP()
    local_search.allow_move_strategy = AllowMoveAZPMaxPRegions(
            areas, spatially_extensive_attr, threshold,
            local_search.allow_move_strategy)
    for partition in feasible_partitions:
        partition = local_search.fit(
                areas, attr, max_p, initial_sol=region_list_to_dict(partition))
        obj_value = objective_func_dict(partition, attr_dict)
        if obj_value < best_obj_value:
            best_obj_value = obj_value
            best_partition = partition
    return best_partition


def grow_regions(neigh_dict, dissimilarities, spatially_extensive_attr,
                 threshold):
    """

    Parameters
    ----------
    neigh_dict : dict
        Each key represents an area. Each value is an iterable of the
        corresponding neighbors.
    dissimilarities : dict
        Each key is a tuple of two areas. Each value is the dissimilarity
        between these two areas.
    spatially_extensive_attr
    threshold

    Returns
    -------
    result : `tuple`
        `result[0]` is a `list`. Each list element is a `set` of a region's
        areas. Note that not every area is assigned to a region after this
        function has terminated, so they won't be in any of the `set`s in
        `result[0]`.
        `result[1]` is a `list` of areas not assigned to any region.
    """
    partition = []
    enclave_areas = []
    unassigned_areas = list(neigh_dict.keys())
    assigned_areas = []

    # todo: rm prints
    while unassigned_areas:
        # print("partition", partition)
        area = pop_randomly_from(unassigned_areas)
        # print("seed in area", area)
        assigned_areas.append(area)
        if spatially_extensive_attr[area] >= threshold:
            # print("  seed --> region :)")
            partition.append({area})
        else:
            region = {area}
            # print("  all neighbors:", neigh_dict[area])
            # print("  already assigned:", assigned_areas)
            unassigned_neighs = set(neigh_dict[area]).difference(
                    assigned_areas)
            feasible = True
            spat_ext_attr = spatially_extensive_attr[area]
            while spat_ext_attr < threshold:
                # print(" ", spat_ext_attr, "<", threshold, "Need neighbors!")
                # print("  potential neighbors:", unassigned_neighs)
                if unassigned_neighs:
                    neigh = find_best_area(region, unassigned_neighs,
                                           dissimilarities)
                    # print(" we choose neighbor", neigh)
                    region.add(neigh)
                    unassigned_neighs.remove(neigh)
                    unassigned_neighs.update(neigh_dict[neigh])
                    unassigned_neighs.difference_update(assigned_areas)
                    spat_ext_attr += spatially_extensive_attr[neigh]
                    unassigned_areas.remove(neigh)
                    assigned_areas.append(neigh)
                else:
                    # print("  Oh no! No neighbors left :(")
                    enclave_areas.extend(region)
                    feasible = False
                    # the following line (present in the algorithm in
                    # [DAR2012]) is commented out because it leads to an
                    # infinite loop:
                    # unassigned_areas.extend(region)
                    for area in region:
                        assigned_areas.remove(area)
                    break
            if feasible:
                partition.append(region)
            # print("  unassigned:", unassigned_areas)
            # print("  assigned:", assigned_areas)
            # print()
    print("grow_regions produced", partition, "- enclaves:", enclave_areas)
    return partition, enclave_areas


def find_best_area(region, candidates, dissimilarities):
    """

    Parameters
    ----------
    region : iterable
        Each element represents an area.
    candidates : iterable
        Each element represents an area bordering on region.
    dissimilarities : dict
        Each key is a tuple of two areas. Each value is the dissimilarity
        between these two areas.

    Returns
    -------
    best_area :
        An element of `candidates` with minimal dissimilarity when being moved
        to the region `region`.
    """
    candidates = {area: sum(dissimilarities[area, area] for area in region)
                  for area in candidates}
    best_candidates = [area for area in candidates
                       if candidates[area] == min(candidates.values())]
    return random_element_from(best_candidates)


def assign_enclaves(partition, enclave_areas, neigh_dict, dissimilarities):
    """
    Start with a partial partition (not all areas are assigned to a region) and
    a list of enclave areas (i.e. areas not present in the partial partition).
    Then assign all enclave areas to regions in the partial partition and
    return the resulting partition.

    Parameters
    ----------
    partition : `list`
        Each element (of type `set`) represents a region.
    enclave_areas : `list`
        Each element represents an area.
    neigh_dict : `dict`
        Each key represents an area. Each value is an iterable of the
        corresponding neighbors.
    dissimilarities : `dict`
        Each key is a tuple of two areas. Each value is the dissimilarity
        between these two areas.

    Returns
    -------
    partition : `list`
        Each element (of type `set`) represents a region.
    """
    print("partition:", partition, "- enclaves:", enclave_areas)
    while enclave_areas:
        neighbors_of_assigned = [area for area in enclave_areas
                                 if any(neigh not in enclave_areas
                                        for neigh in neigh_dict[area])]
        area = pop_randomly_from(neighbors_of_assigned)
        neigh_regions_idx = []
        for neigh in neigh_dict[area]:
            try:
                neigh_regions_idx.append(
                        find_sublist_containing(neigh, partition, index=True))
            except LookupError:
                pass
        region_idx = find_best_region_idx(area, partition, neigh_regions_idx,
                                          dissimilarities)
        partition[region_idx].add(area)
        enclave_areas.remove(area)
    return partition


def find_best_region_idx(area, partition, candidate_regions_idx,
                         dissimilarities):
    """

    Parameters
    ----------
    area :
        The area to be moved to one of the regions specified by
        `candidate_regions_idx`.
    partition : `list`
        Each element (of type `set`) represents a region.
    candidate_regions_idx : iterable
        Each element is the index of a region in the `partition` list.
    dissimilarities : `dict`
        Each key is a tuple of two areas. Each value is the dissimilarity
        between these two areas.

    Returns
    -------
    best_idx : int
        The index of a region (w.r.t. `partition`) which has the smallest sum
        of dissimilarities after area `area` is moved to the region.
    """
    dissim_per_idx = {region_idx: sum(dissimilarities[area, area2]
                                      for area2 in partition[region_idx])
                      for region_idx in candidate_regions_idx}
    minimum_dissim = min(dissim_per_idx.values())
    best_idxs = [idx for idx in dissim_per_idx
                 if dissim_per_idx[idx] == minimum_dissim]
    return random_element_from(best_idxs)
