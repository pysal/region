from collections import namedtuple

import libpysal as ps
from geopandas import GeoDataFrame
import networkx as nx

from region.util import dissim_measure, find_sublist_containing, \
    random_element_from, pop_randomly_from

Move = namedtuple("move", "area from_idx to_idx")


def max_p_regions(areas, attr, spatially_extensive_attr, threshold, max_it=10):
    """

    Parameters
    ----------
    areas : dict
        Each key represents an area. Each value is an iterable of the
        corresponding neighbors.
    attr : dict
        Each key represents an area. Each value is the corresponding clustering
        criterion (a `float` or `ndarray`).
    spatially_extensive_attr : dict
        Each key represents an area. Each value is the corresponding spatially
        extensive attribute (a `float` or `ndarray`).
    threshold : float
        Lower bound for the sum of `spatially_extensive_attr` within a region.
    max_it : int, default: 10
        The maximum number of partitions produced in the algorithm's
        construction phase.

    Returns
    -------
    region_dict : dict
        Each key represents an area. Each value represents the corresponding
        region.
    """
    d = {(a1, a2): dissim_measure(attr[a1], attr[a2])
         for a1 in areas for a2 in areas}

    best_partition = []
    het = float("inf")
    feasible_partitions = []
    partitions_before_enclaves_assignment = []
    max_p = 0  # maximum number of regions

    # construction phase
    print("constructing")
    for _ in range(max_it):
        print(" ", _)
        partition, unassigned_areas = grow_regions(
                areas, d, spatially_extensive_attr, threshold)
        n_regions = len(partition)
        if n_regions > max_p:
            partitions_before_enclaves_assignment = [partition]
            max_p = n_regions
        elif n_regions == max_p:
            partitions_before_enclaves_assignment.append(partition)

    print("assigning enclaves")
    for partition in partitions_before_enclaves_assignment:
        print("  cleaning up in partition", partition)
        feasible_partitions.append(assign_enclaves(partition, unassigned_areas,
                                                   areas, d))

    return feasible_partitions  # todo rm

    # local search phase
    # todo

    region_dict = {}
    #todo: make region_dict from best_partition
    return region_dict


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

    """
    partition = []
    enclave_areas = []  # todo: rm? (not really needed)
    unassigned_areas = list(neigh_dict.keys())
    assigned_areas = []

    # todo: rm prints
    while unassigned_areas:
        print("partition", partition)
        area = pop_randomly_from(unassigned_areas)
        print("seed in area", area)
        assigned_areas.append(area)
        if spatially_extensive_attr[area] >= threshold:
            print("  seed --> region :)")
            partition.append({area})
        else:
            region = {area}
            print("  all neighbors:", neigh_dict[area])
            print("  already assigned:", assigned_areas)
            unassigned_neighs = set(neigh_dict[area]).difference(
                    assigned_areas)
            feasible = True
            spat_ext_attr = spatially_extensive_attr[area]
            while spat_ext_attr < threshold:
                print(" ", spat_ext_attr, "<", threshold, "Need neighbors!")
                print("  potential neighbors:", unassigned_neighs)
                if unassigned_neighs:
                    neigh = find_best_area(region, unassigned_neighs,
                                           dissimilarities)
                    print(" we choose neighbor", neigh)
                    region.add(neigh)
                    unassigned_neighs.remove(neigh)
                    unassigned_neighs.update(neigh_dict[neigh])
                    unassigned_neighs.difference_update(assigned_areas)
                    spat_ext_attr += spatially_extensive_attr[neigh]
                    unassigned_areas.remove(neigh)
                    assigned_areas.append(neigh)
                else:
                    print("  Oh no! No neighbors left :(")
                    enclave_areas.extend(region)
                    feasible = False
                    # unassigned_areas.extend(region)  # todo: rm?
                    for area in region:
                        assigned_areas.remove(area)
                    break
            if feasible:
                partition.append(region)
            print("  unassigned:", unassigned_areas)
            print("  assigned:", assigned_areas)
            print()

    return partition, unassigned_areas


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


def assign_enclaves(partition, unassigned_areas, neigh_dict, dissimilarities):
    """

    Parameters
    ----------
    partition : list
        Each element (of type `set`) represents a region.
    unassigned_areas : list
        Each element represents an area.
    neigh_dict : dict
        Each key represents an area. Each value is an iterable of the
        corresponding neighbors.
    dissimilarities : dict
        Each key is a tuple of two areas. Each value is the dissimilarity
        between these two areas.

    Returns
    -------

    """
    while unassigned_areas:
        neighbors_of_assigned = [area for area in unassigned_areas
                                 if any(neigh not in unassigned_areas
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
        partition[region_idx].append(area)
        unassigned_areas.remove(area)
    return partition


def find_best_region_idx(area, partition, candidate_regions_idx,
                         dissimilarities):
    """

    Parameters
    ----------
    area :
        The area to be moved to one of the regions specified by
        `candidate_regions_idx`.
    partition : list
        Each element (of type `set`) represents a region.
    candidate_regions_idx : iterable
        Each element is the index of a region in the `partition` list.
    dissimilarities : dict
        Each key is a tuple of two areas. Each value is the dissimilarity
        between these two areas.

    Returns
    -------
    best_idx : int
        The index of a region (w.r.t. `partition`) which has the smallest sum
        of dissimilarities after area is moved to the region.

    """
    dissim_per_idx = {region_idx: sum(dissimilarities[area, area2]
                                      for area2 in partition[region_idx])
                      for region_idx in candidate_regions_idx}
    minimum_dissim = min(dissim_per_idx.values())
    best_idxs = [idx for idx in dissim_per_idx
                 if dissim_per_idx[idx] == minimum_dissim]
    return random_element_from(best_idxs)


# todo: rm following func (not needed)
def find_best_region(area, candidates, dissimilarities):
    """

    Parameters
    ----------
    area :
        The area to be moved.
    candidates : iterable
        Each element represents a region bordering on area `area`.
    dissimilarities : dict
        Each key is a tuple of two areas. Each value is the dissimilarity
        between these two areas.

    Returns
    -------
    best_region : iterable
        An element of `candidates` with minimal dissimilarity when being moved
        to the region `region`.
    """
    candidates = {region: sum(dissimilarities[area, area2] for area2 in region)
                  for region in candidates}
    best_candidates = [region for region in candidates
                       if candidates[region] == min(candidates.values())]
    return random_element_from(best_candidates)
