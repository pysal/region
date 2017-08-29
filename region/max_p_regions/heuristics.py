import random

import networkx as nx
import libpysal as ps
from scipy import sparse as sp

from region import fit_functions
from region.p_regions.azp import AZP
from region.p_regions.azp_util import AllowMoveAZPMaxPRegions
from region.util import find_sublist_containing, random_element_from,\
                        pop_randomly_from,array_from_dict_values,\
                        array_from_region_list, objective_func_arr, \
                        set_distance_metric, raise_distance_metric_not_set


class MaxPHeu:
    def __init__(self, local_search=None, random_state=None):
        """
        Class offering the implementation of the algorithm for solving the
        max-p-regions problem as described in [DAR2012]_.

        Parameters
        ----------
        local_search : None or :class:`AZP` or :class:`AZPSimulatedAnnealing`
            If None, then the AZP is used.
            Pass an instance of :class:`AZP` (or one of its subclasses) or
            :class:`AZPSimulatedAnnealing` to use a customized local search
            algorithm.
        random_state : None, int, str, bytes, or bytearray
            Random seed.
        """
        self.n_regions = None
        self.labels_ = None
        self.local_search = local_search
        self.random_state = random_state
        random.seed(random_state)
        self.distance_metric = raise_distance_metric_not_set

    def fit_from_dict(self, neighbor_dict, attr, spatially_extensive_attr,
                      threshold, max_it=10, distance_metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        neighbor_dict : `dict`
            Each key is an area and each value is an iterable of the key area's
            neighbors.
        attr : `dict`
            Each key is an area and each value is the corresponding attribute
            which serves as clustering criterion.
        spatially_extensive_attr :
            Each key is an area and each value is the corresponding spatial
            extensive attribute which is used to ensure that the sum of
            spatially extensive attributes in each region adds up to a
            threshold defined by the `threshold` argument.
        threshold : float
            Lower bound for the sum of `spatially_extensive_attr` within a
            region.
        max_it : int, default: 10
            The maximum number of partitions produced in the algorithm's
            construction phase.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        n_areas = len(neighbor_dict)
        adj = sp.dok_matrix((n_areas, n_areas))
        sorted_areas = sorted(neighbor_dict)
        for area in sorted_areas:
            for neighbor in neighbor_dict[area]:
                adj[area, neighbor] = 1
        adj = adj.tocsr()

        self.fit_from_scipy_sparse_matrix(adj,
                                          array_from_dict_values(attr,
                                                                 sorted_areas),
                                          spatially_extensive_attr, threshold,
                                          max_it=max_it,
                                          distance_metric=distance_metric)

    def fit_from_geodataframe(self, gdf, attr, spatially_extensive_attr,
                              threshold, max_it=10,
                              distance_metric="euclidean", contiguity="rook"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        gdf : :class:`geopandas.GeoDataFrame`

        attr : `str`
            A string to select a column of the :class:`geopandas.GeoDataFrame`
            `areas`. The selected data is used for calculating the objective
            function.
        spatially_extensive_attr : `str`
            A string to select a column of the :class:`geopandas.GeoDataFrame`
            `gdf`. The selected data is used to ensure that the spatially
            extensive attribute in each region adds up to a threshold defined
            by the `threshold` argument.
        threshold : float
            Lower bound for the sum of `spatially_extensive_attr` within a
            region.
        max_it : int, default: 10
            The maximum number of partitions produced in the algorithm's
            construction phase.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        contiguity : {"rook", "queen"}, default: "rook"
            See corresponding argument in
            :func:`region.fit_functions.fit_from_geodataframe`.
        """
        fit_functions.fit_from_geodataframe(self, gdf, attr,
                                            spatially_extensive_attr,
                                            threshold, max_it=max_it,
                                            distance_metric=distance_metric,
                                            contiguity=contiguity)

    def fit_from_networkx(self, graph, attr, spatially_extensive_attr,
                          threshold, max_it=10, distance_metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        graph : `networkx.Graph`
            Graph representing the contiguity relation.
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used for
            calculating the objective function.
        spatially_extensive_attr : :class:`numpy.ndarray`
            Each element specifies an area's spatially extensive attribute
            which is used to ensure that the sum of spatially extensive
            attributes in each region adds up to a threshold defined by the
            `threshold` argument.
        threshold : float
            Lower bound for the sum of `spatially_extensive_attr` within a
            region.
        max_it : int, default: 10
            The maximum number of partitions produced in the algorithm's
            construction phase.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        self.fit_from_scipy_sparse_matrix(adj, attr, spatially_extensive_attr,
                                          threshold, max_it=max_it,
                                          distance_metric=distance_metric)

    def fit_from_scipy_sparse_matrix(self, adj, attr, spatially_extensive_attr,
                                     threshold, max_it=10,
                                     distance_metric="euclidean"):
        """
        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Sparse matrix representing the contiguity relation.
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used for
            calculating the objective function.
        spatially_extensive_attr : :class:`numpy.ndarray`
            Each element specifies an area's spatially extensive attribute
            which is used to ensure that the sum of spatially extensive
            attributes in each region adds up to a threshold defined by the
            `threshold` argument.
        threshold : float
            Lower bound for the sum of `spatially_extensive_attr` within a
            region.
        max_it : int, default: 10
            The maximum number of partitions produced in the algorithm's
            construction phase.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        set_distance_metric(self, distance_metric)
        weights = ps.weights.weights.WSP(adj).to_W()
        areas_dict = weights.neighbors

        best_partition = None
        best_obj_value = float("inf")
        feasible_partitions = []
        partitions_before_enclaves_assignment = []
        max_p = 0  # maximum number of regions

        # construction phase
        print("constructing")
        for _ in range(max_it):
            print(" ", _)
            partition, enclaves = self.grow_regions(
                    adj, attr, spatially_extensive_attr, threshold)
            n_regions = len(partition)
            if n_regions > max_p:
                partitions_before_enclaves_assignment = [(partition, enclaves)]
                max_p = n_regions
            elif n_regions == max_p:
                partitions_before_enclaves_assignment.append((partition,
                                                              enclaves))

        print("\n" + "assigning enclaves")
        for partition, enclaves in partitions_before_enclaves_assignment:
            print("  cleaning up in partition", partition)
            feasible_partitions.append(self.assign_enclaves(
                    partition, enclaves, areas_dict, attr))

        # local search phase
        if self.local_search is None:
            self.local_search = AZP()
        self.local_search.allow_move_strategy = AllowMoveAZPMaxPRegions(
                adj, spatially_extensive_attr, threshold,
                self.local_search.allow_move_strategy)
        for partition in feasible_partitions:
            self.local_search.fit_from_scipy_sparse_matrix(
                    adj, attr, max_p,
                    initial_sol=array_from_region_list(partition))
            partition = self.local_search.labels_
            print("optimized partition", partition)
            obj_value = objective_func_arr(self.distance_metric, partition,
                                           attr)
            if obj_value < best_obj_value:
                best_obj_value = obj_value
                best_partition = partition
        self.labels_ = best_partition

    def fit_from_w(self, w, attr, spatially_extensive_attr, threshold,
                   max_it=10, distance_metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        w : :class:`libpysal.weights.weights.W`
            W object representing the contiguity relation.
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used for
            calculating the objective function.
        spatially_extensive_attr : :class:`numpy.ndarray`
            Each element specifies an area's spatially extensive attribute
            which is used to ensure that the sum of spatially extensive
            attributes in each region adds up to a threshold defined by the
            `threshold` argument.
        threshold : float
            Lower bound for the sum of `spatially_extensive_attr` within a
            region.
        max_it : int, default: 10
            The maximum number of partitions produced in the algorithm's
            construction phase.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        adj = w.sparse
        self.fit_from_scipy_sparse_matrix(adj, attr, spatially_extensive_attr,
                                          threshold, max_it=max_it,
                                          distance_metric=distance_metric)

    def grow_regions(self, adj, attr, spatially_extensive_attr, threshold):
        """
        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        spatially_extensive_attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        threshold : float
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

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
        unassigned_areas = list(range(adj.shape[0]))
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
                unassigned_neighs = set(adj[area].nonzero()[1]).difference(
                        assigned_areas)
                feasible = True
                spat_ext_attr = spatially_extensive_attr[area]
                while spat_ext_attr < threshold:
                    # print(" ", spat_ext_attr, "<", threshold, "Need neighs!")
                    # print("  potential neighbors:", unassigned_neighs)
                    if unassigned_neighs:
                        neigh = self.find_best_area(region, unassigned_neighs,
                                                    attr)
                        # print(" we choose neighbor", neigh)
                        region.add(neigh)
                        unassigned_neighs.remove(neigh)
                        unassigned_neighs.update(set(adj[neigh].nonzero()[1]))
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

    def find_best_area(self, region, candidates, attr):
        """

        Parameters
        ----------
        region : iterable
            Each element represents an area.
        candidates : iterable
            Each element represents an area bordering on region.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

        Returns
        -------
        best_area :
            An element of `candidates` with minimal dissimilarity when being
            moved to the region `region`.
        """
        candidates = {area: sum(self.distance_metric(attr[area], attr[area2])
                                for area2 in region)
                      for area in candidates}
        best_candidates = [area for area in candidates
                           if candidates[area] == min(candidates.values())]
        return random_element_from(best_candidates)

    def assign_enclaves(self, partition, enclave_areas, neigh_dict, attr):
        """
        Start with a partial partition (not all areas are assigned to a region)
        and a list of enclave areas (i.e. areas not present in the partial
        partition). Then assign all enclave areas to regions in the partial
        partition and return the resulting partition.

        Parameters
        ----------
        partition : `list`
            Each element (of type `set`) represents a region.
        enclave_areas : `list`
            Each element represents an area.
        neigh_dict : `dict`
            Each key represents an area. Each value is an iterable of the
            corresponding neighbors.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

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
            region_idx = self.find_best_region_idx(area, partition,
                                                   neigh_regions_idx, attr)
            partition[region_idx].add(area)
            enclave_areas.remove(area)
        return partition

    def find_best_region_idx(self, area, partition, candidate_regions_idx, attr):
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
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

        Returns
        -------
        best_idx : int
            The index of a region (w.r.t. `partition`) which has the smallest
            sum of dissimilarities after area `area` is moved to the region.
        """
        dissim_per_idx = {region_idx:
                          sum(self.distance_metric(attr[area], attr[area2])
                              for area2 in partition[region_idx])
                          for region_idx in candidate_regions_idx}
        minimum_dissim = min(dissim_per_idx.values())
        best_idxs = [idx for idx in dissim_per_idx
                     if dissim_per_idx[idx] == minimum_dissim]
        return random_element_from(best_idxs)
