import abc
from collections import deque
import math
import random
from functools import reduce

import numpy as np
import networkx as nx
from scipy import sparse as sp

from region import fit_functions
from region.csgraph_utils import sub_adj_matrix, neighbors, is_connected
from region.p_regions.azp_util import AllowMoveStrategy, \
                                            AllowMoveAZP,\
                                            AllowMoveAZPSimulatedAnnealing
from region.util import Move, make_move, assert_feasible, separate_components,\
                        generate_initial_sol, copy_func, \
                        array_from_dict_values, set_distance_metric, \
                        objective_func_arr, pop_randomly_from, count,\
                        get_distance_metric_function, random_element_from


class AZP:
    def __init__(self, allow_move_strategy=None, random_state=None):
        """
        Class offering the implementation of the AZP algorithm (see [OR1995]_).

        Parameters
        ----------
        allow_move_strategy : None or :class:`AllowMoveStrategy`
            If None, then the AZP algorithm in [1]_ is chosen.
            For a different behavior for allowing moves an AllowMoveStrategy
            instance can be passed as argument.
        random_state : None, int, str, bytes, or bytearray
            Random seed.
        """
        self.n_regions = None
        self.labels_ = None
        self.random_state = random_state
        random.seed(self.random_state)

        wrong_allow_move_arg_msg = "The allow_move_strategy argument must " \
                                   "be either None, or an instance of " \
                                   "AllowMoveStrategy."
        correct_strategy = isinstance(allow_move_strategy, AllowMoveStrategy)
        if allow_move_strategy is None or correct_strategy:
            self.allow_move_strategy = allow_move_strategy
        else:
            raise ValueError(wrong_allow_move_arg_msg)

        self.distance_metric = None

    def fit_from_scipy_sparse_matrix(self, adj, data, n_regions,
                                     initial_sol=None,
                                     distance_metric="euclidean"):
        """
        Perform the AZP algorithm as described in [OR1995]_ and assign the
        resulting region labels to the instance's :attr:`labels_` attribute.

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Adjacency matrix representing the contiguity relation.
        data : :class:`numpy.ndarray`
            Array of data according to which the clustering is performed.
        n_regions : `int`
            Number of desired regions.
        initial_sol : :class:`numpy.ndarray`
            Array of labels.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        metric = get_distance_metric_function(distance_metric)
        if self.allow_move_strategy is None:
            self.allow_move_strategy = AllowMoveAZP(attr=data, metric=metric)
        set_distance_metric(self, metric)
        # step 1
        if initial_sol is not None:
            assert_feasible(initial_sol, adj, n_regions)
            initial_sol_gen = separate_components(adj, initial_sol)
        else:
            initial_sol_gen = generate_initial_sol(adj, n_regions)
        region_labels = -np.ones(adj.shape[0])
        for labels_component in initial_sol_gen:
            in_comp_idx = np.where(labels_component != -1)[0]
            # print("Clustering component ", in_comp_idx)
            labels_component = self._azp_connected_component(
                    adj, labels_component, data, in_comp_idx)
            region_labels[in_comp_idx] = labels_component

        self.n_regions = n_regions
        self.labels_ = region_labels

    fit = copy_func(fit_from_scipy_sparse_matrix)
    fit.__doc__ = "Alias for :meth:`fit_from_scipy_sparse_matrix`.\n\n" \
                  + fit_from_scipy_sparse_matrix.__doc__

    def fit_from_w(self, w, data, n_regions, initial_sol=None,
                   distance_metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        w : :class:`libpysal.weights.weights.W`
            W object representing the contiguity relation.
        data : :class:`numpy.ndarray`
            Array of data according to which the clustering is performed.
        n_regions : `int`
            Number of desired regions.
        initial_sol : :class:`numpy.ndarray`
            Array of labels.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        adj = w.sparse
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol,
                                          distance_metric=distance_metric)

    def fit_from_networkx(self, graph, data, n_regions, initial_sol=None,
                          distance_metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        graph : `networkx.Graph`
            Graph representing the contiguity relation.
        data : :class:`numpy.ndarray`
            Array of data according to which the clustering is performed.
        n_regions : `int`
            Number of desired regions.
        initial_sol : :class:`numpy.ndarray`
            Array of labels.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol)

    def fit_from_geodataframe(self, gdf, data, n_regions, contiguity="rook",
                              initial_sol=None, distance_metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        gdf : :class:`geopandas.GeoDataFrame`

        data : `str` or `list`
            The clustering criteria (columns of the GeoDataFrame `areas`) are
            specified as string (for one column) or list of strings (for
            multiple columns).
        n_regions : `int`
            Number of desired regions.
        contiguity : `str`
            See the corresponding argument in
            :func:`region.fit_functions.fit_from_geodataframe`.
        initial_sol : :class:`numpy.ndarray`
            Array of labels.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        fit_functions.fit_from_geodataframe(self, gdf, data, n_regions,
                                            contiguity=contiguity,
                                            initial_sol=initial_sol,
                                            distance_metric=distance_metric)

    def fit_from_dict(self, neighbor_dict, data, n_regions, initial_sol=None,
                      distance_metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix:.

        Parameters
        ----------
        neighbor_dict : `dict`
            Each key is an area and each value is an iterable of the key area's
            neighbors.
        data : `dict`
            Each key is an area and each value is the corresponding attribute
            which serves as clustering criterion
        n_regions : `int`
            Number of desired regions.
        initial_sol : `dict`
            Each key represents an area. Each value represents the region, the
            corresponding area is assigned to initially.
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

        if initial_sol is not None:
            initial_sol = array_from_dict_values(initial_sol, sorted_areas,
                                                 dtype=np.int32)
        self.fit_from_scipy_sparse_matrix(adj,
                                          array_from_dict_values(data,
                                                                 sorted_areas),
                                          n_regions, initial_sol,
                                          distance_metric=distance_metric)

    def _azp_connected_component(self, adj, initial_clustering, data,
                                 comp_idx):
        """
        Implementation of the AZP algorithm for a spatially connected set of
        areas (i.e. for every area there is a path to every other area).

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Adjacency matrix representing the contiguity relation. The matrix'
            shape is (N, N) where N denotes the number of *all* areas (not only
            those that are in a connected component).
        initial_clustering : :class:`numpy.ndarray`
            Array of labels. The array's shape is (N) where N denotes the
            number of *all* areas (not only those that are in a connected
            component).
        data : :class:`numpy.ndarray`
            Clustering criterion. The array's shape is (N) where N denotes the
            number of *all* areas (not only those that are in a connected
            component).
        comp_idx : :class:`numpy.ndarray`
            Indices of all areas belonging to a connected component of the
            graph represented by the adjacency matrix `adj`. Only those areas
            specified by this argument are considered when the method is
            executed.

        Returns
        -------
        labels_copy : `list`
            Each element is an iterable of areas representing a region.
        """
        # if there is only one region in the initial solution, just return it.
        distinct_regions = list(np.unique(initial_clustering[comp_idx]))
        if len(distinct_regions) == 1:
            return initial_clustering
        distinct_regions_copy = distinct_regions.copy()

        adj = sub_adj_matrix(adj, comp_idx)
        print("comp_adj.shape:", adj.shape)
        initial_clustering = initial_clustering[comp_idx]
        print("initial_clustering", initial_clustering)
        data = data[comp_idx]
        print("data", data)
        self.allow_move_strategy.set_comp_idx(comp_idx)
        #  step 2: make a list of the M regions
        labels = initial_clustering

        # print("Init with: ", initial_clustering)
        obj_val_start = float("inf")  # since Python 3.5 math.inf also possible
        obj_val_end = objective_func_arr(self.distance_metric, labels, data)
        print("start with obj. val.:", obj_val_end)
        # step 7: Repeat until no further improving moves are made
        while obj_val_end < obj_val_start:  # improvement
            # print("obj_val:", obj_val_start, "-->", obj_val_end,
            #       "...continue...")
            # print("=" * 45)
            # print("step 7")
            obj_val_start = obj_val_end
            print("step 2")
            distinct_regions = distinct_regions_copy.copy()
            # step 6: when the list for region K is exhausted return to step 3
            # and select another region and repeat steps 4-6
            # print("-" * 35)
            # print("step 6")
            while distinct_regions:
                # step 3: select & remove any region K at random from this list
                print("step 3")
                region = pop_randomly_from(distinct_regions)
                print("  chosen region:", region)
                while True:
                    # step 4: identify a set of zones bordering on members of
                    # region K that could be moved into region K without
                    # destroying the internal contiguity of the donor region(s)
                    print("step 4")
                    region_areas = np.where(labels == region)[0]
                    # print("region consists of areas", region_areas)
                    # print("adj", adj.todense())
                    neighbors_of_region = reduce(
                            np.union1d,
                            (neighbors(adj, area) for area in region_areas))
                    neighbors_of_region = np.setdiff1d(neighbors_of_region,
                                                       region_areas)
                    # print("  neighbors of region", region, "are:")
                    # print(neighbors_of_region)
                    candidates = []
                    for neigh in neighbors_of_region:
                        neigh_region = labels[neigh]
                        # print("  labels:", labels)
                        # print("  We could move area {} from {} to {}".format(neigh, neigh_region, region))
                        # print("  adj before subbing:\n", adj.todense())
                        sub_adj = sub_adj_matrix(
                                adj,
                                np.where(labels == neigh_region)[0],
                                wo_nodes=neigh)
                        # print("  submatrix:\n", sub_adj.todense())
                        if is_connected(sub_adj):
                            # if area is alone in its region, it must stay
                            if count(labels, neigh_region) > 1:
                                candidates.append(neigh)
                    # step 5: randomly select zones from this list until either
                    # there is a local improvement in the current value of the
                    # objective function or a move that is equivalently as good
                    # as the current best. Then make the move, update the list
                    # of candidate zones, and return to step 4 or else repeat
                    # step 5 until the list is exhausted.
                    print("step 5")
                    while candidates:
                        print("step 5 loop")
                        cand = pop_randomly_from(candidates)
                        if self.allow_move_strategy(cand, region, labels):
                            print("  MOVING {} from {} to {}".format(cand, labels[cand], region))
                            make_move(cand, region, labels)
                            # print("new labels:\n", labels, sep="")
                            # print("new obj. val.:", objective_func_arr(self.distance_metric, labels, data))
                            break
                    else:
                        break

            obj_val_end = objective_func_arr(self.distance_metric, labels,
                                             data)
        return labels


class AZPSimulatedAnnealing:
    def __init__(self, init_temperature=None,
                 max_iterations=float("inf"), min_sa_moves=0,
                 nonmoving_steps_before_stop=3,
                 repetitions_before_termination=5, random_state=None):

        self.allow_move_strategy = None
        self.azp = None

        if init_temperature is not None:
            self.init_temperature = init_temperature
        else:
            raise NotImplementedError("TODO")  # todo

        self.labels_ = None

        self.maxit = max_iterations
        self.min_sa_moves = min_sa_moves

        self.min_sa_moves_reached = False
        self.move_made = False
        self.nonmoving_steps_before_stop = nonmoving_steps_before_stop

        self.visited = []
        self.reps_before_termination = repetitions_before_termination

        self.random_state = random_state

    def fit_from_geodataframe(self, gdf, data, n_regions,
                              contiguity="rook",initial_sol=None,
                              cooling_factor=0.85,
                              distance_metric="euclidean"):
        """
        Parameters
        ----------
        gdf : :class:`geopandas.GeoDataFrame`
            See the corresponding argument in
            :func:`AZP.fit_from_geodataframe`.
        data : `str` or `list`
            See the corresponding argument in
            :func:`AZP.fit_from_geodataframe`.
        n_regions : `int`
            See the corresponding argument in
            :func:`AZP.fit_from_geodataframe`.
        contiguity : `str`
            See the corresponding argument in
            :func:`AZP.fit_from_geodataframe`.
        initial_sol : :class:`numpy.ndarray`
            See the corresponding argument in
            :func:`AZP.fit_from_geodataframe`.
        cooling_factor : float
            Float :math:`\\in (0, 1)` specifying the cooling factor for the
            simulated annealing.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        fit_functions.fit_from_geodataframe(
                self, gdf, data, n_regions, contiguity=contiguity,
                initial_sol=initial_sol, cooling_factor=cooling_factor,
                distance_metric=distance_metric)

    def fit_from_dict(self, neighbor_dict, data, n_regions, initial_sol=None,
                      cooling_factor=0.85, distance_metric="euclidean"):
        """
        Parameters
        ----------
        neighbor_dict : `dict`
            See the corresponding argument in :func:`AZP.fit_from_dict`.
        data : `dict`
            See the corresponding argument in :func:`AZP.fit_from_dict`.
        n_regions : `int`
            See the corresponding argument in :func:`AZP.fit_from_dict`.
        initial_sol : `dict`
            See the corresponding argument in :func:`AZP.fit_from_dict`.
        cooling_factor : float
            Float :math:`\\in (0, 1)` specifying the cooling factor for the
            simulated annealing.
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

        if initial_sol is not None:
            initial_sol = array_from_dict_values(initial_sol, sorted_areas,
                                                 dtype=np.int32)
        self.fit_from_scipy_sparse_matrix(
                adj,
                array_from_dict_values(data, sorted_areas),
                n_regions,
                initial_sol=initial_sol,
                cooling_factor=cooling_factor,
                distance_metric=distance_metric)

    def fit_from_networkx(self, graph, data, n_regions, initial_sol=None,
                          cooling_factor=0.85, distance_metric="euclidean"):
        """
        Parameters
        ----------
        graph : `networkx.Graph`
            See the corresponding argument in :func:`AZP.fit_from_networkx`.
        data : :class:`numpy.ndarray`
            See the corresponding argument in :func:`AZP.fit_from_networkx`.
        n_regions : `int`
            See the corresponding argument in :func:`AZP.fit_from_networkx`.
        initial_sol : :class:`numpy.ndarray`
            See the corresponding argument in :func:`AZP.fit_from_networkx`.
        cooling_factor : float
            Float :math:`\\in (0, 1)` specifying the cooling factor for the
            simulated annealing.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol,
                                          cooling_factor=cooling_factor,
                                          distance_metric=distance_metric)

    def fit_from_scipy_sparse_matrix(self, adj, data, n_regions,
                                     initial_sol=None, cooling_factor=0.85,
                                     distance_metric="euclidean"):
        """
        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            See the corresponding argument in
            :func:`AZP.fit_from_scipy_sparse_matrix`.
        data : :class:`numpy.ndarray`
            See the corresponding argument in
            :func:`AZP.fit_from_scipy_sparse_matrix`.
        n_regions : `int`
            See the corresponding argument in
            :func:`AZP.fit_from_scipy_sparse_matrix`.
        initial_sol : :class:`numpy.ndarray`
            See the corresponding argument in
            :func:`AZP.fit_from_scipy_sparse_matrix`.
        cooling_factor : float
            Float :math:`\\in (0, 1)` specifying the cooling factor for the
            simulated annealing.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        if not (0 < cooling_factor < 1):
            raise ValueError("The cooling_factor argument must be greater "
                             "than 0 and less than 1")
        metric = get_distance_metric_function(distance_metric)
        self.allow_move_strategy = AllowMoveAZPSimulatedAnnealing(
                attr=data, metric=metric,
                init_temperature=self.init_temperature,
                min_sa_moves=self.min_sa_moves)
        self.allow_move_strategy.register_min_sa_moves(self.sa_moves_alert)
        self.allow_move_strategy.register_move_made(self.move_made_alert)

        self.azp = AZP(allow_move_strategy=self.allow_move_strategy,
                       random_state=self.random_state)
        # todo: rm print() calls
        # step a
        # print(("#"*60 + "\n") * 5 + "STEP A")
        t = self.init_temperature
        nonmoving_steps = 0
        # step d: repeat step b and c
        while nonmoving_steps < self.nonmoving_steps_before_stop:
            # print(("#"*60 + "\n") * 2 + "STEP B")
            it = 0
            self.min_sa_moves_reached = False
            # step b
            while it < self.maxit and not self.min_sa_moves_reached:
                it += 1
                old_sol = initial_sol
                self.azp.fit_from_scipy_sparse_matrix(adj, data, n_regions,
                                                      initial_sol, metric)
                initial_sol = self.azp.labels_

                # print("old_sol", old_sol)
                # print("new_sol", initial_sol)
                if old_sol is not None:
                    # print("EQUAL" if (old_sol == initial_sol).all()
                    #       else "NOT EQUAL")
                    if (old_sol == initial_sol).all():
                        # print("BREAK")
                        break
            # print("visited", self.visited)
            # added termination condition (not in Openshaw & Rao (1995))
            # print(initial_sol)
            if self.visited.count(tuple(initial_sol)) >= self.reps_before_termination:
                # print("VISITED", initial_sol, "FOR",
                #       self.reps_before_termination,
                #       "TIMES --> TERMINATING.")
                break
            self.visited.append(tuple(initial_sol))
            # step c
            # print(("#"*60 + "\n") * 2 + "STEP C")
            t *= cooling_factor
            self.allow_move_strategy.update_temperature(t)

            if self.move_made:
                # print("MOVE MADE")
                self.move_made = False
            else:
                # print("NO MOVE MADE")
                nonmoving_steps += 1
        self.labels_ = initial_sol

    def fit_from_w(self, w, data, n_regions, initial_sol=None,
                   cooling_factor=0.85, distance_metric="euclidean"):
        """
        Parameters
        ----------
        w : :class:`libpysal.weights.weights.W`
            See the corresponding argument in :func:`AZP.fit_from_w`.
        data : :class:`numpy.ndarray`
            See the corresponding argument in :func:`AZP.fit_from_w`.
        n_regions : `int`
            See the corresponding argument in :func:`AZP.fit_from_w`.
        initial_sol : :class:`numpy.ndarray`
            See the corresponding argument in :func:`AZP.fit_from_w`.
        cooling_factor : float
            Float :math:`\\in (0, 1)` specifying the cooling factor for the
            simulated annealing.
        distance_metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        adj = w.sparse
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol,
                                          cooling_factor=cooling_factor,
                                          distance_metric=distance_metric)

    def sa_moves_alert(self):
        self.min_sa_moves_reached = True

    def move_made_alert(self):
        self.move_made = True


class AZPTabu(AZP, abc.ABC):
    def _make_move(self, area, new_region, labels):
        old_region = labels[area]
        make_move(area, new_region, labels)
        # step 5: Tabu the reverse move for R iterations.
        reverse_move = Move(area, new_region, old_region)
        self.tabu.append(reverse_move)

    def _objval_diff(self, area, new_region, labels, data):
        old_region = labels[area]
        # before move
        objval_before = objective_func_arr(
                self.distance_metric, labels, data, [old_region, new_region])
        # after move
        labels[area] = new_region
        objval_after = objective_func_arr(
                self.distance_metric, labels, data, [old_region, new_region])
        labels[area] = old_region
        return objval_after - objval_before

    def reset_tabu(self, tabu_len=None):
        tabu_len = self.tabu.maxlen if tabu_len is None else tabu_len
        self.tabu = deque([], tabu_len)


class AZPBasicTabu(AZPTabu):
    def __init__(self, tabu_length=None,
                 repetitions_before_termination=5, random_state=None):
        self.tabu = deque([], tabu_length)
        self.visited = []
        self.reps_before_termination = repetitions_before_termination
        super().__init__(random_state=random_state)

    def _azp_connected_component(self, adj, initial_clustering, data,
                                 comp_idx):
        self.reset_tabu()
        # if there is only one region in the initial solution, just return it.
        distinct_regions = list(np.unique(initial_clustering[comp_idx]))
        if len(distinct_regions) == 1:
            return initial_clustering

        adj = sub_adj_matrix(adj, comp_idx)
        print("comp_adj.shape:", adj.shape)
        initial_clustering = initial_clustering[comp_idx]
        print("initial_clustering", initial_clustering)
        data = data[comp_idx]
        print("data", data)
        self.allow_move_strategy.set_comp_idx(comp_idx)

        #  step 2: make a list of the M regions
        labels = initial_clustering

        # todo: rm print-statements
        # print("Init with: ", initial_clustering)
        visited = []
        stop = False
        while True:  # TODO: condition??
            # print("visited", visited)
            # added termination condition (not in Openshaw & Rao (1995))
            label_tup = tuple(labels)
            if visited.count(label_tup) >= self.reps_before_termination:
                stop = True
                # print("VISITED", label_tup, "FOR",
                #       self.reps_before_termination,
                #       "TIMES --> TERMINATING BEFORE NEXT NON-IMPROVING MOVE")
            visited.append(label_tup)
            # print("=" * 45)
            # print("obj_value:", obj_val_end)
            # print(region_list)
            # print("-" * 35)
            # step 1 Find the global best move that is not prohibited or tabu.
            # print("step 1")
            # find possible moves (globally)
            best_move = None
            best_objval_diff = float("inf")
            for area in range(labels.shape[0]):
                old_region = labels[area]
                sub_adj = sub_adj_matrix(
                            adj,
                            np.where(labels == old_region)[0],
                            wo_nodes=area)
                # moving the area must not destroy spatial contiguity in donor
                # region and if area is alone in its region, it must stay:
                if is_connected(sub_adj) and count(labels, old_region) > 1:
                    for neigh in neighbors(adj, area):
                        new_region = labels[neigh]
                        if new_region != old_region:
                            possible_move = Move(area, old_region, new_region)
                            if possible_move not in self.tabu:
                                objval_diff = self._objval_diff(
                                        possible_move.area,
                                        possible_move.new_region, labels, data)
                                if objval_diff < best_objval_diff:
                                    best_move = possible_move
                                    best_objval_diff = objval_diff
            # print("  best move:", best_move, "objval_diff:", best_objval_diff)
            # step 2: Make this move if it is an improvement or equivalet in
            # value.
            print("step 2")
            if best_move is not None and best_objval_diff <= 0:
                print(labels)
                print("IMPROVING MOVE")
                self._make_move(best_move.area, best_move.new_region, labels)
            else:
                # step 3: if no improving move can be made, then see if a tabu
                # move can be made which improves on the current local best
                # (termed an aspiration move)
                print("step 3")
                print("Tabu:", self.tabu)
                improving_tabus = [
                    move for move in self.tabu
                    if labels[move.area] == move.old_region and
                    self._objval_diff(move.area, move.new_region,
                                      labels, data) < 0
                ]
                print(labels)
                if improving_tabus:
                    aspiration_move = random_element_from(improving_tabus)
                    # print("ASPIRATION MOVE")
                    self._make_move(aspiration_move.area,
                                    aspiration_move.new_region, labels)
                else:
                    # step 4: If there is no improving move and no aspirational
                    # move, then make the best move even if it is nonimproving
                    # (that is, results in a worse value of the objective
                    # function).
                    print("step 4")
                    print("No improving, no aspiration ==> make the best move")
                    if stop:
                        break
                    if best_move is not None:
                        self._make_move(best_move.area, best_move.new_region,
                                        labels)
        return labels
    _azp_connected_component.__doc__ = AZP._azp_connected_component.__doc__


class AZPReactiveTabu(AZPTabu):
    def __init__(self, max_iterations, k1, k2, random_state=None):
        self.tabu = deque([], maxlen=1)
        super().__init__(random_state=random_state)
        self.avg_it_until_rep = 1
        self.rep_counter = 1
        self.maxit = max_iterations
        self.visited = []
        self.k1 = k1
        self.k2 = k2

    def _azp_connected_component(self, adj, initial_clustering, data,
                                 comp_idx):
        self.reset_tabu(1)
        # if there is only one region in the initial solution, just return it.
        distinct_regions = list(np.unique(initial_clustering[comp_idx]))
        if len(distinct_regions) == 1:
            return initial_clustering

        adj = sub_adj_matrix(adj, comp_idx)
        print("comp_adj.shape:", adj.shape)
        initial_clustering = initial_clustering[comp_idx]
        print("initial_clustering", initial_clustering)
        data = data[comp_idx]
        print("data", data)
        self.allow_move_strategy.set_comp_idx(comp_idx)

        #  step 2: make a list of the M regions
        labels = initial_clustering

        # todo: rm print-statements
        # print("Init with: ", initial_clustering)
        it_since_tabu_len_changed = 0
        obj_val_start = float("inf")
        # step 12: Repeat steps 3-11 until either no further improvements are
        # made or a maximum number of iterations are exceeded.
        for it in range(self.maxit):
            # print("=" * 45)
            # print(region_list)
            obj_val_end = objective_func_arr(self.distance_metric, labels,
                                             data)
            # print("obj_value:", obj_val_end)
            if not obj_val_end < obj_val_start:
                break  # step 12
            obj_val_start = obj_val_end

            it_since_tabu_len_changed += 1
            # print("-" * 35)
            # step 3: Define the list of all possible moves that are not tabu
            # and retain regional connectivity.
            # print("step 3")
            possible_moves = []
            for area in range(labels.shape[0]):
                old_region = labels[area]
                sub_adj = sub_adj_matrix(
                            adj,
                            np.where(labels == old_region)[0],
                            wo_nodes=area)
                # moving the area must not destroy spatial contiguity in donor
                # region and if area is alone in its region, it must stay:
                if is_connected(sub_adj) and count(labels, old_region) > 1:
                    for neigh in neighbors(adj, area):
                        new_region = labels[neigh]
                        if new_region != old_region:
                            possible_move = Move(area, old_region, new_region)
                            if possible_move not in self.tabu:
                                possible_moves.append(possible_move)
            # step 4: Find the best nontabu move.
            # print("step 4")
            best_move = None
            best_move_index = None
            best_objval_diff = float("inf")
            for i, move in enumerate(possible_moves):
                obj_val_diff = self._objval_diff(
                        move.area, move.new_region, labels, data)
                if obj_val_diff < best_objval_diff:
                    best_move_index, best_move = i, move
                    best_objval_diff = obj_val_diff
            # print("  best move:", best_move)
            # step 5: Make the move. Update the tabu status.
            # print("step 5: make", best_move)
            self._make_move(best_move.area, best_move.new_region, labels)
            # step 6: Look up the current zoning system in a list of all zoning
            # systems visited so far during the search. If not found then go
            # to step 10.
            # print("step 6")
            # Sets can't be permuted so we convert our list to a set:
            label_tup = tuple(labels)
            if label_tup in self.visited:
                # step 7: If it is found and it has been visited more than K1
                # times already and this cyclical behavior has been found on
                # at least K2 other occasions (involving other zones) then go
                # to step 11.
                # print("step 7")
                # print("  labels", labels)
                # print("  self.visited:", self.visited)
                times_visited = self.visited.count(label_tup)
                cycle = list(reversed(self.visited))
                cycle = cycle[:cycle.index(label_tup) + 1]
                cycle = list(reversed(cycle))
                # print("  cycle:", cycle)
                it_until_repetition = len(cycle)
                if times_visited > self.k1:
                    if self.k2 > 0:
                        times_cycle_found = 0
                        for i in range(len(self.visited) - len(cycle)):
                            if self.visited[i:i+len(cycle)] == cycle:
                                times_cycle_found += 1
                                if times_cycle_found >= self.k2:
                                    break
                    if times_cycle_found >= self.k2:
                        # step 11: Delete all stored zoning systems and make P
                        # random moves, P = 1 + self.avg_it_until_rep/2, and
                        # update tabu to preclude a return to the previous
                        # state.
                        # print("step 11")
                        # we save the labels such that we can access it if
                        # this step yields a poor solution.
                        last_step = (11, tuple(labels))
                        self.visited = []
                        p = math.floor(1 + self.avg_it_until_rep/2)
                        possible_moves.pop(best_move_index)
                        for _ in range(p):
                            move = possible_moves.pop(
                                    random.randrange(len(possible_moves)))
                            self._make_move(move.area, move.new_region,
                                            labels)
                        continue
                    # step 8: Update a moving average of the repetition
                    # interval self.avg_it_until_rep, and increase the
                    # prohibition period R to 1.1*R.
                    # print("step 8")
                    self.rep_counter += 1
                    avg_it = self.avg_it_until_rep
                    self.avg_it_until_rep = 1 / self.rep_counter * \
                        ((self.rep_counter-1)*avg_it + it_until_repetition)

                    self.tabu = deque(self.tabu, 1.1*self.tabu.maxlen)
                    # step 9: If the number of iterations since R was last
                    # changed exceeds self.avg_it_until_rep, then decrease R to
                    # max(0.9*R, 1).
                    # print("step 9")
                    if it_since_tabu_len_changed > self.avg_it_until_rep:
                        new_tabu_len = max([0.9*self.tabu.maxlen, 1])
                        new_tabu_len = math.floor(new_tabu_len)
                        self.tabu = deque(self.tabu, new_tabu_len)
                    it_since_tabu_len_changed = 0  # step 8

            # step 10: Save the zoning system and go to step 12.
            # print("step 10")
            self.visited.append(tuple(labels))
            last_step = 10

        if last_step == 10:
            try:
                return np.array(self.visited[-2])
            except IndexError:
                return np.array(self.visited[-1])
        # if step 11 was the last one, the result is in last_step[1]
        return np.array(last_step[1])
    _azp_connected_component.__doc__ = AZP._azp_connected_component.__doc__
