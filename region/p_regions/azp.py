import abc
from collections import deque
import math
import random

import numpy as np
import networkx as nx
from scipy import sparse as sp

from region import fit_functions
from region.p_regions.azp_util import AllowMoveStrategy, \
                                            AllowMoveAZP,\
                                            AllowMoveAZPSimulatedAnnealing
from region.util import find_sublist_containing, Move, make_move, \
                        objective_func, dict_to_region_list, assert_feasible, \
                        separate_components, generate_initial_sol, copy_func, \
                        array_from_dict_values


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
        if allow_move_strategy is None:
            self.allow_move_strategy = allow_move_strategy = AllowMoveAZP()
        if isinstance(allow_move_strategy, AllowMoveStrategy):
            self.allow_move_strategy = allow_move_strategy
        else:
            raise ValueError(wrong_allow_move_arg_msg)

    def fit_from_scipy_sparse_matrix(self, adj, data, n_regions,
                                     initial_sol=None):
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
        """
        # step 1
        if initial_sol is not None:
            assert_feasible(initial_sol, adj, n_regions)
            initial_sol_gen = separate_components(adj, initial_sol)
        else:
            initial_sol_gen = generate_initial_sol(adj, n_regions)
        region_labels = -np.ones(adj.shape[0])
        regions_built = 0
        for comp in initial_sol_gen:
            in_comp = comp != -1
            print("Clustering component ", in_comp)
            comp_data = data[in_comp]
            initial_region_labels = comp[in_comp]
            print("Starting with initial clustering", initial_region_labels)
            comp_adj = adj[in_comp]
            comp_adj = comp_adj[:, in_comp]
            region_list_component = self._azp_connected_component(
                comp_adj, initial_region_labels, comp_data)
            region_list_flat = [
                find_sublist_containing(i, region_list_component, index=True)
                for i in range(sum(in_comp))
            ]
            region_labels[in_comp] = region_list_flat
            region_labels[in_comp] += regions_built
            regions_built += len(set(region_list_flat))
        self.n_regions = n_regions
        self.labels_ = region_labels

    fit = copy_func(fit_from_scipy_sparse_matrix)
    fit.__doc__ = "Alias for :meth:`fit_from_scipy_sparse_matrix`.\n\n" \
                  + fit_from_scipy_sparse_matrix.__doc__

    def fit_from_w(self, w, data, n_regions, initial_sol=None):
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
        """
        adj = w.sparse
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol)

    def fit_from_networkx(self, graph, data, n_regions, initial_sol=None):
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
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol)

    def fit_from_geodataframe(self, gdf, data, n_regions, contiguity="rook",
                              initial_sol=None):
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
        """
        fit_functions.fit_from_geodataframe(self, gdf, data, n_regions,
                                            contiguity=contiguity,
                                            initial_sol=initial_sol)

    def fit_from_dict(self, neighbor_dict, data, n_regions, initial_sol=None):
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
                                          n_regions, initial_sol)

    def _azp_connected_component(self, adj, initial_clustering, data):
        """
        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Adjacency matrix representing the contiguity relation.
        initial_clustering : `list`
            Each list element is a `set` containing the areas of a region.
        data : :class:`numpy.ndarray`
            Clustering criterion. The length of this one-dimensional array is
            equal to the number of regions.

        Returns
        -------
        region_list_copy : `list`
            Each element is an iterable of areas representing a region.
        """
        graph = nx.from_scipy_sparse_matrix(adj)
        nx.set_node_attributes(graph, "data",
                               {n: data_n for n, data_n in enumerate(data)})
        initial_clustering_dict = {area: reg for area, reg
                                   in enumerate(initial_clustering)}
        initial_clustering = dict_to_region_list(initial_clustering_dict)
        # if there is only one region in the initial solution, just return it.
        if len(initial_clustering) == 1:
            return initial_clustering
        #  step 2: make a list of the M regions
        region_list = initial_clustering
        region_list_copy = region_list.copy()

        # todo: rm print-statements
        print("Init with: ", initial_clustering)
        obj_val_start = float("inf")  # since Python 3.5 math.inf also possible
        obj_val_end = objective_func(region_list, graph)
        # step 7: Repeat until no further improving moves are made
        while obj_val_end < obj_val_start:  # improvement
            print("obj_val:", obj_val_start, "-->", obj_val_end,
                  "...continue...")
            print("=" * 45)
            # print("step 7")
            obj_val_start = obj_val_end
            print("step 2")
            region_list = region_list_copy.copy()
            print("obj_value:", obj_val_end)
            print(region_list)
            # step 6: when the list for region K is exhausted return to step 3
            # and select another region and repeat steps 4-6
            print("-" * 35)
            # print("step 6")
            while region_list:
                # step 3: select & remove any region K at random from this list
                print("step 3")
                random_position = random.randrange(len(region_list))
                region = region_list.pop(random_position)
                region_idx = region_list_copy.index(region)
                print("  chosen region:", region)
                while True:
                    # step 4: identify a set of zones bordering on members of
                    # region K that could be moved into region K without
                    # destroying the internal contiguity of the donor region(s)
                    print("step 4")
                    neighbors_of_region = [neigh for area in region
                                           for neigh in graph.neighbors(area)
                                           if neigh not in region]

                    candidates = {}
                    print("  neighbors_of_region:", neighbors_of_region)
                    for neigh in neighbors_of_region:
                        print("  neigh:", neigh)
                        region_index_of_neigh = find_sublist_containing(
                            neigh, region_list_copy, index=True)
                        region_of_neigh = region_list_copy[
                            region_index_of_neigh]
                        try:
                            if nx.is_connected(
                                    graph.subgraph(region_of_neigh - {neigh})):
                                candidates[neigh] = region_index_of_neigh
                        except nx.NetworkXPointlessConcept:
                            # if area is the only one in region, it has to stay
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
                        cand_region_idx = candidates[cand]
                        cand_region = region_list_copy[candidates[cand]]
                        del candidates[cand]
                        if self.allow_move_strategy(
                                cand, cand_region, region, graph):
                            make_move(cand, cand_region_idx, region_idx,
                                      region_list_copy)
                            break
                    else:
                        break

            obj_val_end = objective_func(region_list_copy, graph)
        print("RETURN: ", region_list_copy)
        return region_list_copy


class AZPSimulatedAnnealing:
    def __init__(self, init_temperature=None,
                 max_iterations=float("inf"), min_sa_moves=0,
                 nonmoving_steps_before_stop=3,
                 repetitions_before_termination=5, random_state=None):

        if init_temperature is not None:
            self.init_temperature = init_temperature
        else:
            raise NotImplementedError("TODO")  # todo

        self.allow_move_strategy = AllowMoveAZPSimulatedAnnealing(
            init_temperature=init_temperature, min_sa_moves=min_sa_moves)
        self.allow_move_strategy.register_min_sa_moves(self.sa_moves_alert)
        self.allow_move_strategy.register_move_made(self.move_made_alert)

        self.azp = AZP(allow_move_strategy=self.allow_move_strategy,
                       random_state=random_state)
        self.labels_ = None

        self.maxit = max_iterations
        self.min_sa_moves = min_sa_moves

        self.min_sa_moves_reached = False
        self.move_made = False
        self.nonmoving_steps_before_stop = nonmoving_steps_before_stop

        self.visited = []
        self.reps_before_termination = repetitions_before_termination

    def fit_from_geodataframe(self, gdf, data, n_regions,
                              contiguity="rook",initial_sol=None,
                              cooling_factor=0.85):
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
        """
        fit_functions.fit_from_geodataframe(
                self, gdf, data, n_regions, contiguity=contiguity,
                initial_sol=initial_sol, cooling_factor=cooling_factor)

    def fit_from_dict(self, neighbor_dict, data, n_regions, initial_sol=None,
                      cooling_factor=0.85):
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
                cooling_factor=cooling_factor)

    def fit_from_networkx(self, graph, data, n_regions, initial_sol=None,
                          cooling_factor=0.85):
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
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol,
                                          cooling_factor=cooling_factor)

    def fit_from_scipy_sparse_matrix(self, adj, data, n_regions,
                                     initial_sol=None, cooling_factor=0.85):
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
        """
        if not (0 < cooling_factor < 1):
            raise ValueError("The cooling_factor argument must be greater "
                             "than 0 and less than 1")
        # todo: rm print() calls
        # step a
        print(("#"*60 + "\n") * 5 + "STEP A")
        t = self.init_temperature
        nonmoving_steps = 0
        # step d: repeat step b and c
        while nonmoving_steps < self.nonmoving_steps_before_stop:
            print(("#"*60 + "\n") * 2 + "STEP B")
            it = 0
            self.min_sa_moves_reached = False
            # step b
            while it < self.maxit and not self.min_sa_moves_reached:
                it += 1
                old_sol = initial_sol
                self.azp.fit_from_scipy_sparse_matrix(adj, data, n_regions,
                                                      initial_sol)
                initial_sol = self.azp.labels_

                print("old_sol", old_sol)
                print("new_sol", initial_sol)
                if old_sol is not None:
                    print("EQUAL" if (old_sol == initial_sol).all()
                          else "NOT EQUAL")
                    if (old_sol == initial_sol).all():
                        print("BREAK")
                        break
            print("visited", self.visited)
            # added termination condition (not in Openshaw & Rao (1995))
            print(initial_sol)
            if self.visited.count(tuple(initial_sol)) >= self.reps_before_termination:
                print("VISITED", initial_sol, "FOR",
                      self.reps_before_termination,
                      "TIMES --> TERMINATING.")
                break
            self.visited.append(tuple(initial_sol))
            # step c
            print(("#"*60 + "\n") * 2 + "STEP C")
            t *= cooling_factor
            self.allow_move_strategy.update_temperature(t)

            if self.move_made:
                print("MOVE MADE")
                self.move_made = False
            else:
                print("NO MOVE MADE")
                nonmoving_steps += 1
            print(old_sol)
            print(initial_sol)
        self.labels_ = initial_sol

    def fit_from_w(self, w, data, n_regions, initial_sol=None,
                   cooling_factor=0.85):
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
        """
        adj = w.sparse
        self.fit_from_scipy_sparse_matrix(adj, data, n_regions, initial_sol,
                                          cooling_factor=cooling_factor)

    def sa_moves_alert(self):
        self.min_sa_moves_reached = True

    def move_made_alert(self):
        self.move_made = True


class AZPTabu(AZP, abc.ABC):
    def _make_move(self, area, from_idx, to_idx, region_list):
        make_move(area, from_idx, to_idx, region_list)
        # step 5: Tabu the reverse move for R iterations.
        reverse_move = Move(area, to_idx, from_idx)
        self.tabu.append(reverse_move)

    def _objval_diff(self, area, from_idx, to_idx, region_list, graph):
        from_region = region_list[from_idx]
        to_region = region_list[to_idx]
        # before move
        objval_before = objective_func(
                [from_region, to_region], graph)
        # after move
        region_of_cand_after = from_region.copy()
        region_of_cand_after.remove(area)
        objval_after = objective_func(
            [region_of_cand_after,
             to_region.union({area})], graph)
        return objval_after - objval_before


class AZPBasicTabu(AZPTabu):
    def __init__(self, tabu_length=None,
                 repetitions_before_termination=5, random_state=None):
        self.tabu = deque([], tabu_length)
        self.visited = []
        self.reps_before_termination = repetitions_before_termination
        super().__init__(random_state=random_state)

    def _azp_connected_component(self, adj, initial_clustering, data):
        graph = nx.from_scipy_sparse_matrix(adj)
        nx.set_node_attributes(graph, "data",
                               {n: data_n for n, data_n in enumerate(data)})
        initial_clustering_dict = {area: reg for area, reg
                                   in enumerate(initial_clustering)}
        initial_clustering = dict_to_region_list(initial_clustering_dict)
        # if there is only one region in the initial solution, just return it.
        if len(initial_clustering) == 1:
            return initial_clustering
        #  step 2: make a list of the M regions
        region_list = initial_clustering
        areas_in_component = (a for sublist in initial_clustering
                              for a in sublist)
        graph = graph.subgraph(areas_in_component)

        # todo: rm print-statements
        print("Init with: ", initial_clustering)
        visited = []
        stop = False
        while True:  # TODO: condition??
            print("visited", visited)
            # added termination condition (not in Openshaw & Rao (1995))
            region_set = set(frozenset(region) for region in region_list)
            if visited.count(region_set) >= self.reps_before_termination:
                stop = True
                print("VISITED", region_list, "FOR",
                      self.reps_before_termination,
                      "TIMES --> TERMINATING BEFORE NEXT NON-IMPROVING MOVE")
            visited.append(region_set)
            print("=" * 45)
            obj_val_end = objective_func(region_list, graph)
            print("obj_value:", obj_val_end)
            print(region_list)
            print("-" * 35)
            # step 1 Find the global best move that is not prohibited or tabu.
            print("step 1")
            # find possible moves (globally)
            best_move = None
            best_objval_diff = float("inf")
            for area in graph.nodes():
                try:
                    from_idx = find_sublist_containing(
                            area, region_list, index=True)
                    from_region = region_list[from_idx]
                    area_region_wo_area = from_region - {area}
                    if nx.is_connected(graph.subgraph(area_region_wo_area)):
                        for neigh in graph.neighbors(area):
                            if neigh not in from_region:
                                to_idx = find_sublist_containing(
                                        neigh, region_list, index=True)
                                possible_move = Move(area, from_idx, to_idx)
                                if possible_move not in self.tabu:
                                    objval_diff = self._objval_diff(
                                            *possible_move, region_list, graph)
                                    if objval_diff < best_objval_diff:
                                        best_move = possible_move
                                        best_objval_diff = objval_diff
                except nx.NetworkXPointlessConcept:
                    # if area is the only one in region, it has to stay
                    pass
            print("  best move:", best_move, "objval_diff:", best_objval_diff)
            # step 2: Make this move if it is an improvement or equivalet in
            # value.
            print("step 2")
            if best_move is not None and best_objval_diff <= 0:
                print(region_list)
                print("IMPROVING MOVE")
                self._make_move(*best_move, region_list)
            else:
                # step 3: if no improving move can be made, then see if a tabu
                # move can be made which improves on the current local best
                # (termed an aspiration move)
                print("step 3")
                print("Tabu:", self.tabu)
                improving_tabus = [
                    move for move in self.tabu
                    if move.area in region_list[move.from_idx] and
                    self._objval_diff(*move, region_list, graph) < 0]
                if improving_tabus:
                    random_position = random.randrange(len(improving_tabus))
                    aspiration_move = improving_tabus[random_position]
                    print(region_list)
                    print("ASPIRATION MOVE")
                    self._make_move(*aspiration_move, region_list)
                else:
                    # step 4: If there is no improving move and no aspirational
                    # move, then make the best move even if it is nonimproving
                    # (that is, results in a worse value of the objective
                    # function).
                    print("step 4")
                    print(region_list)
                    print("No improving, no aspiration ==> do the best you can")
                    if stop:
                        break
                    if best_move is not None:
                        self._make_move(*best_move, region_list)
        return region_list
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

    def _azp_connected_component(self, adj, initial_clustering, data):
        graph = nx.from_scipy_sparse_matrix(adj)
        nx.set_node_attributes(graph, "data",
                               {n: data_n for n, data_n in enumerate(data)})
        initial_clustering_dict = {area: reg for area, reg
                                   in enumerate(initial_clustering)}
        initial_clustering = dict_to_region_list(initial_clustering_dict)
        # if there is only one region in the initial solution, just return it.
        if len(initial_clustering) == 1:
            return initial_clustering
        last_step = 1
        #  step 2: make a list of the M regions
        region_list = initial_clustering
        areas_in_component = (a for sublist in initial_clustering
                              for a in sublist)
        graph = graph.subgraph(areas_in_component)

        # todo: rm print-statements
        print("Init with: ", initial_clustering)
        it_since_tabu_len_changed = 0
        obj_val_start = float("inf")
        # step 12: Repeat steps 3-11 until either no further improvements are
        # made or a maximum number of iterations are exceeded.
        for it in range(self.maxit):
            print("=" * 45)
            print(region_list)
            obj_val_end = objective_func(region_list, graph)
            print("obj_value:", obj_val_end)
            if not obj_val_end < obj_val_start:
                break  # step 12
            obj_val_start = obj_val_end

            it_since_tabu_len_changed += 1
            print("-" * 35)
            # step 3: Define the list of all possible moves
            print("step 3")
            possible_moves = []
            for area in graph.nodes():
                try:
                    from_idx = find_sublist_containing(
                            area, region_list, index=True)
                    from_region = region_list[from_idx]
                    area_region_wo_area = from_region - {area}
                    if nx.is_connected(graph.subgraph(area_region_wo_area)):
                        for neigh in graph.neighbors(area):
                            if neigh not in from_region:
                                to_idx = find_sublist_containing(
                                        neigh, region_list, index=True)
                                move = Move(area, from_idx, to_idx)
                                if move not in self.tabu:
                                    possible_moves.append(move)
                except nx.NetworkXPointlessConcept:
                    # if area is the only one in region, it has to stay
                    pass
            # step 4: Find the best nontabu move.
            print("step 4")
            best_move = None
            best_move_index = None
            best_objval_diff = float("inf")
            for i, move in enumerate(possible_moves):
                obj_val_diff = self._objval_diff(
                        *move, region_list, graph)
                if obj_val_diff < best_objval_diff:
                    best_move_index, best_move = i, move
                    best_objval_diff = obj_val_diff
            print("  best move:", best_move)
            # step 5: Make the move. Update tabu status.
            print("step 5: make", best_move)
            self._make_move(*best_move, region_list)
            # step 6: Look up the current zoning system in a list of all zoning
            # systems visited so far during the search. If not found then go
            # to step 10.
            print("step 6")
            # Sets can't be permuted so we convert our list to a set:
            zoning_system = set(frozenset(s) for s in region_list)
            if zoning_system in self.visited:
                # step 7: If it is found and it has been visited more than K1
                # times already and this cyclical behavior has been found on
                # at least K2 other occasions (involving other zones) then go
                # to step 11.
                print("step 7")
                print("  region_list", region_list)
                print("  self.visited:", self.visited)
                times_visited = self.visited.count(zoning_system)
                cycle = list(reversed(self.visited))
                cycle = cycle[:cycle.index(zoning_system) + 1]
                cycle = list(reversed(cycle))
                print("  cycle:", cycle)
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
                        print("step 11")
                        # we save region_list such that we can access it if
                        # this step yields a poor solution.
                        last_step = (11, region_list)
                        self.visited = []
                        p = math.floor(1 + self.avg_it_until_rep/2)
                        possible_moves.pop(best_move_index)
                        for _ in range(p):
                            move = possible_moves.pop(
                                    random.randrange(len(possible_moves)))
                            self._make_move(*move, region_list)
                        continue
                    # step 8: Update a moving average of the repetition
                    # interval self.avg_it_until_rep, and increase the
                    # prohibition period R to 1.1*R.
                    print("step 8")
                    self.rep_counter += 1
                    avg_it = self.avg_it_until_rep
                    self.avg_it_until_rep = 1 / self.rep_counter * \
                        ((self.rep_counter-1)*avg_it + it_until_repetition)

                    self.tabu = deque(self.tabu, 1.1*self.tabu.maxlen)
                    # step 9: If the number of iterations since R was last
                    # changed exceeds self.avg_it_until_rep, then decrease R to
                    # max(0.9*R, 1).
                    print("step 9")
                    if it_since_tabu_len_changed > self.avg_it_until_rep:
                        new_tabu_len = max([0.9*self.tabu.maxlen, 1])
                        new_tabu_len = math.floor(new_tabu_len)
                        self.tabu = deque(self.tabu, new_tabu_len)
                    it_since_tabu_len_changed = 0  # step 8

            # step 10: Save the zoning system and go to step 12.
            print("step 10")
            self.visited.append(zoning_system)
            last_step = 10

        if last_step == 10:
            try:
                return self.visited[-2]
            except IndexError:
                return self.visited[-1]
        # if step 11 was the last one, the result is in last_step[1]
        return last_step[1]
    _azp_connected_component.__doc__ = AZP._azp_connected_component.__doc__
