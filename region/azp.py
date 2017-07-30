import abc
from collections import deque, namedtuple
import math
import random

import libpysal as ps
from geopandas import GeoDataFrame
import networkx as nx

from region.util import dataframe_to_dict, find_sublist_containing,\
                        generate_initial_sol, regionalized_components, \
                        make_move, objective_func
from region.move_allowing_strategies import AllowMoveStrategy, \
                                            AllowMoveAZP,\
                                            AllowMoveAZPSimulatedAnnealing


Move = namedtuple("move", "area from_idx to_idx")


class AZP:
    def __init__(self, n_regions, allow_move_strategy=None, random_state=None):
        """

        Parameters
        ----------
        n_regions : int
            The number of regions the areas are clustered into.
        allow_move_strategy : None or AllowMoveStrategy
            If None, then the AZP algorithm in [1]_ is chosen.
            For a different behavior for allowing moves an AllowMoveStrategy
            instance can be passed as argument.
        random_state : None, int, str, bytes, or bytearray
            Random seed.

        References
        ----------
        .. [1] Openshaw S, Rao L. "Algorithms for reengineering 1991 census geography." Environ Plan A. 1995 Mar;27(3):425-46.
        """
        self.labels_ = None
        self.n_regions = n_regions
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

    def fit(self, areas, data, contiguity=None, initial_sol=None):
        """

        Parameters
        ----------
        areas : GeoDataFrame

        data : str or list
            A string to select one column or a list of strings to select
            multiple columns.
        contiguity : {"rook", "queen"}
            This argument defines the contiguity relationship between areas.
        initial_sol : None or dict, default: None
            If None, a starting solution will be computed.
            If `initial_sol` is a dict then the each key must be an area and
            each value must be the corresponding region-ID in the initial
            clustering.

        Returns
        -------
        result_dict : dict
            Each key is an area. Each value is the ID of the region (integer)
            an area belongs to.
        """
        num_areas = len(areas)

        if isinstance(data, str):
            data = [data]
        else:
            data = list(data)
        # todo: check if all elements of data correspond to a col in areas
        # todo: check if all elements of data are different

        if self.n_regions >= num_areas:
            raise ValueError("The n_regions argument must be "
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
        n_comp = nx.number_connected_components(graph)
        if n_comp > self.n_regions:
            raise ValueError("The n_regions argument must not be less than "
                             "the number of connected components.")
        nx.set_node_attributes(graph, "data", dataframe_to_dict(areas, data))

        # step 1
        if initial_sol is not None:
            n_regions_per_comp = {comp: nx.number_connected_components(comp)
                                  for comp in
                                  regionalized_components(initial_sol, graph)}
        else:
            n_regions_per_comp = generate_initial_sol(graph, self.n_regions)
        print(n_regions_per_comp)
        region_list = []
        for comp, n_regions_in_comp in n_regions_per_comp.items():
            # do steps 2-7 for each component separately ...
            if n_regions_in_comp > 1:
                region_list_component = self._azp_connected_component(
                    graph, list(nx.connected_components(comp)))
            else:
                region_list_component = [set(area for area in comp.nodes())]
            # ... and collect the results
            region_list += region_list_component
        result_dict = {}
        for area in graph.nodes():
            # print("area", area, "region_list", region_list)
            result_dict[area] = find_sublist_containing(area, region_list,
                                                        index=True)
        return result_dict

    def _azp_connected_component(self, graph, initial_clustering):
        """

        Parameters
        ----------
        graph : `networkx.Graph`
            A graph containing all areas in `initial_clustering` as nodes.
        initial_clustering : `list`
            Each list element is a `set` containing the areas of a region.
        """
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
                while True:
                    # step 4: identify a set of zones bordering on members of
                    # region K that could be moved into region K without
                    # destroying the internal contiguity of the donor region(s)
                    print("step 4")
                    neighbors_of_region = [neigh for area in region
                                           for neigh in graph.neighbors(area)
                                           if neigh not in region]

                    candidates = {}
                    for neigh in neighbors_of_region:
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
    def __init__(self, n_regions, init_temperature=None,
                 max_iterations=float("inf"), min_sa_moves=0,
                 nonmoving_steps_before_stop=3, random_state=None):

        if init_temperature is not None:
            self.init_temperature = init_temperature
        else:
            raise NotImplementedError("TODO")  # todo

        self.allow_move_strategy = AllowMoveAZPSimulatedAnnealing(
            init_temperature=init_temperature, min_sa_moves=min_sa_moves)
        self.allow_move_strategy.register_min_sa_moves(self.sa_moves_alert)
        self.allow_move_strategy.register_move_made(self.move_made_alert)

        self.azp = AZP(n_regions=n_regions,
                       allow_move_strategy=self.allow_move_strategy,
                       random_state=random_state)

        self.maxit = max_iterations
        self.min_sa_moves = min_sa_moves

        self.min_sa_moves_reached = False
        self.move_made = False
        self.nonmoving_steps_before_stop = nonmoving_steps_before_stop

    def fit(self, areas, data, contiguity=None, initial_sol=None,
            cooling_factor=0.85):
        """

        Parameters
        ----------
        areas :
        data :
        contiguity :
        initial_sol :
        cooling_factor :

        Returns
        -------

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
                initial_sol = self.azp.fit(
                        areas, data, contiguity, initial_sol)

                print("old_sol", old_sol)
                print("new_sol", initial_sol)
                if old_sol is not None:
                    print("EQUAL" if old_sol == initial_sol else "NOT EQUAL")
                    if old_sol == initial_sol:
                        print("BREAK")
                        break
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
            print(old_sol.values())
            print(initial_sol.values())
        return initial_sol

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
    def __init__(self, n_regions, tabu_length=None, random_state=None):
        self.tabu = deque([], tabu_length)
        self.visited = []
        super().__init__(n_regions=n_regions, random_state=random_state)

    def _azp_connected_component(self, graph, initial_clustering,
                                 repetitions_before_termination=5):
        """

        Parameters
        ----------
        graph : `networkx.Graph`
            A graph containing all areas in `initial_clustering` as nodes.
        initial_clustering : `list`
            Each list element is a `set` containing the areas of a region.
        """
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
            # added termination condition (not in Openshaw & Rao (1995))
            if visited.count(region_list) >= repetitions_before_termination:
                stop = True
                print("VISITED", region_list, "FOR",
                      repetitions_before_termination,
                      "TIMES --> TERMINATING BEFORE NEXT NON-IMPROVING MOVE")
            visited.append(region_list)
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


class AZPReactiveTabu(AZPTabu):
    def __init__(self, n_regions, max_iterations, k1, k2, random_state=None):
        self.tabu = deque([], maxlen=1)
        super().__init__(n_regions=n_regions, random_state=random_state)
        self.avg_it_until_rep = 1
        self.rep_counter = 1
        self.maxit = max_iterations
        self.visited = []
        self.k1 = k1
        self.k2 = k2

    def _azp_connected_component(self, graph, initial_clustering):
        """

        Parameters
        ----------
        graph : `networkx.Graph`
            A graph containing all areas in `initial_clustering` as nodes.
        initial_clustering : `list`
            Each list element is a `set` containing the areas of a region.
        """
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
