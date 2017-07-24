import random

import libpysal as ps
from geopandas import GeoDataFrame
import networkx as nx

from region.util import dataframe_to_dict, find_sublist_containing,\
                        generate_initial_sol, regionalized_components, \
                        make_move, objective_func
from region.move_allowing_strategies import AllowMoveStrategy, \
                                            AllowMoveAZP,\
                                            AllowMoveAZPSimulatedAnnealing, \
                                            AllowMoveAZPTabuSearch


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
            region_list_component = self._azp_connected_component(
                graph, list(nx.connected_components(comp)))
            # ... and collect the results
            region_list += region_list_component
        result_dict = {}
        for area in graph.nodes():
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
                        region_of_cand = region_list_copy[candidates[cand]]
                        del candidates[cand]
                        if self.allow_move_strategy.move_allowed(
                                cand, region_of_cand, region, graph):
                            make_move(cand, region_of_cand, region,
                                      region_list_copy)
                            break
                    else:
                        break

            obj_val_end = objective_func(region_list_copy, graph)
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
            while it < self.maxit and not self.min_sa_moves_reached:  # todo check convergence
                it += 1
                initial_sol = self.azp.fit(
                        areas, data, contiguity, initial_sol)
            # step c
            print(("#"*60 + "\n") * 2 + "STEP C")
            t *= cooling_factor
            self.allow_move_strategy.update_temperature(t)

            if self.move_made:
                self.move_made = False
            else:
                nonmoving_steps += 1

    def sa_moves_alert(self):
        self.min_sa_moves_reached = True

    def move_made_alert(self):
        self.move_made = True


class AZPTabuSearch:
    def __init__(self):
        self.allow_move_strategy = AllowMoveAZPTabuSearch()
