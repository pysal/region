import abc
import math
import numbers
import random

from region.util import objective_func


class AllowMoveStrategy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, moving_area, from_region, to_region, graph,
                 distance_metric):  # todo: remove `graph` arg and introduce __init__ with `adj` arg (as in :class:`AllowMoveAZPMaxPRegions` in this file)
        """

        Parameters
        ----------
        moving_area

        from_region : `set`

        to_region : `set`

        graph : :class:`networkx.Graph`

        distance_metric : str or function, default: "euclidean"

        Returns
        -------
        is_allowed : `bool`
            `True` if area `moving_area` is allowed to move from `from_region`
            to `to_region`.
            `False` otherwise.
        """


class AllowMoveAZP(AllowMoveStrategy):
    def __call__(self, moving_area, from_region, to_region, graph,
                 distance_metric):
        # before move
        obj_val_before = objective_func(distance_metric,
                                        [from_region, to_region], graph)
        # after move
        region_of_cand_after = from_region.copy()
        region_of_cand_after.remove(moving_area)
        obj_val_after = objective_func(distance_metric,
            [region_of_cand_after, to_region.union({moving_area})], graph)
        if obj_val_after <= obj_val_before:
            return True
        else:
            return False


class AllowMoveAZPSimulatedAnnealing(AllowMoveStrategy):
    def __init__(self, init_temperature, min_sa_moves=float("inf")):
        self.observers_min_sa_moves = []
        self.observers_move_made = []
        self.t = init_temperature
        if not isinstance(min_sa_moves, numbers.Integral) and min_sa_moves < 1:
            raise ValueError("The min_sa_moves argument must be a positive "
                             "integer.")
        self.minsa = min_sa_moves
        self.sa = 0  # number of SA-moves

    def __call__(self, moving_area, from_region, to_region, graph,
                 distance_metric):
        # before move
        obj_val_before = objective_func(distance_metric,
                                        [from_region, to_region], graph)
        # after move
        from_region_after = from_region.copy()
        from_region_after.remove(moving_area)
        obj_val_after = objective_func(distance_metric,
            [from_region_after, to_region.union({moving_area})], graph)
        if obj_val_after <= obj_val_before:
            self.notify_move_made()
            return True
        else:
            obj_val_diff = obj_val_before - obj_val_after
            prob = math.exp(obj_val_diff / self.t)
            move_allowed = random.random() < prob
            if move_allowed:
                self.notify_move_made()
                self.sa += 1
                if self.sa == self.minsa:
                    self.notify_min_sa_moves()
                return True
            return False

    def register_min_sa_moves(self, observer_func):
        """
        Parameters
        ----------
        observer_func : callable
            A function to call when the minimum number of SA moves has been
            reached. This number is called Q in [OR1995]_ (page 431).
        """
        if callable(observer_func):
            self.observers_min_sa_moves.append(observer_func)
        else:
            raise ValueError("The observer_func must be callable.")

    def register_move_made(self, observer_func):
        if callable(observer_func):
            self.observers_move_made.append(observer_func)
        else:
            raise ValueError("The observer_func must be callable.")

    def notify_min_sa_moves(self):
        for observer_func in self.observers_min_sa_moves:
            observer_func()

    def notify_move_made(self):
        for observer_func in self.observers_move_made:
            observer_func()

    def update_temperature(self, temp):
        self.t = temp

    def reset(self):
        self.sa = 0  # number of SA-moves


class AllowMoveAZPMaxPRegions(AllowMoveStrategy):
    """
    Ensures that the spatially extensive attribute adds up to a
    given threshold in each region. Only moves preserving this condition in
    both the donor as well as the recipient region are allowed. The check for
    the recipient region is necessary in case there is an area with a negative
    spatially extensive attribute.
    """
    def __init__(self, adj, spatially_extensive_attr, threshold,
                 decorated_strategy):
        """

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`

        spatially_extensive_attr : :class:`numpy.ndarray`

        threshold : `float`

        decorated_strategy : :class:`AllowMoveStrategy`

        """
        self.adj = adj
        self.spatially_extensive_attr = spatially_extensive_attr
        self.threshold = threshold
        self._decorated_strategy = decorated_strategy

    def __call__(self, moving_area, donor_region, recipient_region, graph,
                 distance_metric):
        donor_sum = sum(self.spatially_extensive_attr[area]
                        for area in donor_region if area is not moving_area)
        threshold_reached_donor = donor_sum >= self.threshold

        recipient_sum = sum(self.spatially_extensive_attr[area]
                            for area in recipient_region) \
            + self.spatially_extensive_attr[moving_area]
        threshold_reached_recipient = recipient_sum >= self.threshold

        if threshold_reached_donor and threshold_reached_recipient:
            # todo: refactor to use sparse matrix instead of networkx graph:
            return self._decorated_strategy(moving_area, donor_region,
                                            recipient_region, graph,
                                            distance_metric)
        return False

    def __getattr__(self, name):
        """
        Forward calls to unimplemented methods to _decorated_strategy
        (necessary e.g. for :class:`AllowMoveAZPSimulatedAnnealing`).
        """
        return getattr(self._decorated_strategy, name)
