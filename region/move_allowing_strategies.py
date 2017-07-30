import abc
import numbers

import math
import random

from region.util import objective_func


class AllowMoveStrategy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, moving_area, from_region, to_region, graph):
        """

        Parameters
        ----------
        moving_area
        from_region
        to_region
        graph

        Returns
        -------
        is_allowed : bool
            True if area `moving_area` is allowed to move from `from_region` to
            `to_region`.
            False otherwise.
        """


class AllowMoveAZP(AllowMoveStrategy):
    def __call__(self, moving_area, from_region, to_region, graph):
        # before move
        obj_val_before = objective_func([from_region, to_region], graph)
        # after move
        region_of_cand_after = from_region.copy()
        region_of_cand_after.remove(moving_area)
        obj_val_after = objective_func(
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

    def __call__(self, moving_area, from_region, to_region, graph):
        # before move
        obj_val_before = objective_func([from_region, to_region], graph)
        # after move
        from_region_after = from_region.copy()
        from_region_after.remove(moving_area)
        obj_val_after = objective_func(
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
            reached. This number is called Q in [1]_ (page 431).

        References
        ----------
        .. [1] Openshaw S, Rao L. "Algorithms for reengineering 1991 census geography." Environ Plan A. 1995 Mar;27(3):425-46.
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
