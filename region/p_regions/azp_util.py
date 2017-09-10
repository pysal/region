import abc
import math
import numbers
import random

import numpy as np

from region.util import objective_func_arr, objective_func_diff


class AllowMoveStrategy(abc.ABC):
    def __init__(self, attr, metric):
        """
        Parameters
        ----------
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used to
            determine the objective value of a given clustering.
        metric : function
            See the `metric` argument in
            :func:`region.util.set_distance_metric`.
        """
        self.attr_all = attr
        self.attr = None
        self.metric = metric
        self.comp_idx = None
        self.objective = float("inf")

    def start_new_component(self, initial_labels, comp_idx):
        """
        Parameters
        ----------
        initial_labels : :class:`numpy.ndarray`

        comp_idx : :class:`numpy.ndarray`
            Set the instance's `comp_idx` attribute. This method should be
            called whenever a new connected component is clustered.
        """
        self.comp_idx = comp_idx
        self.attr = self.attr_all[comp_idx]
        self.objective = objective_func_arr(self.metric, initial_labels,
                                            self.attr)

    @abc.abstractmethod
    def __call__(self, moving_area, new_region, labels):
        """
        Assess whether a potential move is allowed or not. The criteria for
        allowing a move are defined in the implementations of this abstract
        method (in subclasses).

        Parameters
        ----------
        moving_area : int
            Area involved in a potential move.
        new_region : int
            Region to which the area `moving_area` is moved if the move is
            allowed.
        labels : :class:`numpy.ndarray`
            Region labels of areas in the currently considered connected
            component.

        Returns
        -------
        is_allowed : `bool`
            `True` if area `moving_area` is allowed to move to `new_region`.
            `False` otherwise.
        """


class AllowMoveAZP(AllowMoveStrategy):
    def __init__(self, attr, metric):
        super().__init__(attr=attr, metric=metric)

    def __call__(self, moving_area, new_region, labels):
        donor_diff, recipient_diff = objective_func_diff(
                self.metric, labels, self.attr, moving_area, new_region)
        diff = donor_diff + recipient_diff
        if diff <= 0:
            # print("  allowing move because diff {} <= 0".format(diff))
            self.objective += diff
            return True
        else:
            # print("  disallowing move because diff {} > 0".format(diff))
            return False


class AllowMoveAZPSimulatedAnnealing(AllowMoveStrategy):
    def __init__(self, attr, metric, init_temperature,
                 min_sa_moves=float("inf")):
        self.observers_min_sa_moves = []
        self.observers_move_made = []
        self.t = init_temperature
        if not isinstance(min_sa_moves, numbers.Integral) and min_sa_moves < 1:
            raise ValueError("The min_sa_moves argument must be a positive "
                             "integer.")
        self.minsa = min_sa_moves
        self.sa = 0  # number of SA-moves
        super().__init__(attr=attr, metric=metric)

    def __call__(self, moving_area, new_region, labels):
        dist_met = self.metric
        old_region = labels[moving_area]
        # before move
        obj_val_before = objective_func_arr(dist_met, labels, self.attr,
                                            {old_region, new_region})
        # after move
        labels[moving_area] = new_region
        obj_val_after = objective_func_arr(dist_met, labels, self.attr,
                                           {old_region, new_region})
        labels[moving_area] = old_region
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
    def __init__(self, metric, attr, spatially_extensive_attr, threshold,
                 decorated_strategy):
        """

        Parameters
        ----------
        metric : function
            See corresponding argument in
            :meth:`region.p_regions.azp_util.AllowMoveStrategy.__init__`.
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used to
            determine the objective value of a given clustering.
        spatially_extensive_attr : :class:`numpy.ndarray`, default: None
            See corresponding argument in
            :meth:`region.max_p_regions.heuristics.MaxPHeu.fit_from_scipy_sparse_matrix`.
        threshold : `float`
            See corresponding argument in
            :meth:`region.max_p_regions.heuristics.MaxPHeu.fit_from_scipy_sparse_matrix`
        decorated_strategy : :class:`AllowMoveStrategy`
            The :class:`AllowMoveStrategy` related to the algorithms local
            search.
        """
        self.spatially_extensive_attr_all = spatially_extensive_attr
        self.spatially_extensive_attr = None
        self.threshold = threshold
        self._decorated_strategy = decorated_strategy
        super().__init__(attr=attr, metric=metric)

    def start_new_component(self, comp_idx):
        """
        Parameters
        ----------
        comp_idx
            Set the instances `comp_idx` attribute. This method should be
            called whenever a new connected component is clustered.
        """
        self.spatially_extensive_attr = self.spatially_extensive_attr_all[
                comp_idx]
        super().start_new_component(comp_idx)

    def __call__(self, moving_area, new_region, labels, comp_idx):
        spat_ext = self.spatially_extensive_attr

        if spat_ext[moving_area] > 0:
            donor_region = labels[moving_area]
            donor_idx = np.where(labels == donor_region)[0]
            donor_sum = spat_ext[donor_idx] - spat_ext[moving_area]
            threshold_reached_donor = donor_sum >= self.threshold
            if not threshold_reached_donor:
                return False

        elif spat_ext[moving_area] < 0:
            recipient_idx = np.where(labels == new_region)[0]
            recipient_sum = spat_ext[recipient_idx] + spat_ext[moving_area]
            threshold_reached_recipient = recipient_sum >= self.threshold
            if not threshold_reached_recipient:
                return False

        return self._decorated_strategy(moving_area, new_region, labels,
                                        comp_idx)

    def __getattr__(self, name):
        """
        Forward calls to unimplemented methods to _decorated_strategy
        (necessary e.g. for :class:`AllowMoveAZPSimulatedAnnealing`).
        """
        return getattr(self._decorated_strategy, name)
