import abc
import math
import numbers
import random

import numpy as np

from region.util import objective_func_arr, objective_func_diff


class AllowMoveStrategy(abc.ABC):
    def __init__(self, attr=None, metric=None):
        """
        Parameters
        ----------
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used to
            determine the objective value of a given clustering.
        metric : function
            See the `metric` argument in
            :func:`region.util.get_metric_function`.
        """
        self.attr_all = attr
        self.attr = None
        self.metric = metric
        self.comp_idx = None
        self.objective = float("inf")

    @property
    def attr_all(self):
        if self._attr_all is None:
            raise NameError("attr_all of {} not set.".format(self))
        return self._attr_all

    @attr_all.setter
    def attr_all(self, value):
        self._attr_all = value

    @property
    def metric(self):
        if self._metric is None:
            raise NameError("metric of {} not set.".format(self))
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value

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
        self.objective = objective_func_arr(initial_labels, self.attr,
                                            metric=self.metric)

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
    def __call__(self, moving_area, new_region, labels):
        diff = objective_func_diff(labels, self.attr, moving_area, new_region,
                                   self.metric)
        if diff <= 0:
            # print("  allowing move of {} to {}".format(moving_area, new_region))
            # print("  because diff {} <= 0".format(diff))
            self.objective += diff
            return True
        else:
            # print("  disallowing move of {} to {}".format(moving_area,
            #                                               new_region))
            # print("  because diff {} > 0".format(diff))
            return False


class AllowMoveAZPSimulatedAnnealing(AllowMoveStrategy):
    def __init__(self, init_temperature, sa_moves_term=float("inf"), attr=None,
                 metric=None):
        self.observers_min_sa_moves = []
        self.observers_move_made = []
        self.t = init_temperature
        if not isinstance(sa_moves_term, numbers.Integral) or \
                sa_moves_term < 1:
            raise ValueError("The sa_moves_term argument must be a positive "
                             "integer.")
        print("sa_moves_term:", sa_moves_term)
        self.sa_moves_term = sa_moves_term
        self.sa = 0  # number of SA-moves
        super().__init__(attr=attr, metric=metric)

    def __call__(self, moving_area, new_region, labels):
        diff = objective_func_diff(labels, self.attr, moving_area, new_region,
                                   self.metric)
        if diff <= 0:
            # print("  allowing move of {} to {}".format(moving_area, new_region))
            # print("  because diff {} <= 0".format(diff))
            self.objective += diff
            self.notify_move_made()
            return True
        else:
            # print("  disallowing move of {} to {}".format(moving_area,
            #                                               new_region))
            # print("  because diff {} > 0".format(diff))
            prob = math.exp(-diff / self.t)
            move_allowed = random.random() < prob
            if move_allowed:
                self.notify_move_made()
                self.sa += 1
                if self.sa >= self.sa_moves_term:
                    self.notify_min_sa_moves()
                print("move {} to {} anyway!".format(moving_area, new_region))
                self.objective += diff
                return True
            print("not moving {} to {}".format(moving_area, new_region))
            return False

    def register_sa_moves_term(self, observer_func):
        """
        Parameters
        ----------
        observer_func : callable
            A function to call when the certain number of SA-moves has been
            reached. This number is called Q in [OR1995]_ (page 431).
        """
        if callable(observer_func):
            self.observers_min_sa_moves.append(observer_func)
        else:
            raise ValueError("The observer_func must be callable.")

    def register_move_made(self, observer_func):
        """
        Parameters
        ----------
        observer_func : callable
            A function to call when a move is allowed.
        """
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
    def __init__(self, attr, spatially_extensive_attr, threshold, metric,
                 decorated_strategy):
        """

        Parameters
        ----------
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used to
            determine the objective value of a given clustering.
        spatially_extensive_attr : :class:`numpy.ndarray`, default: None
            See corresponding argument in
            :meth:`region.max_p_regions.heuristics.MaxPRegionsHeu.fit_from_scipy_sparse_matrix`.
        threshold : numbers.Real or :class:`numpy.ndarray`
            See corresponding argument in
            :meth:`region.max_p_regions.heuristics.MaxPRegionsHeu.fit_from_scipy_sparse_matrix`
        metric : function
            See corresponding argument in
            :meth:`region.p_regions.azp_util.AllowMoveStrategy.__init__`.
        decorated_strategy : :class:`AllowMoveStrategy`
            The :class:`AllowMoveStrategy` related to the algorithms local
            search.
        """
        self._decorated_strategy = decorated_strategy
        self.spatially_extensive_attr_all = spatially_extensive_attr
        self.spatially_extensive_attr = None
        self.threshold = threshold
        super().__init__(attr=attr, metric=metric)

    @property
    def attr_all(self):
        if self._attr_all is None:
            raise NameError("attr_all of {} not set.".format(self))
        return self._attr_all

    @attr_all.setter
    def attr_all(self, value):
        self._attr_all = value
        self._decorated_strategy.attr_all = value

    @property
    def metric(self):
        if self._metric is None:
            raise NameError("metric of {} not set.".format(self))
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value
        self._decorated_strategy.metric = value

    def start_new_component(self, initial_labels, comp_idx):
        """
        Parameters
        ----------
        initial_labels : :class:`numpy.ndarray`

        comp_idx
            Set the instances `comp_idx` attribute. This method should be
            called whenever a new connected component is clustered.
        """
        self.spatially_extensive_attr = self.spatially_extensive_attr_all[
                comp_idx]
        super().start_new_component(initial_labels, comp_idx)
        self._decorated_strategy.start_new_component(initial_labels, comp_idx)

    def __call__(self, moving_area, new_region, labels):
        sp_ext = self.spatially_extensive_attr

        if (sp_ext[moving_area]).any() > 0:
            donor_region = labels[moving_area]
            donor_idx = np.where(labels == donor_region)[0]
            donor_sum = sum(sp_ext[donor_idx]) - sp_ext[moving_area]
            threshold_reached_donor = (donor_sum >= self.threshold).all()
            if not threshold_reached_donor:
                print("Region", donor_idx, "must not lose", moving_area)
                return False
            print("Region", donor_idx, "may donate", moving_area)

        elif (sp_ext[moving_area]).any() < 0:
            recipient_idx = np.where(labels == new_region)[0]
            recipient_sum = sum(sp_ext[recipient_idx]) + sp_ext[moving_area]
            threshold_reached_recipient = (recipient_sum >=
                                           self.threshold).all()
            if not threshold_reached_recipient:
                print("Area", moving_area, "with spat_ext_attr<0 may not move")
                return False

        return self._decorated_strategy(moving_area, new_region, labels)

    def __getattr__(self, name):
        """
        Forward calls to unimplemented methods to _decorated_strategy
        (necessary e.g. for :class:`AllowMoveAZPSimulatedAnnealing`).
        """
        return getattr(self._decorated_strategy, name)
