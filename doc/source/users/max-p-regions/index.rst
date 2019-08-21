The Max-p-Regions Problem
=========================

The task
--------

As the p-regions problem the `max-p-regions` problem is about finding a clustering which satisfies the spatial contiguity condition. A key difference is that in the max-p-regions problem there is no predefined number of regions. The primary goal is to find a clustering that has as many regions as possible. In order to avoid a clustering in which every area forms its own region there is another condition. It is stated by using so called `spatial extensive attributes` which each area has. The condition requires that the sum of these spatial extensive attributes reaches or exceeds a predefined threshold.

When the maximum number of regions is found, a second goal has to be met, namely to find the best clustering with the maximum number of regions. Here, optimality is defined by the areas' attributes. (These attributes may -- and in most applications will -- be different from the spatial extensive attributes mentioned above.) The max-p-regions problem can be solved using either an `exact <#max-p-region-exact>`_ or a `heuristic <#max-p-region-heu>`_ approach. For a detailed description please refer to [DAR2012]_.



.. _max-p-region-exact:

Exact methods
-------------

[DAR2012]_ shows a way to translate the max-p-regions problem into an integer programming problem. It is implemented in the :class:`region.max_p_regions.exact.MaxPRegionsExact` class.

.. _max-p-region-heu:

Heuristic methods
-----------------

Since solving the problem exactly may be very slow, [DAR2012]_ also suggests an heuristic approach. This algorithm involves two steps - a so called `construction phase` and a `local search phase`. You can find the implementation in the :class:`region.max_p_regions.heuristics.MaxPRegionsHeu` class. Please note that region's implementation uses a modified heuristic p-regions-algorithm for the local search phase. The modification ensures the threshold condition on the spatial extensive attributes is met.



