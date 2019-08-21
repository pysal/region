Introduction
============

Regionalization or spatial clustering
-------------------------------------

Let's assume we have a map of areas and each area has an attribute (or a set of attributes) attached to it. Let's further assume we want to cluster the areas according to the attribute(s) into larger spatial units -- let's call these larger spatial units `regions`. If we used a conventional clustering algorithm like the K-Means, we would often end up with regions that are not spatially contiguous (that means the regions wouldn't form a connected set of areas). Sometimes this is not what we want and we are rather interested in a clustering under contiguity restrictions. Other terms for this task are `regionalization` and `spatial clustering`.

This sort of clustering is often one of several steps when analysing geographical data. For example, it helps condense large amounts of data, thus facilitating calculations and improving the readability of visualizations.

region
------

The region package offers various clustering algorithms which can be used with Python 3. The algorithms work with a range of common representations of your map, this means you can use geopandas' ``GeoDataFrame``\s, PySAL's ``W`` objects, networkx' ``Graph``\s, sparse (adjacency) matrices, or simple dictionaries.

With the `p-regions problem <p-regions/index.html>`_ and the `max-p-regions problem <max-p-regions/index.html>`_, region addresses different regionalization tasks.

