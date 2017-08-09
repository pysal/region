import collections

from geopandas import GeoDataFrame
import numpy as np
import networkx as nx
import libpysal as ps
import pulp

from region.util import dataframe_to_dict


def check_solver(solver):
    if not isinstance(solver, str) \
            or solver.lower() not in ["cbc", "cplex", "glpk", "gurobi"]:
        raise ValueError("The solver argument must be one of the following"
                         ' strings: "cbc", "cplex", "glpk", or "gurobi".')


def get_solver_instance(solver_string):
    solver = {"cbc": pulp.solvers.COIN_CMD,
              "cplex": pulp.solvers.CPLEX,
              "glpk": pulp.solvers.GLPK,
              "gurobi": pulp.solvers.GUROBI}[solver_string.lower()]
    solver_instance = solver()
    return solver_instance


def graph_attr_to_dict(graph, attr, array_values=False):
    """
    Parameters
    ----------
    graph : networkx.Graph

    attr : str, iterable, or dict
        If str, then it specifies the an attribute of the graph's nodes.
        If iterable of strings, then multiple attributes of the graph's nodes
        are specified.
        If dict, then each key is a node and each value the corresponding
        attribute value. (This format is also this function's return format.)
    array_values : bool, default: False
        If True, then each value is transformed into a :class:`numpy.ndarray`.

    Returns
    -------
    result_dict : dict
        Each key is a node in the graph.
        If `array_values` is False, then each value is a list of attribute
        values corresponding to the key node.
        If `array_values` is True, then each value this list of attribute
        values is turned into a :class:`numpy.ndarray`. That requires the
        values to be shape-compatible for stacking.
    """
    if isinstance(attr, dict):
        return attr
    if isinstance(attr, str):
        attr = [attr]
    data_dict = {node: [] for node in graph.nodes()}
    for a in attr:
        for node, value in nx.get_node_attributes(graph, a).items():
            data_dict[node].append(value)
    if array_values:
        for node in data_dict:
            data_dict[node] = np.array(data_dict[node])
    return data_dict


def fit_from_geodataframe(instance, areas, data, *args, contiguity="rook",
                          **kwargs):
    """

    Parameters
    ----------
    instance :
        An object offering a `fit_from_geodataframe` and a `fit_from_dict`
        method.
    areas : GeoDataFrame

    data : str or list
        The clustering criteria (columns of the GeoDataFrame `areas`) are
        specified as string (for one column) or list of strings (for
        multiple columns).
    contiguity : {"rook", "queen"}, default: "rook"
        Defines the contiguity relationship between areas. Possible
        contiguity definitions are:

        * "rook" - Rook contiguity.
        * "queen" - Queen contiguity.
    """
    if not isinstance(areas, GeoDataFrame):
        raise ValueError("The areas argument must be a GeoDataFrame.")
    if not isinstance(data, (str, collections.Sequence)):
        raise ValueError("The data argument has to be of one of the "
                         "following types: str or a sequence of strings.")
    if isinstance(data, str):
        data = [data]
    else:  # isinstance(data, collections.Sequence)
        data = list(data)
    values_dict = dataframe_to_dict(areas, data)

    if not isinstance(contiguity, str) or \
            contiguity.lower() not in ["rook", "queen"]:
        raise ValueError("The contiguity argument must be either None "
                         "or one of the following strings: "
                         '"rook" or"queen".')
    if contiguity.lower() == "rook":
        weights = ps.weights.Contiguity.Rook.from_dataframe(areas)
    else:  # contiguity.lower() == "queen"
        weights = ps.weights.Contiguity.Queen.from_dataframe(areas)
    neighbors_dict = weights.neighbors

    instance.fit_from_dict(neighbors_dict, values_dict, *args, **kwargs)


def fit_from_networkx(instance, areas, data, *args, **kwargs):
    """

    Parameters
    ----------
    instance :
        An object offering a `fit_from_networkx` and a `fit_from_dict` method.
    areas : `networkx.Graph`

    data : str, list or dict
        If the clustering criteria are present in the networkx.Graph
        `areas` as node attributes, then they can be specified as a string
        (for one criterion) or as a list of strings (for multiple
        criteria).
        Alternatively, a dict can be used with each key being a node of the
        networkx.Graph `areas` and each value being the corresponding
        clustering criterion (a scalar (e.g. `float` or `int`) or a
        :class:`numpy.ndarray`).
        If there are no clustering criteria are present in the
        networkx.Graph `areas` as node attributes, then a dictionary must
        be used for this argument.
    """
    if not isinstance(areas, nx.Graph):
        raise ValueError("The areas argument must be a networkx.Graph "
                         "object.")
    data_dict = graph_attr_to_dict(areas, data, array_values=True)
    areas = nx.to_dict_of_lists(areas)
    instance.fit_from_dict(areas, data_dict, *args, **kwargs)


def fit_from_w(instance, areas, data, *args, **kwargs):
    """

    Parameters
    ----------
    instance :
        An object offering a `fit_from_w` and a `fit_from_dict` method.
    areas : libpysal.weights.W

    data : dict
        Each key is an area of `areas` and each value represents the
        corresponding value of the clustering criterion. A value can be
        scalar (e.g. `float` or `int`) or a :class:`numpy.ndarray`.
    """
    if not isinstance(areas, ps.weights.W):
        raise ValueError("The areas argument must be a libpysal.weights.W "
                         "object.")
    areas = areas.neighbors
    instance.fit_from_dict(areas, data, *args, **kwargs)
