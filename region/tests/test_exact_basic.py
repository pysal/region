import numpy as np
from pandas.core.series import Series
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
import libpysal as ps
import pytest

# TODO: CHANGE THIS FILE ACCORDING TO THE NEW API OF exact_algorithms.py!
from ..exact_algorithms import cluster_exact


@pytest.fixture(params=["flow", "order", "tree"])
def method(request):
    return request.param


value_list = [726.7, 623.6, 487.3,
              200.4, 245.0, 481.0,
              170.9, 225.9, 226.9]

gdf_lattice = GeoDataFrame(
        {"values": value_list},
        geometry=[Polygon([(x, y),
                           (x, y+1),
                           (x+1, y+1),
                           (x+1, y)]) for y in range(3) for x in range(3)]
)

expected_clustering1 = [0, 0, 0,
                        1, 1, 0,
                        1, 1, 1]

expected_clustering2 = [1, 1, 1,
                        0, 0, 1,
                        0, 0, 0]


# tests with a GeoDataFrame as areas argument
def test_geodataframe_basic(method):
    result = cluster_exact(gdf_lattice, "values", num_regions=2, method=method)
    assert type(result) is Series
    result = list(result)
    assert result == expected_clustering1 or result == expected_clustering2


# tests with a dict as areas argument
def test_dict_basic(method):
    rook = ps.weights.Contiguity.Rook.from_dataframe(gdf_lattice)
    areas = rook.neighbors
    value_dict = dict(zip(gdf_lattice.index,
                          np.array(gdf_lattice[["values"]])))
    result = cluster_exact(areas, value_dict, num_regions=2, method=method)
    assert type(result) is dict
    result = [result[i] for i in range(9)]
    assert result == expected_clustering1 or result == expected_clustering2
