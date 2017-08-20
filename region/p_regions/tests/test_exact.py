import libpysal as ps
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from region.p_regions.exact import ClusterExact
from region.tests.util import compare_region_lists
from region.util import dict_to_region_list
from region.util import dataframe_to_dict


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

expected_clustering_dict = {area: region
                            for area, region in zip(range(9), [0, 0, 0,
                                                               1, 1, 0,
                                                               1, 1, 1])}
expected_clustering = dict_to_region_list(expected_clustering_dict)


# tests with a GeoDataFrame as areas argument
def test_geodataframe_basic(method):
    cluster_object = ClusterExact(n_regions=2)
    cluster_object.fit_from_geodataframe(gdf_lattice, "values", method=method)
    result = dict_to_region_list(cluster_object.labels_)
    compare_region_lists(result, expected_clustering)


# tests with a dict as areas argument
def test_dict_basic(method):
    rook = ps.weights.Contiguity.Rook.from_dataframe(gdf_lattice)
    areas = rook.neighbors
    value_dict = dataframe_to_dict(gdf_lattice, "values")
    cluster_object = ClusterExact(n_regions=2)
    cluster_object.fit(areas, value_dict, method=method)
    result = dict_to_region_list(cluster_object.labels_)
    compare_region_lists(result, expected_clustering)
