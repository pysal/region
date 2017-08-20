import libpysal as ps
import networkx as nx
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from region.max_p_regions.exact import MaxPExact
from region.tests.util import compare_region_lists
from region.util import dict_to_region_list, dataframe_to_dict

value_list = [350.2, 400.5, 430.8,
              490.4, 410.9, 450.4,
              560.1, 500.7, 498.6]
spatially_extensive_attr_list = [30, 25, 31,
                                 28, 32, 30,
                                 35, 27, 33]
threshold = 120

attr_str = "attr"
spatially_extensive_attr_str = "spatially_extensive_attr"

gdf_lattice = GeoDataFrame(
        {attr_str: value_list,
         spatially_extensive_attr_str: spatially_extensive_attr_list},
        geometry=[Polygon([(x, y),
                           (x, y+1),
                           (x+1, y+1),
                           (x+1, y)]) for y in range(3) for x in range(3)]
)

expected_clustering_dict = {area: region
                            for area, region in zip(range(9), [0, 0, 0,
                                                               1, 0, 0,
                                                               1, 1, 1])}
expected_clustering = dict_to_region_list(expected_clustering_dict)


# tests with a GeoDataFrame
def test_geodataframe_basic():
    cluster_object = MaxPExact()

    cluster_object.fit_from_geodataframe(gdf_lattice, attr_str,
                                         spatially_extensive_attr_str,
                                         threshold=threshold)
    result = dict_to_region_list(cluster_object.labels_)
    compare_region_lists(result, expected_clustering)


# tests with a dict as areas argument
def test_dict_basic():
    rook = ps.weights.Contiguity.Rook.from_dataframe(gdf_lattice)
    areas = rook.neighbors
    attr_dict = dataframe_to_dict(gdf_lattice, attr_str)
    spatially_extensive_attr_dict = dataframe_to_dict(
            gdf_lattice, spatially_extensive_attr_str)
    cluster_object = MaxPExact()
    cluster_object.fit(areas, attr_dict, spatially_extensive_attr_dict,
                       threshold=threshold)
    result = dict_to_region_list(cluster_object.labels_)
    compare_region_lists(result, expected_clustering)


# tests with Graph
# ... with dicts as attr and spatially_extensive_attr
def test_graph_dict_basic():
    rook = ps.weights.Contiguity.Rook.from_dataframe(gdf_lattice)
    areas = rook.to_networkx()
    attr_dict = dataframe_to_dict(gdf_lattice, attr_str)
    spatially_extensive_attr_dict = dataframe_to_dict(
            gdf_lattice, spatially_extensive_attr_str)
    cluster_object = MaxPExact()
    cluster_object.fit_from_networkx(areas, attr_dict,
                                     spatially_extensive_attr_dict,
                                     threshold=threshold)
    result = dict_to_region_list(cluster_object.labels_)
    compare_region_lists(result, expected_clustering)


# ... with strings as attr and spatially_extensive_attr
def test_graph_str_basic():
    rook = ps.weights.Contiguity.Rook.from_dataframe(gdf_lattice)
    areas = rook.to_networkx()
    nx.set_node_attributes(areas, attr_str,
                           dataframe_to_dict(gdf_lattice, attr_str))
    nx.set_node_attributes(areas, spatially_extensive_attr_str,
                           dataframe_to_dict(gdf_lattice,
                                             spatially_extensive_attr_str))
    cluster_object = MaxPExact()
    cluster_object.fit_from_networkx(areas, attr_str,
                                     spatially_extensive_attr_str,
                                     threshold=threshold)
    result = dict_to_region_list(cluster_object.labels_)
    compare_region_lists(result, expected_clustering)


# test with W
def test_w_basic():
    areas = ps.weights.Contiguity.Rook.from_dataframe(gdf_lattice)
    attr_dict = dataframe_to_dict(gdf_lattice, attr_str)
    spatially_extensive_attr_dict = dataframe_to_dict(
            gdf_lattice, spatially_extensive_attr_str)
    cluster_object = MaxPExact()
    cluster_object.fit_from_w(areas, attr_dict, spatially_extensive_attr_dict,
                       threshold=threshold)
    result = dict_to_region_list(cluster_object.labels_)
    compare_region_lists(result, expected_clustering)

# todo: test numpy arrays for attr as well as spatially_extensive_attr & threshold
