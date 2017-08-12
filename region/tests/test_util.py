import networkx as nx
import pytest
from shapely.geometry import Polygon

from .. import util
from ..azp import AZP


def test_distribute_regions_among_components__one_area():
    single_node = nx.Graph()
    single_node.add_node(0)
    n_regions = 1
    result = util.distribute_regions_among_components(n_regions=n_regions,
                                                      graph=single_node)
    assert type(result) == dict
    assert next(iter(result.values())) == n_regions


def test_distribute_regions_among_components__no_areas():
    with pytest.raises(ValueError) as exc_info:
        no_node = nx.Graph()
        n_regions = 1
        util.distribute_regions_among_components(n_regions=n_regions,
                                                 graph=no_node)
    assert str(exc_info.value) == "There must be at least one area."


def test_AZP_azp_connected_component__one_area():
    single_node = nx.Graph()
    single_node.add_node(0)
    azp = AZP(n_regions=1)
    region_list = azp._azp_connected_component(single_node, [{0}])
    assert region_list == [{0}]
