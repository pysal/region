import networkx as nx
import pytest
from shapely.geometry import Polygon

from .. import util
from ..azp import AZP


def all_elements_equal(iterable):
    return all(iterable[0] == element for element in iterable)


def not_all_elements_equal(iterable):
    return not all_elements_equal(iterable)


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


def test_assert_feasible__pass_disconnected():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])  # not connected
    regions = [{1}, {2}]  # not connected
    try:
        util.assert_feasible(regions, graph)
        util.assert_feasible(regions, graph, n_regions=2)
    except ValueError:
        pytest.fail()


def test_assert_feasible__pass_connected():
    graph = nx.Graph([(1, 2)])  # nodes 1 & 2 connected
    # with one region
    regions = [{1, 2}]
    try:
        util.assert_feasible(regions, graph)
        util.assert_feasible(regions, graph, n_regions=1)
    except ValueError:
        pytest.fail()
    # with two regions
    regions = [{1}, {2}]  # two regions
    try:
        util.assert_feasible(regions, graph)
        util.assert_feasible(regions, graph, n_regions=2)
    except ValueError:
        pytest.fail()

def test_assert_feasible__contiguity():
    with pytest.raises(ValueError) as exc_info:
        graph = nx.Graph()
        graph.add_nodes_from([1, 2])  # not connected
        regions = [{1, 2}]  # connected
        util.assert_feasible(regions, graph)
    assert "not spatially contiguous" in str(exc_info)


def test_assert_feasible__number_of_regions():
    with pytest.raises(ValueError) as exc_info:
        graph = nx.Graph([(1, 2)])  # nodes 1 & 2 connected
        regions = [{1, 2}]  # one region
        n_regions = 2  # let's assume that two regions are required
        util.assert_feasible(regions, graph, n_regions=n_regions)
    assert "The number of regions is" in str(exc_info)


def test_random_element_from():
    lst = list(range(1000))
    n_pops = 5
    popped = []
    for _ in range(n_pops):
        lst_copy = list(lst)
        popped.append(util.random_element_from(lst_copy))
        assert len(lst_copy) == len(lst)
    assert not_all_elements_equal(popped)


def test_pop_randomly_from():
    lst = list(range(1000))
    n_pops = 5
    popped = []
    for _ in range(n_pops):
        lst_copy = list(lst)
        popped.append(util.pop_randomly_from(lst_copy))
        assert len(lst_copy) == len(lst) - 1
    assert not_all_elements_equal(popped)


def test_AZP_azp_connected_component__one_area():
    single_node = nx.Graph()
    single_node.add_node(0)
    azp = AZP(n_regions=1)
    region_list = azp._azp_connected_component(single_node, [{0}])
    assert region_list == [{0}]
