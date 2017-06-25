# encoding: latin2
"""
Test distance functions
"""
__author__ = "Juan C. Duque"
__credits__ = "Copyright (c) 2009-11 Juan C. Duque"
__license__ = "New BSD License"
__version__ = "1.0.0"
__maintainer__ = "RiSE Group"
__email__ = "contacto@rise-group.org"

from clusterpy.core.toolboxes.cluster.componentsAlg import distanceFunctions
from unittest import TestCase, skip

class TestDistanceFunctions(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_square_double_of_many_items_list(self):
        """
        Squares each element of a list and then returns the sum.
        """
        square_double = distanceFunctions.square_double

        input_list = [-4, -3, 0, 1, 2, 3, 0.5]
        expected_out = float(39.25)
        output = square_double(input_list)
        assert expected_out == output

    def test_distance_area_2_area_euclidean_squared(self):
        """
        Test distances between areas using euclidean squared
        Each area is a collection of properties.
        """
        distanceA2AEuclideanSquared = distanceFunctions.distanceA2AEuclideanSquared

        input_list = [[-1.0395419], [-0.5878644], [0.8539247], [1.52013208]]
        expected_out = [[0.20401256400624995],
                        [2.0787558088788094],
                        [0.4438322731664644]]

        output = distanceA2AEuclideanSquared(input_list)
        assert output == expected_out

    def test_distance_area_2_area_euclidean_squared_multivariable(self):
        """
        Test distances between areas using euclidean squared on multiple variables.
        Each area is a collection of properties.
        """
        distanceA2AEuclideanSquared = distanceFunctions.distanceA2AEuclideanSquared

        a, b, c = [1, 2], [2, 3], [3, 4]
        input_list = [a, b, c]
        expected_out = [[2], [2]]
        output = distanceA2AEuclideanSquared(input_list)
        assert output == expected_out

    def test_distance_area_2_area_euclidean_squared_empty(self):
        """
        Test distances between areas using euclidean squared on empty list.
        Each area is a collection of properties.

        Distance between two empty lists is zero, since they are the same.
        """
        distanceA2AEuclideanSquared = distanceFunctions.distanceA2AEuclideanSquared

        input_list = [[], []]
        expected_out = [[0]]
        output = distanceA2AEuclideanSquared(input_list)
        assert output == expected_out

    def test_hamming_distance(self):
        """
        Test Hamming distance for areas. Example taken from the documentation.
        """
        getHammingDistance = distanceFunctions.getHammingDistance

        X = [3, 1, 1, 0, 3, 0, 1, 0, 2, 0, 0, 3, 2, 2, 3, 3]
        Y = [0, 0, 0, 3, 0, 3, 3, 3, 2, 3, 3, 1, 2, 2, 1, 1]
        expected_out = float(0.1875)
        output = getHammingDistance(X, Y)
        assert output == expected_out

    def test_hamming_distance_same_area_distribution(self):
        """
        Test Hamming distance for areas. Same area distribution.
        """
        getHammingDistance = distanceFunctions.getHammingDistance

        X = [1 , 2 , 3 , 4, 0]
        Y = [4 , 1 , 0 , 3, 2]
        expected_out = float(1)
        output = getHammingDistance(X, Y)
        assert output == expected_out

    def test_hamming_distance_empty_area_distribution(self):
        """
        Test Hamming distance for areas with one area empty.
        """
        getHammingDistance = distanceFunctions.getHammingDistance

        X = [1 , 2 , 3 , 4, 0]
        Y = []
        expected_out = float(0)
        output = getHammingDistance(X, Y)
        assert output == expected_out

    def test_hamming_distance_completely_different_area_distribution(self):
        """
        Test Hamming distance for completely different areas.

        No matter how different the areas are, the minimum value that the
        function could yield given two non-empty lists is 1/len(area). Or
        the match of the first area.
        """
        getHammingDistance = distanceFunctions.getHammingDistance

        X = [0, 1, 1, 2, 2, 2]
        Y = [1, 1, 1, 2, 2, 2]

        expected_out = float(1.0/len(X))
        output = getHammingDistance(X, Y)
        assert output == expected_out

    def test_hamming_distance_two_empty_areas_distribution(self):
        """
        Test Hamming distance for areas with two areas empty.
        """
        getHammingDistance = distanceFunctions.getHammingDistance

        X = []
        Y = []
        self.assertRaises(ZeroDivisionError,
                          getHammingDistance, X, Y)

    @skip
    def test_distance_area_2_area_hausdorff(self):
        """
        Test Hausdorff distance between areas.
        """
        assert False
