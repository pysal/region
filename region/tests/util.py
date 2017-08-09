def compare_region_lists(actual, desired):
    """
    Parameters
    ----------
    actual : list
        Every element (of type ``set``) represents a region.
    desired : list
        Every element (of type ``set``) represents a region.

    Raises
    ------
    AssertionError
        If the two arguments don't represent the same clustering.
    """
    # check number of regions
    assert len(actual) == len(desired)
    # check equality of regions
    assert all(region in desired for region in actual)