import numpy as np
import collections


def dataframe_to_dict(df, cols):
    """

    Parameters
    ----------
    df : `pandas.DataFrame` or `geopandas.GeoDataFrame`
    cols : list
        A list of strings. Each string is the name of a column of `df`.

    Returns
    -------
    result : dict
        The keys are the elements of the DataFrame's index.
        Each value is an `numpy.ndarray` holding the corresponding values in
        the columns specified by `cols`.

    """
    return dict(zip(df.index, np.array(df[cols])))


def find_sublist_containing(el, lst, index=False):
    """

    Parameters
    ----------
    el :
        The element to search for in the sublists of `lst`.
    lst : collections.Sequence
        A sequence of sequences or sets.
    index : bool, default: False
        If False (default), the subsequence or subset containing `el` is
        returned.
        If True, the index of the subsequence or subset in `lst` is returned.

    Returns
    -------
    result : collections.Sequence, collections.Set or int
        See the `index` argument for more information.
    """
    for idx, sublst in enumerate(lst):
        if el in sublst:
            return idx if index else sublst
    raise LookupError(
            "{} not found in any of the sublists of {}".format(el, lst))


def dissim_measure(v1, v2):
    """
    Parameters
    ----------
    v1 : float or ndarray
    v2 : float or ndarray

    Returns
    -------
    result : numpy.float64
        The dissimilarity between the values v1 and v2.
    """
    return np.linalg.norm(v1 - v2)
