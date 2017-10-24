import numpy as np
from .hashing import hash


def make_const(array):
    """
    Takes an ndarray as input and converts it to a const_ndarray instance
    in place. See const_ndarray.
    :param array: An ndarray instance to convert.
    :return: The converted array.
    """
    array.__class__ = const_ndarray
    array.__joblib_hash = hash(array)
    array.flags.writeable = False
    return array


class const_ndarray(np.ndarray):
    """
    Represents an immutable ndarray that can be uniquely identified and
    compared by its joblib_hash property. Joblib returns ndarray instances
    with this class if a Memory object is instantiated with immutable=True.
    It is recommended to pass const_ndarray instances into memoized functions
    rather than mutable ndarrays. Joblib will just read the joblib_hash field
    rather than computing the hash value.
    """
    def __init__(self, *args, **kwargs):
        np.ndarray.__init__(self, *args, **kwargs)
        self.__joblib_hash = None
        make_const(self)

    @property
    def joblib_hash(self):
        return self.__joblib_hash
