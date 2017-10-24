import numpy as np
from .hashing import hash


class const_ndarray(np.ndarray):
    """
    Represents an immutable ndarray that can be uniquely identified and
    compared by its joblib_hash property. Joblib returns ndarray instances
    with this class if a Memory object is instantiated with immutable=True.
    It is recommended to pass const_ndarray instances into memoized functions
    rather than mutable ndarrays. Joblib will just read the joblib_hash field
    rather than computing the hash value.
    """
    def __new__(cls, obj):
        return obj.view(cls)

    def __init__(self, *args, **kwargs):
        super(const_ndarray, self).__init__()
        # The hasher looks for the __joblib_hash property to decide whether it
        # should be computed.
        self.__joblib_hash = None
        h = hash(self)
        self.__joblib_hash = h
        self.flags.writeable = False

    @property
    def joblib_hash(self):
        return self.__joblib_hash


