from joblib.hashing import NumpyHasher
from joblib.memory import Memory
import numpy as np
from joblib.const_ndarray import const_ndarray
import pytest

def f(x):
    return 2 * x


def test_const_ndarray_init_from_mutable():
    arr = np.array([0, 1, 2, 3, 4])
    assert np.all(arr == const_ndarray(arr))


def test_const_ndarray_is_immutable():
    arr = const_ndarray(np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        arr[0] = 1


def test_const_ndarray_has_hash():
    arr = np.array([0, 1, 2])
    arr = const_ndarray(arr)

    assert arr.joblib_hash is not None


def test_getbuffer_not_called_on_const_ndarray_in_hash(mocker):
    arr = np.array([0, 1, 2])
    arr = const_ndarray(arr)

    hasher = NumpyHasher(hash_name='md5', coerce_mmap=False)
    mock__getbuffer = mocker.patch.object(hasher, '_getbuffer')
    hasher.hash(arr)
    assert len(mock__getbuffer.call_args_list) == 0


def test_const_ndarray_as_cached_func_arg(tmpdir):
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    f_cache = memory.cache(f)
    arr = const_ndarray(np.array([0, 1, 2]))
    result = f_cache(arr) == f(arr)
    assert np.any(result)
    result = f_cache(arr) == f(arr)
    assert np.any(result)
