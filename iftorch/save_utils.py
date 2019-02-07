# coding=utf-8
"""The _utils module contains various utility functions."""
from __future__ import absolute_import, division, print_function
import errno
import json
import os

from future.standard_library import install_aliases
import h5py
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
import six
from sqlitedict import SqliteDict
import torch

install_aliases()


def get_path(output_dir, name, extension):
    """Returns the path given the output_dir, name and
    extension.

    Parameters
    ----------
    output_dir : str

    name : str

    extension : str

    Returns
    -------
    path : str
    """
    if output_dir is not None:
        path = str(
            Path(Path(output_dir) / (name + '.' + extension))
        )
    else:
        path = name + '.' + extension

    return path


def create_directory(directory):
    """Creates the directory.

    Parameters
    ----------
    directory : str
    """
    if directory is not None:
        try:
            os.makedirs(directory)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise err


def save_mapping(mapping, path, mapping_name):
    """Save mapping to sqlite.

    Parameters
    ----------
    mapping : dict[str | int, str | int | list[int]]

    path : str

    mapping_name : str
    """
    try:
        # key is not encoded in sqlitedict
        mapping_json = {
            json.dumps(key): value
            for key, value in six.iteritems(mapping)
        }
        sqlite = SqliteDict(path,
                            tablename=mapping_name,
                            autocommit=False,
                            encode=json.dumps,
                            flag='w')
        sqlite.update(items=mapping_json)
        sqlite.commit()
    finally:
        sqlite.close()


def load_mapping(path, mapping_name):
    """Loads mapping from sqlite.

    Parameters
    ----------
    path : str

    mapping_name : str

    Returns
    -------
    dict[str | int, str | int]
    """
    sqlite_dict = SqliteDict(path,
                             tablename=mapping_name,
                             autocommit=False,
                             decode=json.loads,
                             flag='r')
    # key is not encoded in sqlitedict
    return {
        json.loads(key): sqlite_dict[key]
        for key in sqlite_dict
    }


def save_array(array, path, array_name, mode='w'):
    """Saves the array to disk.

    Parameters
    ----------
    array : np.ndarray

    path : str

    array_name : str

    mode : str
    """
    with h5py.File(path, mode) as h5fh:
        h5fh.create_dataset(array_name,
                            data=array,
                            track_times=False)


def load_array(path, array_name):
    """Loads the array from disk.

    Parameters
    ----------
    path : str

    array_name : str

    Returns
    -------
    np.ndarray
    """
    with h5py.File(path, mode='r') as h5fh:
        return h5fh[array_name].value


def save_scipy_sparse_csr(csr_array, path, csr_array_name):
    """Saves a scipy sparse csr matrix.

    Parameters
    ----------
    csr_array : scipy.sparse.csr.csr_matrix

    path : str

    csr_array_name : str
    """
    save_array(csr_array.data, path, csr_array_name + '/data')
    save_array(csr_array.indices, path, csr_array_name + '/indices', 'a')
    save_array(csr_array.indptr, path, csr_array_name + '/indptr', 'a')
    save_array(np.array(csr_array.shape),
               path,
               csr_array_name + '/shape',
               'a')


def load_scipy_sparse_csr(path, csr_array_name):
    """Saves a scipy sparse csr matrix.

    Parameters
    ----------
    path : str

    csr_array_name : str

    Returns
    -------
    csr_array : scipy.sparse.csr.csr_matrix
    """
    data = load_array(path, csr_array_name + '/data')
    indices = load_array(path, csr_array_name + '/indices')
    indptr = load_array(path, csr_array_name + '/indptr')
    shape = load_array(path, csr_array_name + '/shape')

    return csr_matrix((data, indices, indptr), shape=tuple(shape))


def reload_array(reloaded_type, path, array_name):
    """Reloads the array from disk and returns
    the array as numpy.ndarray[float] or
    torch.cuda.FloatTensor or as a h5py array.

    Parameters
    ----------
    reloaded_type : {'numpy', 'torch', 'torch.cuda', 'h5py'}

    path : str

    array_name : str
    """
    h5fh = h5py.File(path, mode='r')
    h5_array = h5fh[array_name]
    if reloaded_type == 'numpy':
        np_array = h5_array.value
        h5fh.close()
        return np_array
    elif reloaded_type == 'torch.cuda':
        try:
            torch_cuda_array = torch.cuda.FloatTensor(h5_array.value)
        except Exception:
            raise ValueError("GPU is not available.")
        finally:
            h5fh.close()
        return torch_cuda_array
    elif reloaded_type == 'torch':
        torch_array = torch.FloatTensor(h5_array.value)
        h5fh.close()
        return torch_array
    elif reloaded_type == 'h5py':
        return h5_array
    else:
        raise ValueError(
            "Option '%s' not in {'numpy', 'torch', 'torch.cuda', 'h5py'}."
            % reloaded_type
        )
