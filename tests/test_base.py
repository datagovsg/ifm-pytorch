# coding=utf-8
"""Test base module."""
from __future__ import absolute_import, division, print_function
import shutil

from future.standard_library import install_aliases
import h5py
import numpy as np
import torch

from iftorch.save_utils import (
    create_directory,
    get_path,
    save_array,
    load_array,
    reload_array
)

from iftorch.base import BaseFeature, SaveLoadMixin

install_aliases()


class Feature(BaseFeature, SaveLoadMixin):
    def __init__(self, data):
        self._data = data

    def save_to_file(self, filename, output_dir=None):
        if output_dir is not None:
            create_directory(output_dir)

        h5_path = get_path(output_dir, filename, 'h5')
        save_array(self._data, h5_path, 'array')

    @classmethod
    def load_cls_from_file(cls, filename, output_dir=None):
        h5_path = get_path(output_dir, filename, 'h5')
        data = load_array(h5_path, 'array')
        return cls(data)

    def extract(self, users, items):
        return self._data[users, items]

    def reload(self,
               reloaded_type,
               filename,
               output_dir=None,
               save=False):
        if save:
            self.save_to_file(filename, output_dir)

        h5_path = get_path(output_dir, filename, 'h5')
        self._data = reload_array(reloaded_type, h5_path, 'array')

    @property
    def data(self):
        return self._data

    def n_users(self):
        return self._data.shape[0]

    def n_items(self):
        return self._data.shape[1]


def test_save_load_reload():
    test_features = Feature(
        np.asarray([[10.0, 9.0, 8.0],
                    [7.0, 6.0, 5.0],
                    [4.0, 3.0, 2.0]])
    )
    users = np.asarray([0, 1, 2])
    items = np.asarray([1, 2, 0])
    extracted_features = np.asarray([9.0, 5.0, 4.0])

    test_features.save_to_file('test', 'test_dir')
    reloaded_test_features = Feature.load_cls_from_file('test', 'test_dir')
    np.testing.assert_array_equal(test_features.data,
                                  reloaded_test_features.data)

    test_features.reload('numpy', 'test', 'array', 'test_dir')
    assert isinstance(test_features.data, np.ndarray)
    np.testing.assert_array_equal(test_features.data,
                                  reloaded_test_features.data)
    np.testing.assert_array_equal(test_features.extract(users, items),
                                  extracted_features)

    test_features.reload('torch', 'test', 'array', 'test_dir')
    assert isinstance(test_features.data, torch.FloatTensor)
    np.testing.assert_array_equal(test_features.data.cpu().detach().numpy(),
                                  reloaded_test_features.data)

    test_features.reload('h5py', 'test', 'array', 'test_dir')
    print(type(test_features.data))
    assert isinstance(test_features.data, h5py._hl.dataset.Dataset)
    np.testing.assert_array_equal(test_features.data.value,
                                  reloaded_test_features.data)
    shutil.rmtree('test_dir')
