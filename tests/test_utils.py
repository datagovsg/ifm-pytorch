# coding=utf-8
"""Test _utils module."""
from __future__ import absolute_import, division, print_function
import os
import shutil

from future.standard_library import install_aliases
import numpy as np
import pytest
import torch

from iftorch._utils import dataloader, reverse_mapping

from iftorch.save_utils import (
    get_path,
    create_directory,
    save_mapping,
    load_mapping,
    save_array,
    load_array,
    reload_array
)

install_aliases()


def test_dataloader():
    """Tests dataloader."""
    dataset = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

    # With no shuffling
    results = list(dataloader(dataset, 2, shuffle=False))
    np.testing.assert_array_equal(results[0], np.array([[1, 1], [2, 2]]))
    np.testing.assert_array_equal(results[1], np.array([[3, 3], [4, 4]]))

    results = list(dataloader(dataset, 3, shuffle=False))
    np.testing.assert_array_equal(
        results[0],
        np.array([[1, 1], [2, 2], [3, 3]])
    )
    np.testing.assert_array_equal(results[1], np.array([[4, 4]]))

    # With shuffling
    np.random.seed(12345)
    results = list(dataloader(dataset, 3, shuffle=True))

    np.random.seed(12345)
    indices = list(range(4))
    np.random.shuffle(indices)

    np.testing.assert_array_equal(
        results[0],
        np.array([2 * [index + 1] for index in indices[:3]])
    )
    np.testing.assert_array_equal(
        results[1], np.array([2 * [indices[3] + 1]])
    )


def test_reverse_mapping():
    """Tests _reverse_mapping."""
    original_dict = {'12': 34, '56': 78}
    reversed_dict = {34: '12', 78: '56'}

    assert reversed_dict == reverse_mapping(original_dict)


def test_create_directory():
    """Tests create_directory."""
    create_directory('test_dir')
    assert os.path.isdir('test_dir')
    create_directory('test_dir/test_subdir')
    assert os.path.isdir('test_dir/test_subdir')
    create_directory('test_dir/test_subdir')
    shutil.rmtree('test_dir')


def test_get_path():
    """Tests get_path."""
    assert (
        get_path('test_dir/test_subdir', 'test_file', 'txt')
        == 'test_dir/test_subdir/test_file.txt'
    )

    assert get_path(None, 'test_file', 'txt') == 'test_file.txt'


def test_save_load_mapping():
    """Test save_mapping and load_mapping."""
    create_directory('test_dir')
    sqlite_path = 'test_dir/test.sqlite'
    mapping = {1: 3, '2': '4', 3: '5', '4': 7}
    save_mapping(mapping, sqlite_path, 'test')
    reloaded_mapping = load_mapping(sqlite_path, 'test')
    assert mapping == reloaded_mapping
    shutil.rmtree('test_dir')


def test_save_load_array():
    """Test save_array and load_array."""
    create_directory('test_dir')
    h5_path = 'test_dir/test.h5'
    array = np.asarray([1.0, 2.0, 3.0])
    save_array(array, h5_path, 'test')
    reloaded_array = load_array(h5_path, 'test')
    np.testing.assert_array_equal(array, reloaded_array)
    shutil.rmtree('test_dir')


def test_reload_array():
    """Test reload_array."""
    create_directory('test_dir')
    h5_path = 'test_dir/test.h5'
    array = np.asarray([1.0, 2.0, 3.0])
    save_array(array, h5_path, 'array')

    reloaded_array = reload_array('numpy', h5_path, 'array')
    np.testing.assert_array_equal(array, reloaded_array)

    reloaded_array = reload_array('h5py', h5_path, 'array')
    np.testing.assert_array_equal(array[0:2], reloaded_array[0:2])
    np.testing.assert_array_equal(array, reloaded_array[0:3])

    reloaded_array = reload_array('torch', h5_path, 'array')
    assert isinstance(reloaded_array, torch.FloatTensor)
    np.testing.assert_array_equal(array,
                                  reloaded_array.cpu().detach().numpy())

    with pytest.raises(ValueError) as info:
        reload_array('wrong_type', h5_path, 'array')
    assert (
        str(info.value)
        == "Option 'wrong_type' not in "
           "{'numpy', 'torch', 'torch.cuda', 'h5py'}."
    )

    shutil.rmtree('test_dir')
