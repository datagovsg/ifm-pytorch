# coding=utf-8
"""Tests the dataset module."""
from __future__ import absolute_import, division, print_function
import shutil

from future.standard_library import install_aliases
import numpy as np

from iftorch.mapping import RawInnerMap
from iftorch.dataset import UserItemSet

install_aliases()


def test_user_item_set(test_raw2inner_user_str, test_raw2inner_item_str):
    """Test UserItemSet class."""
    raw_inner_map = RawInnerMap(test_raw2inner_user_str,
                                test_raw2inner_item_str)
    users_raw = np.asarray(['1', '3', '1'])
    items_raw = np.asarray(['2', '5', '2'])

    user_item_set = UserItemSet.load_cls_from_raw(users_raw,
                                                  items_raw,
                                                  raw_inner_map)
    np.testing.assert_array_equal(
        user_item_set.user_item_selections_inner,
        np.asarray([[5, 4], [10, 8], [5, 4]])
    )
    assert user_item_set.user2items_inner == {5: [4, 4], 10: [8]}
    assert user_item_set.n_selections == 3

    np.testing.assert_array_equal(user_item_set.users_inner,
                                  np.asarray([5, 10]))
    np.testing.assert_array_equal(user_item_set.items_inner,
                                  np.asarray([4, 8]))

    user_item_set.save_to_file('test_name', 'test_dir')
    user_item_set_reloaded = UserItemSet.load_cls_from_file('test_name',
                                                            'test_dir')
    assert user_item_set == user_item_set_reloaded
    shutil.rmtree('test_dir')
