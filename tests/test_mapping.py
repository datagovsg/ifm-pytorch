# coding=utf-8
"""Test mapping module."""
from __future__ import absolute_import, division, print_function
import shutil

from future.standard_library import install_aliases
import numpy as np
import pytest

from iftorch.mapping import RawInnerMap

install_aliases()


def key_error_exception_message(key_error_exception):
    """Extracts the message of a KeyError exception

    Parameters
    ----------
    key_error_exception: KeyError

    Returns
    -------
    str
    """
    return str(key_error_exception)[1:-1]


def test_useritemmap_user_str(test_raw2inner_user_str,
                              test_raw2inner_item_str):
    """Tests functionality of UserItemMap class"""
    user_item_map = RawInnerMap(test_raw2inner_user_str,
                                test_raw2inner_item_str)

    # test whether raw/inner id exists
    assert user_item_map.exists_user_raw('3')
    assert not user_item_map.exists_user_raw('2')

    assert user_item_map.exists_user_inner(5)
    assert not user_item_map.exists_user_inner(4)

    # test to user/users inner methods
    assert user_item_map.to_user_inner('1') == 5
    assert user_item_map.to_user_inner('3') == 10
    with pytest.raises(KeyError) as info:
        user_item_map.to_user_inner('2')
    assert (
        key_error_exception_message(info.value)
        == "User (raw) '2' is unknown."
    )

    assert user_item_map.to_users_inner(['3', '1']) == [10, 5]
    with pytest.raises(KeyError) as info:
        user_item_map.to_users_inner(['3', '2', '1', '5'])
    assert (
        key_error_exception_message(info.value)
        == "Users (raw) ['2', '5'] are unknown."
    )

    np.testing.assert_array_equal(
        user_item_map.to_users_inner(np.asarray(['3', '1'])),
        np.asarray([10, 5])
    )
    with pytest.raises(KeyError) as info:
        user_item_map.to_users_inner(np.asarray(['3', '2', '1', '5']))
    assert (
        key_error_exception_message(info.value)
        == "Users (raw) ['2', '5'] are unknown."
    )

    # test to user/users raw methods
    assert user_item_map.to_user_raw(5) == '1'
    assert user_item_map.to_user_raw(10) == '3'
    with pytest.raises(KeyError) as info:
        user_item_map.to_user_raw(4)
    assert (
        key_error_exception_message(info.value)
        == "User (inner) 4 is unknown."
    )

    assert user_item_map.to_users_raw([10, 5]) == ['3', '1']
    with pytest.raises(KeyError) as info:
        user_item_map.to_users_raw([10, 4, 5, 8])
    assert (
        key_error_exception_message(info.value)
        == "Users (inner) [4, 8] are unknown."
    )

    np.testing.assert_array_equal(
        user_item_map.to_users_raw(np.asarray([10, 5])),
        np.asarray(['3', '1'])
    )
    with pytest.raises(KeyError) as info:
        user_item_map.to_users_raw(np.asarray([10, 4, 5, 8]))
    assert (
        key_error_exception_message(info.value)
        == "Users (inner) [4, 8] are unknown."
    )

    np.testing.assert_array_equal(user_item_map.all_users_raw,
                                  np.asarray(['1', '3']))
    np.testing.assert_array_equal(user_item_map.all_users_inner,
                                  np.asarray([5, 10]))


def test_useritemmap_item_str(test_raw2inner_user_str,
                              test_raw2inner_item_str):
    """Tests functionality for items in UserItemMap class"""
    user_item_map = RawInnerMap(test_raw2inner_user_str,
                                test_raw2inner_item_str)

    # test whether raw/inner id exists
    assert user_item_map.exists_item_raw('5')
    assert not user_item_map.exists_item_raw('3')

    assert user_item_map.exists_item_inner(8)
    assert not user_item_map.exists_item_inner(5)

    # test to item/items inner methods
    assert user_item_map.to_item_inner('2') == 4
    assert user_item_map.to_item_inner('5') == 8
    with pytest.raises(KeyError) as info:
        user_item_map.to_item_inner('1')
    assert (
        key_error_exception_message(info.value)
        == "Item (raw) '1' is unknown."
    )

    assert user_item_map.to_items_inner(['5', '2']) == [8, 4]
    with pytest.raises(KeyError) as info:
        user_item_map.to_items_inner(['3', '2', '1', '5'])
    assert (
        key_error_exception_message(info.value)
        == "Items (raw) ['3', '1'] are unknown."
    )

    np.testing.assert_array_equal(
        user_item_map.to_items_inner(np.asarray(['5', '2'])),
        np.asarray([8, 4])
    )
    with pytest.raises(KeyError) as info:
        user_item_map.to_items_inner(np.asarray(['3', '2', '1', '5']))
    assert (
        key_error_exception_message(info.value)
        == "Items (raw) ['3', '1'] are unknown."
    )

    # test to item/items raw methods
    assert user_item_map.to_item_raw(4) == '2'
    assert user_item_map.to_item_raw(8) == '5'
    with pytest.raises(KeyError) as info:
        user_item_map.to_item_raw(5)
    assert (
        key_error_exception_message(info.value)
        == "Item (inner) 5 is unknown."
    )

    assert user_item_map.to_items_raw([8, 4]) == ['5', '2']
    with pytest.raises(KeyError) as info:
        user_item_map.to_items_raw([8, 10, 4, 5])
    assert (
        key_error_exception_message(info.value)
        == "Items (inner) [10, 5] are unknown."
    )

    np.testing.assert_array_equal(
        user_item_map.to_items_raw(np.asarray([8, 4])),
        np.asarray(['5', '2'])
    )
    with pytest.raises(KeyError) as info:
        user_item_map.to_items_raw(np.asarray([8, 10, 4, 5]))
    assert (
        key_error_exception_message(info.value)
        == "Items (inner) [10, 5] are unknown."
    )

    np.testing.assert_array_equal(user_item_map.all_items_raw,
                                  np.asarray(['2', '5']))
    np.testing.assert_array_equal(user_item_map.all_items_inner,
                                  np.asarray([4, 8]))


def test_useritemmap_save_str(test_raw2inner_user_str,
                              test_raw2inner_item_str):
    """Tests save functionality of UserItemMap class."""
    user_item_map = RawInnerMap(test_raw2inner_user_str,
                                test_raw2inner_item_str)
    user_item_map.save_to_file(name='test_name',
                               output_dir='test_dir')
    user_item_map_reloaded = RawInnerMap.load_cls_from_file(
        name='test_name',
        output_dir='test_dir'
    )
    assert user_item_map == user_item_map_reloaded
    shutil.rmtree('test_dir')


def test_useritemmap_save_int(test_raw2inner_user_int,
                              test_raw2inner_item_int):
    """Tests save functionality of UserItemMap class."""
    user_item_map = RawInnerMap(test_raw2inner_user_int,
                                test_raw2inner_item_int)
    user_item_map.save_to_file(name='test_name',
                               output_dir='test_dir')
    user_item_map_reloaded = RawInnerMap.load_cls_from_file(
        name='test_name',
        output_dir='test_dir'
    )
    assert user_item_map == user_item_map_reloaded
    shutil.rmtree('test_dir')
