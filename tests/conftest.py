# coding=utf-8
"""
Setting up py.text fixtures
"""
from __future__ import absolute_import, division, print_function

from future.standard_library import install_aliases
import pytest

install_aliases()


@pytest.fixture(scope='module')
def test_raw2inner_user_str():
    """Fixture for test raw to inner user data."""
    raw2inner_user = {'1': 5, '3': 10}

    return raw2inner_user


@pytest.fixture(scope='module')
def test_raw2inner_item_str():
    """Fixture for test raw to inner item data."""
    raw2inner_item = {'2': 4, '5': 8}

    return raw2inner_item


@pytest.fixture(scope='module')
def test_raw2inner_user_int():
    """Fixture for test raw to inner user data."""
    raw2inner_user = {1: 5, 3: 10}

    return raw2inner_user


@pytest.fixture(scope='module')
def test_raw2inner_item_int():
    """Fixture for test raw to inner item data."""
    raw2inner_item = {2: 4, 5: 8}

    return raw2inner_item
