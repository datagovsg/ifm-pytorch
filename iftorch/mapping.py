# coding=utf-8
"""The mapping module contains the UserItemMap class."""
from __future__ import absolute_import, division, print_function

from future.standard_library import install_aliases
import numpy as np

from ._utils import reverse_mapping
from .save_utils import (
    create_directory,
    get_path,
    save_mapping,
    load_mapping
)
from .base import SaveLoadMixin

install_aliases()


class RawInnerMap(SaveLoadMixin):
    """Contains information about the mapping between user/item raw id to
    user/item inner id.

    Parameters
    ----------
    raw2inner_user : dict[str | int, int]
        Dict mapping the user raw id to the user inner id.

    raw2inner_item : dict[str | int, int]
        Dict mapping the item raw id to the item inner id.
    """
    def __init__(self,
                 raw2inner_user,
                 raw2inner_item):
        self._raw2inner_user = raw2inner_user
        self._inner2raw_user = reverse_mapping(raw2inner_user)

        self._raw2inner_item = raw2inner_item
        self._inner2raw_item = reverse_mapping(raw2inner_item)

    def exists_user_raw(self, user_raw):
        """Checks whether the user (raw id) exists.

        Parameters
        ----------
        user_raw : str | int

        Returns
        -------
        bool
        """
        return user_raw in self._raw2inner_user

    def exists_item_raw(self, item_raw):
        """Checks whether the item (raw id) exists.

        Parameters
        ----------
        item_raw : str | int

        Returns
        -------
        bool
        """
        return item_raw in self._raw2inner_item

    def exists_user_inner(self, user_inner):
        """Checks whether the user (inner id) exists.

        Parameters
        ----------
        user_inner : int

        Returns
        -------
        bool
        """
        return user_inner in self._inner2raw_user

    def exists_item_inner(self, item_inner):
        """Checks whether the item (inner id) exists.

        Parameters
        ----------
        item_inner : int

        Returns
        -------
        bool
        """
        return item_inner in self._inner2raw_item

    def to_user_inner(self, user_raw):
        """Converts a user raw id to the user inner id.

        Parameters
        ----------
        user_raw : str | int

        Returns
        -------
        int
        """
        try:
            return self._raw2inner_user[user_raw]
        except KeyError:
            raise KeyError('User (raw) ' + repr(user_raw) +
                           ' is unknown.')

    def to_users_inner(self, users_raw):
        """Converts user raw ids to user inner ids.

        Parameters
        ----------
        users_raw : list[str | int] | numpy.ndarray[str | int]

        Returns
        -------
        users_inner : list[int] | numpy.ndarray[int]
        """
        try:
            users_inner = [
                self._raw2inner_user[user_raw]
                for user_raw in users_raw
            ]
        except KeyError:
            unknown_users = [
                user_raw for user_raw in users_raw
                if user_raw not in self._raw2inner_user
            ]
            raise KeyError('Users (raw) ' + repr(unknown_users) +
                           ' are unknown.')

        if isinstance(users_raw, np.ndarray):
            users_inner = np.asarray(users_inner)

        return users_inner

    def to_item_inner(self, item_raw):
        """Converts an item raw id to the item inner id.

        Parameters
        ----------
        item_raw : str | int

        Returns
        -------
        int
        """
        try:
            return self._raw2inner_item[item_raw]
        except KeyError:
            raise KeyError('Item (raw) ' + repr(item_raw) +
                           ' is unknown.')

    def to_items_inner(self, items_raw):
        """Converts item raw ids to item inner ids.

        Parameters
        ----------
        items_raw : list[str | int] | numpy.ndarray[str | int]

        Returns
        -------
        items_inner : list[int] | numpy.ndarray[int]
        """
        try:
            items_inner = [
                self._raw2inner_item[item_raw]
                for item_raw in items_raw
            ]
        except KeyError:
            unknown_items = [
                item_raw for item_raw in items_raw
                if item_raw not in self._raw2inner_item
            ]
            raise KeyError('Items (raw) ' + repr(unknown_items) +
                           ' are unknown.')

        if isinstance(items_raw, np.ndarray):
            items_inner = np.asarray(items_inner)

        return items_inner

    def to_user_raw(self, user_inner):
        """Converts a user inner id to the user raw id.

        Parameters
        ----------
        user_inner : int

        Returns
        -------
        str | int
        """
        try:
            return self._inner2raw_user[user_inner]
        except KeyError:
            raise KeyError('User (inner) ' + repr(user_inner) +
                           ' is unknown.')

    def to_users_raw(self, users_inner):
        """Converts inner user ids to raw user ids.

        Parameters
        ----------
        users_inner : list[int] | numpy.ndarray[int]

        Returns
        -------
        users_raw : list[str | int] | numpy.ndarray[str | int]
        """
        try:
            users_raw = [
                self._inner2raw_user[user_inner]
                for user_inner in users_inner
            ]
        except KeyError:
            unknown_users = [
                user_inner for user_inner in users_inner
                if user_inner not in self._inner2raw_user
            ]
            raise KeyError('Users (inner) ' + repr(unknown_users) +
                           ' are unknown.')

        if isinstance(users_inner, np.ndarray):
            users_raw = np.asarray(users_raw)

        return users_raw

    def to_item_raw(self, item_inner):
        """Converts an item inner id to the item raw id.

        Parameters
        ----------
        item_inner : int

        Returns
        -------
        str | int
        """
        try:
            return self._inner2raw_item[item_inner]
        except KeyError:
            raise KeyError('Item (inner) ' + repr(item_inner) +
                           ' is unknown.')

    def to_items_raw(self, items_inner):
        """Converts item inner ids to item raw ids.

        Parameters
        ----------
        items_inner : list[int] | numpy.ndarray[int]

        Returns
        -------
        items_raw : list[str | int] | numpy.ndarray[str | int]
        """
        try:
            items_raw = [
                self._inner2raw_item[item_inner]
                for item_inner in items_inner
            ]
        except KeyError:
            unknown_items = [
                item_inner for item_inner in items_inner
                if item_inner not in self._inner2raw_item
            ]
            raise KeyError('Items (inner) ' + repr(unknown_items) +
                           ' are unknown.')

        if isinstance(items_inner, np.ndarray):
            items_raw = np.asarray(items_raw)

        return items_raw

    def save_to_file(self, name, output_dir=None):
        """Saves the mappings in UserItemMap to disk using sqlitedict.

        Parameters
        ----------
        name : str

        output_dir : str, optional
        """
        if output_dir is not None:
            create_directory(output_dir)

        sqlite_path = get_path(output_dir, name, 'sqlite')

        save_mapping(self._raw2inner_user,
                     sqlite_path,
                     'rawinner_map/raw2inner_user')

        save_mapping(self._raw2inner_item,
                     sqlite_path,
                     'rawinner_map/raw2inner_item')

    @classmethod
    def load_cls_from_file(cls, name, output_dir=None):
        """Loads the raw to inner mappings from disk.

        Parameters
        ----------
        name : str

        output_dir : str, optional

        Returns
        -------
        UserItemMap
        """
        sqlite_path = get_path(output_dir, name, 'sqlite')

        raw2inner_user = load_mapping(sqlite_path,
                                      'rawinner_map/raw2inner_user')

        raw2inner_item = load_mapping(sqlite_path,
                                      'rawinner_map/raw2inner_item')

        return cls(raw2inner_user, raw2inner_item)

    def __eq__(self, other):
        # pylint: disable=protected-access
        return (
            self._raw2inner_user == other._raw2inner_user
            and self._raw2inner_item == other._raw2inner_item
            and self._inner2raw_user == other._inner2raw_user
            and self._inner2raw_item == other._inner2raw_item
        )

    @property
    def all_users_inner(self):
        """Returns the list of sorted user inner ids.

        Returns
        -------
        np.ndarray[int]
        """
        return np.asarray(sorted(self._inner2raw_user.keys()))

    @property
    def all_users_raw(self):
        """Returns the list of sorted user raw ids.

        Returns
        -------
        np.ndarray[int | str]
        """
        return np.asarray(sorted(self._raw2inner_user.keys()))

    @property
    def all_items_inner(self):
        """Returns the list of sorted item inner ids.

        Returns
        -------
        np.ndarray[int]
        """
        return np.asarray(sorted(self._inner2raw_item.keys()))

    @property
    def all_items_raw(self):
        """Returns the list of sorted item raw ids.

        Returns
        -------
        np.ndarray[int | str]
        """
        return np.asarray(sorted(self._raw2inner_item.keys()))
