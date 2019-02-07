# coding=utf-8
"""The dataset module contains the dataloader method and the UserItemSet
class.
"""
from __future__ import absolute_import, division, print_function
from itertools import groupby
import operator

from future.standard_library import install_aliases
import numpy as np

from .base import SaveLoadMixin
from .save_utils import (
    get_path,
    create_directory,
    save_array,
    load_array
)

install_aliases()


class UserItemSet(SaveLoadMixin):
    """Stores information about users and the items that they selected.

    Parameters
    ----------
    user_item_selections_inner : np.ndarray, shape (n_selections,)
        Each row represents a user (inner id) and a corresponding item
        (inner id) that he selected.
    """
    def __init__(self, user_item_selections_inner):
        self._user_item_selections_inner = user_item_selections_inner
        self._user2items_inner = {}

        sorted_rows = user_item_selections_inner[:, 0].argsort()
        grouped_user_items_iter = groupby(
            user_item_selections_inner[sorted_rows],
            key=operator.itemgetter(0)
        )
        for user_inner, ui_iter in grouped_user_items_iter:
            self._user2items_inner[int(user_inner)] = sorted(
                [int(item_inner) for _, item_inner in ui_iter]
            )  # ml_metrics.apk expects a list not a numpy array

    def save_to_file(self, name, output_dir=None):
        """Saves data to disk.

        Parameters
        ----------
        name : str

        output_dir : str, optional
        """
        if output_dir:
            create_directory(output_dir)
        array_h5_path = get_path(output_dir, name, 'h5')
        save_array(array=self.user_item_selections_inner,
                   path=array_h5_path,
                   array_name='user_item_selections_inner')

    @classmethod
    def load_cls_from_file(cls, name, output_dir=None):
        """Loads data from disk.

        Parameters
        ----------
        name : str

        output_dir : str, optional

        Returns
        -------
        UserItemSet
        """
        array_h5_path = get_path(output_dir, name, 'h5')
        user_item_selections_inner = load_array(
            path=array_h5_path,
            array_name='user_item_selections_inner'
        )

        return cls(user_item_selections_inner)

    @classmethod
    def load_cls_from_raw(cls, users_raw, items_raw, raw_inner_map):
        """Loads the UserItemSet class from raw ids.

        Parameters
        ----------
        users_raw : np.ndarray[str | int], shape (n_selections,)
            Each row represents a user (raw id)

        items_raw : np.ndarray[str | int], shape (n_selections,)
            Each row represents a corresponding item (raw id) that the
            user in users_raw selected.

        raw_inner_map : RawInnerMap

        Returns
        -------
        UserItemSet
        """
        users_inner = raw_inner_map.to_users_inner(users_raw)
        items_inner = raw_inner_map.to_items_inner(items_raw)
        user_item_selections_inner = np.asanyarray(
            [users_inner, items_inner], dtype=np.int64
        ).transpose()

        return cls(user_item_selections_inner)

    def __eq__(self, other):
        """Compares whether two UserItemSet are equal."""
        return np.array_equal(self.user_item_selections_inner,
                              other.user_item_selections_inner)

    @property
    def users_inner(self):
        """Returns a sorted list of the user inner ids.

        Returns
        -------
        np.ndarray[int]
        """
        return np.asarray(sorted(self.user2items_inner.keys()))

    @property
    def items_inner(self):
        """Returns a sorted list of the item inner ids.

        Returns
        -------
        np.ndarray[int]
        """
        return np.asarray(sorted(set(self._user_item_selections_inner[:, 1])))

    @property
    def user_item_selections_inner(self):
        """Returns the underlying user_item_selections_inner

        Returns
        -------
        np.ndarray[np.int64], shape = [n_selections, 2]
        """
        return self._user_item_selections_inner

    @property
    def user2items_inner(self):
        """Returns the underlying user2items_inner

        Returns
        -------
        dict[int, list[int]]
        """
        return self._user2items_inner

    @property
    def n_selections(self):
        """Returns the total number of selections

        Returns
        -------
        int
        """
        return self.user_item_selections_inner.shape[0]
