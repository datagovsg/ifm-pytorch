"""Popularity module."""
import collections

import numpy as np

from iftorch.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommender system based on item's popularity amongst users.

    Attributes
    ----------
    components_ : collections.Counter
                  Number of occurrence of each item
    """
    def __init__(self,
                 user_grouping=None,
                 item_grouping=None):
        """
        Parameters
        ----------
        user_grouping : Mapping from user inner id to group
        item_grouping : Mapping from item inner id to group
        """
        self.user_grouping = user_grouping
        self.item_grouping = item_grouping

    def _fit(self, train_set):
        """Computes the number of occurrence of each item.

        Parameters
        ----------
        train_set : UserItemSet
        """
        self.n_items = max(train_set.items_inner) + 1

        items = [i[1] for i in train_set.user_item_selections_inner]
        self.components_ = collections.Counter(items)

    def fit(self, train_set):
        """Computes the number of occurrence of each item.

        Parameters
        ----------
        train_set : UserItemSet
        """
        self._fit(train_set)
        return self

    def _predict_scores(self, users):
        """Returns the popularity scores of each item.

        Parameters
        ----------
        users : np.ndarray, shape = [len(users),]
            Each row represents a user (inner id)

        Returns
        -------
        np.ndarray[float], shape = [len(users),]
        """
        return self.extract(
            users=np.repeat(users, self.n_items),
            items=np.asarray(len(users) * list(range(self.n_items)))
        )

    def extract(self, users, items):
        """Extracts the scores/features for users and
        the corresponding selected items.

        Parameters
        ----------
        users : np.ndarray, shape = [n_selections,]
            Each row represents a user (inner id)
        items : np.ndarray, shape = [n_selections,]
            Each row represents an item (inner id)
            that the user in the corresponding row in users
            chose

        Returns
        -------
        np.ndarray[float], shape = [n_selections, n_features]
        """
        if self.user_grouping and self.item_grouping:
            user_groups = [self.user_grouping[u]
                           if u in self.user_grouping.keys()
                           else np.nan
                           for u in users]

            item_groups = [self.item_grouping[i]
                           if i in self.item_grouping.keys()
                           else np.nan
                           for i in items]
        else:
            user_groups = [-1] * len(users)
            item_groups = [-1] * len(items)

        user_item_group_matches = [user_groups[i] == item_groups[i]
                                   for i, _ in enumerate(user_groups)]

        score = [self.components_[items[i]]
                 if match
                 else 0
                 for i, match in enumerate(user_item_group_matches)]

        return score
