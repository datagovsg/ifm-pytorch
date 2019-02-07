# coding=utf-8
"""User-User/Item-Item Collaborative filtering module."""
from __future__ import absolute_import, division, print_function

from future.standard_library import install_aliases
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..base import BaseFeature, BaseRecommender, SaveLoadMixin
from ..save_utils import (
    create_directory,
    get_path,
    save_scipy_sparse_csr,
    load_scipy_sparse_csr
)

install_aliases()


class CollaborativeRecommender(BaseFeature, BaseRecommender, SaveLoadMixin):
    """Collaborative filtering recommender system.

    Parameters
    ----------
    collaborative_method : {'user-user', 'item-item'}

    Attributes
    ----------
    components_ : scipy.sparse.csr.csr_matrix, shape = [`n_users`, `n_items`]
        `n_users` refer to the number of users in the training set
        and `n_items` refer to the number of items in the training set
    """
    def __init__(self,
                 collaborative_method):
        if collaborative_method not in {'user-user', 'item-item'}:
            raise ValueError("`collaborative_method` must be in "
                             "{'user-user', 'item-item'}. '%s' "
                             "was given instead." % collaborative_method)
        self.collaborative_method = collaborative_method

    def _fit(self, train_set):
        """Computes the cosine similarity for users.

        Parameters
        ----------
        train_set : UserItemSet

        """
        self._n_users = max(train_set.users_inner) + 1
        self._n_items = max(train_set.items_inner) + 1
        num_rows = self._n_users
        num_cols = self._n_items
        rows = train_set.user_item_selections_inner[:, 0]
        cols = train_set.user_item_selections_inner[:, 1]

        if self.collaborative_method == 'item-item':
            num_rows, num_cols = num_cols, num_rows
            rows, cols = cols, rows

        selections = csr_matrix(
            (train_set.n_selections * [1.0],
             (rows, cols)),
            shape=(num_rows, num_cols)
        )

        similarity = cosine_similarity(selections,
                                       dense_output=False)

        self.components_ = similarity.dot(selections)

        if self.collaborative_method == 'item-item':
            self.components_ = self.components_.transpose().tocsr()

    def fit(self, train_set):
        """Computes the cosine similarity for users.

        Parameters
        ----------
        train_set : UserItemSet

        """
        self._fit(train_set)
        return self

    def _predict_scores(self, users):
        """Returns the scores for the users and corresponding items.

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
        return np.reshape(np.asarray(self.components_[users, items]), -1)

    def save_to_file(self, filename, output_dir=None):
        """Saves data to disk.

        Parameters
        ----------
        filename : str

        output_dir : str, optional (default=None)
        """
        if output_dir:
            create_directory(output_dir)

        # save parameters of CollaborativeRecommender
        sqlite_path = get_path(output_dir, filename, 'sqlite')
        self._save_params(sqlite_path)

        h5_path = get_path(output_dir, filename, 'h5')
        save_scipy_sparse_csr(self.components_,
                              h5_path,
                              self.__class__.__name__)

    @classmethod
    def load_cls_from_file(cls, filename, output_dir=None):
        """Loads CollaborativeRecommender from disk.

        Parameters
        ----------
        filename : str

        output_dir : str, optional (default=None)

        Returns
        -------
        CollaborativeRecommender
        """
        sqlite_path = get_path(output_dir, filename, 'sqlite')
        params = cls._load_params(sqlite_path)
        collaborative_recommender = cls(**params)

        h5_path = get_path(output_dir, filename, 'h5')
        collaborative_recommender.components_ = load_scipy_sparse_csr(
            h5_path, cls.__name__
        )

        return collaborative_recommender

    @property
    def n_users(self):
        """Returns the total number of users."""
        return self.components_.shape[0]

    @property
    def n_items(self):
        """Returns the total number of items."""
        return self.components_.shape[1]
