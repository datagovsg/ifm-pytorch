# coding=utf-8
"""Matrix Factorization-Collaborative filtering module."""
from __future__ import absolute_import, division, print_function

from future.standard_library import install_aliases
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from ..base import BaseFeature, BaseRecommender, SaveLoadMixin
from ..save_utils import (
    create_directory,
    get_path,
    save_array,
    load_array
)

install_aliases()


class SVDRecommender(BaseFeature, BaseRecommender, SaveLoadMixin):
    """SVD matrix factorization recommender system.

    Parameters
    ----------
    n_components : int

    n_oversamples : int, optional (default=4)

    n_iter : int, optional (default=2)

    power_iteration_normalizer : str, optional (default='QR')

    Attributes
    ----------
    user_embeddings_ : np.ndarray[float], shape = [`n_users`, `n_components`]
        `n_users` refer to the number of users in the training set

    item_embeddings_ : np.ndarray[float], shape = [`n_items`, `n_components`]
        `n_items` refer to the number of items in the training set
    """
    def __init__(self,
                 n_components,
                 n_oversamples=4,
                 n_iter=2,
                 power_iteration_normalizer='QR'):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.power_iteration_normalizer = power_iteration_normalizer

    def _fit(self, train_set):
        """Computes the cosine similarity for users.

        Parameters
        ----------
        train_set : UserItemSet

        """
        num_rows = max(train_set.users_inner) + 1
        num_cols = max(train_set.items_inner) + 1
        rows = train_set.user_item_selections_inner[:, 0]
        cols = train_set.user_item_selections_inner[:, 1]

        user_item_selections = csr_matrix(
            (train_set.n_selections * [1.0],
             (rows, cols)),
            shape=(num_rows, num_cols)
        )

        U, s, Vh = randomized_svd(
            user_item_selections,
            n_components=self.n_components,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            power_iteration_normalizer=self.power_iteration_normalizer
        )

        self.user_embeddings_ = U * np.sqrt(s)
        self.item_embeddings_ = (Vh.transpose() * np.sqrt(s)).transpose()

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
        return np.reshape(
            self.user_embeddings_[users, :]
                .dot(self.item_embeddings_), -1
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
        return np.sum(
            self.user_embeddings_[users, :]
            * self.item_embeddings_.transpose()[items, :],
            axis=1,
            keepdims=True
        )

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
        save_array(self.user_embeddings_,
                   h5_path,
                   self.__class__.__name__ + '/' + 'user_embeddings')
        save_array(self.item_embeddings_,
                   h5_path,
                   self.__class__.__name__ + '/' + 'item_embeddings',
                   'a')

    @classmethod
    def load_cls_from_file(cls, filename, output_dir=None):
        """Loads SVDRecommender from disk.

        Parameters
        ----------
        filename : str

        output_dir : str, optional (default=None)

        Returns
        -------
        SVDRecommender
        """
        sqlite_path = get_path(output_dir, filename, 'sqlite')
        params = cls._load_params(sqlite_path)
        svd_recommender = cls(**params)

        h5_path = get_path(output_dir, filename, 'h5')
        svd_recommender.user_embeddings_ = load_array(
            h5_path, cls.__name__ + '/' + 'user_embeddings'
        )
        svd_recommender.item_embeddings_ = load_array(
            h5_path, cls.__name__ + '/' + 'item_embeddings'
        )

        return svd_recommender

    @property
    def n_users(self):
        """Returns the total number of users."""
        return self.user_embeddings_.shape[0]

    @property
    def n_items(self):
        """Returns the total number of items."""
        return self.item_embeddings_.shape[1]
