# coding=utf-8
"""Base classes"""
from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod

from future.standard_library import install_aliases
import numpy as np
import six
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ._utils import dataloader, recommended_items
from .save_utils import save_mapping, load_mapping

install_aliases()


class SaveLoadMixin(six.with_metaclass(ABCMeta)):
    """An template mixin for saving and loading data.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def save_to_file(self, filename, output_dir=None):
        """Saves data to disk.

        Parameters
        ----------
        filename : str

        output_dir : str, optional
        """
        pass

    # optional method to load data from file
    def load_from_file(self, filename, output_dir=None):
        """Loads data from disk.

        Parameters
        ----------
        filename : str

        output_dir : str, optional
        """
        raise NotImplementedError

    # optional method to load class from file
    @classmethod
    def load_cls_from_file(cls, filename, output_dir=None):
        """Loads data from disk.

        Parameters
        ----------
        filename : str

        output_dir : str, optional

        Returns
        -------
        SaveLoadMixin
        """
        raise NotImplementedError


class BaseFeature(six.with_metaclass(ABCMeta, BaseEstimator)):
    """An abstract class for extracting the features
    for users (inner ids)-items (inner ids).

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def extract(self, users, items):
        """Extracts features corresponding to users-items.

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
        np.ndarray[float] | torch.FloatTensor | torch.cuda.FloatTensor,
            shape = [n_selections, n_features]
        """
        pass

    def reload(self,
               reloaded_type,
               filename,
               output_dir=None,
               save=False):
        """Loads data from disk.

        Parameters
        ----------
        reloaded_type : {'numpy', 'torch', 'torch.cuda', 'h5py'}
            Reloaded type should be one in the list.

        filename : str
            Name of the file.

        output_dir : str, optional (default=None)
            Output directory to load (and save) the data.

        save : bool, optional (default=False)
            Whether to save the array to disk before reloading.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_users(self):
        """Returns the total number of features.

        Returns
        -------
        int
        """
        pass

    @property
    @abstractmethod
    def n_items(self):
        """Returns the total number of items.

        Returns
        -------
        int
        """
        pass


class BaseRecommender(six.with_metaclass(ABCMeta, BaseEstimator)):
    """An abstract class for training an item recommendation model.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    def _save_params(self, path):
        """Save the parameters.

        Parameters
        ----------
        path : str
        """
        save_mapping(self.get_params(),
                     path,
                     self.__class__.__name__)

    @classmethod
    def _load_params(cls, path):
        """Loads the parameters.

        Parameters
        ----------
        path : str

        Returns
        -------
        mapping of string to any
            Parameter names mapped to their values.
        """
        return load_mapping(path, cls.__name__)

    @abstractmethod
    def _fit(self, train_set):
        """Trains an item recommendation model on the training set.

        Parameters
        ----------
        train_set : UserItemSet
        """
        pass

    @abstractmethod
    def _predict_scores(self, users):
        """Returns the scores for the users for all the items.

        Parameters
        ----------
        users : np.ndarray, shape = [len(users),]
            Each row represents a user (inner id)

        Returns
        -------
        scores : np.ndarray[float], shape = [len(users) * total_num_items,]
        """
        pass

    def predict(self,
                test_set,
                train_set,
                remove_users_not_in_train=False,
                max_score_to_filter=None,
                predict_batch_size=100,
                num_recommendations=150):
        """Predicts the top `num_recommendations` for each
        user in test_set.users_inner.

        Parameters
        ----------
        test_set : UserItemSet

        train_set : UserItemSet

        remove_users_not_in_train : bool
            Indicates whether to filter off users who are not in
            the training set

        max_score_to_filter : float, optional (default=None)

        is_matrix_mult : bool

        predict_batch_size : int, optional (default=100)

        num_recommendations : int, optional (default=150)

        Returns
        -------
        user2recommended_items : dict[int, list[int]]
        """
        if 'prefit' in dir(self) and self.prefit:
            raise NotFittedError('No recommender model has been trained.')

        user2recommended_items = {}
        for users in dataloader(dataset=test_set.users_inner,
                                batch_size=predict_batch_size,
                                shuffle=False):
            if remove_users_not_in_train:
                users = [user for user in users
                         if user in set(train_set.users_inner)]

            num_batch_users = len(users)
            if num_batch_users == 0:
                continue

            item_scores = self._predict_scores(users)
            item_scores = np.reshape(item_scores,
                                     (num_batch_users, -1))

            for row, user in enumerate(users):
                user_train_selected_items = set(
                    train_set.user2items_inner[user]
                )

                user2recommended_items[user] = recommended_items(
                    item_scores=item_scores[row, :],
                    items_to_filter=user_train_selected_items,
                    max_score_to_filter=max_score_to_filter,
                    num_recommendations=num_recommendations
                )

        return user2recommended_items
