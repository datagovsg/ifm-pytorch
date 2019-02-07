# coding=utf-8
"""User/Item embeddings feature module."""
from __future__ import absolute_import, division, print_function

from future.standard_library import install_aliases
import torch

from ..base import BaseFeature, SaveLoadMixin
from ..save_utils import (
    create_directory,
    get_path,
    save_array,
    load_array
)

install_aliases()


class UserItemEmbeddingsFeature(BaseFeature, SaveLoadMixin):
    """Extracts features to feed into torch.nn.Module.

    Parameters
    ----------
    trainable : bool

    user_embeddings : torch.nn.modules.sparse.Embedding

    item_embeddings : torch.nn.modules.sparse.Embedding

    is_cuda : bool, optional (default=None)
    """

    def __init__(self,
                 user_embeddings,
                 item_embeddings,
                 trainable=True,
                 is_cuda=None):
        self.trainable = trainable

        if is_cuda is None:
            self.is_cuda = torch.cuda.is_available()
        else:
            if is_cuda and not torch.cuda.is_available():
                raise ValueError("GPU is not available.")
            self.is_cuda = is_cuda

        if self.is_cuda:
            self._torch = torch.cuda
        else:
            self._torch = torch

        self.user_embeddings = torch.nn.Embedding.from_pretrained(
            self._torch.FloatTensor(user_embeddings),
            freeze=(not self.trainable)
        )
        self.item_embeddings = torch.nn.Embedding.from_pretrained(
            self._torch.FloatTensor(item_embeddings),
            freeze=(not self.trainable)
        )

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
        torch.FloatTensor | torch.cuda.FloatTensor,
            shape = [n_selections, n_features]
        """
        users_tensor = self._torch.LongTensor(users)
        items_tensor = self._torch.LongTensor(items)

        user_embeddings = self.user_embeddings(users_tensor)
        item_embeddings = self.item_embeddings(items_tensor)

        return user_embeddings, item_embeddings

    def save_to_file(self, filename, output_dir=None):
        """Saves data to disk.

        Parameters
        ----------
        filename : str

        output_dir : str, optional (default=None)
        """
        if output_dir:
            create_directory(output_dir)

        # save weights of UserItemEmbeddingsFeature
        h5_path = get_path(output_dir, filename, 'h5')
        save_array(self.user_embeddings.weight.cpu().detach().numpy(),
                   h5_path,
                   self.__class__.__name__ + '/' + 'user_embeddings')

        save_array(self.item_embeddings.weight.cpu().detach().numpy(),
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
        UserItemEmbeddingsFeature
        """
        h5_path = get_path(output_dir, filename, 'h5')
        user_embeddings = load_array(
            h5_path, cls.__name__ + '/' + 'user_embeddings'
        )
        item_embeddings = load_array(
            h5_path, cls.__name__ + '/' + 'item_embeddings'
        )

        return cls(user_embeddings, item_embeddings)

    @property
    def n_users(self):
        """Returns number of users"""
        return len(self.user_embeddings.weight)

    @property
    def n_items(self):
        """Returns number of items"""
        return len(self.item_embeddings.weight)
