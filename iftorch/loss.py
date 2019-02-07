# coding=utf-8
"""Base classes for loss"""
from __future__ import absolute_import, division, print_function

from future.standard_library import install_aliases
import numpy as np
import torch

from .base import BaseRecommender
from ._utils import dataloader

install_aliases()


def _hinge_loss(positive_predictions,
                negative_predictions,
                weights,
                gap):
    """Hinge pairwise loss function.

    Parameters
    ----------
    positive_predictions: torch.FloatTensor | torch.cuda.FloatTensor
        Tensor containing predictions for known positive items

    negative_predictions: torch.FloatTensor | torch.cuda.FloatTensor
        Tensor containing predictions for sampled negative items

    weights: torch.FloatTensor | torch.cuda.FloatTensor
        Tensor containing weights for the loss of each pair

    gap: float

    Returns
    -------
    loss: torch.FloatTensor | torch.cuda.FloatTensor
        The mean value of the loss function.
    """
    loss = weights * torch.clamp(
        negative_predictions - positive_predictions + gap,
        0.0
    )
    return loss.mean()


class ModifiedMaxWarpLoss(BaseRecommender):
    """Implements Warp loss to train a neural networks model
    for item recommendation.

    Parameters
    ----------
    num_epochs : int

    batch_size : int

    num_negative_samples : int

    max_abs_score : float

    gap : float

    verbose : int, optional (default=0)
        Verbosity level.

    is_cuda : bool, optional (default=None)
    """
    def __init__(self,
                 num_epochs,
                 batch_size=100,
                 num_negative_samples=100,
                 max_abs_score=100.0,
                 gap=1.0,
                 verbose=0.0,
                 is_cuda=None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_negative_samples = num_negative_samples

        if max_abs_score < 0.0:
            raise ValueError('Absolute max score should be non-negative.')
        self.max_abs_score = max_abs_score
        self.gap = gap
        self.verbose = verbose

        if is_cuda is None:
            self.is_cuda = torch.cuda.is_available()
        else:
            self.is_cuda = is_cuda

        if self.is_cuda:
            self._torch = torch.cuda
        else:
            self._torch = torch

        self.prefit = True

    def _torch_scores(self, users, items):
        """Returns the scores for the users and corresponding items.

        Parameters
        ----------
        users : np.ndarray, shape = [len(users),]
            Each row represents a user (inner id)

        items : np.ndarray, shape = [len(users),]
            Each row represents an item (inner id)
            that the user in the corresponding row in users
            chose

        Returns
        -------
        torch.cuda.FloatTensor | torch.FloatTensor, shape = [len(users),]
        """
        user_item_features = self._features.extract(users, items)
        user_item_scores = self.model_(user_item_features)

        return user_item_scores

    def _predict_scores(self, users):
        """Returns the scores for the users and corresponding items.

        Parameters
        ----------
        users : np.ndarray, shape = [len(users),]
            Each row represents a user (inner id)

        Returns
        -------
        np.ndarray[float], shape = [len(users) x self._features.n_items]
        """
        return self._scoring_method(users)

    def _init_model_vars(self, model, optimizer, features, scoring_method):
        """Initializes the model variables for training.

        Parameters
        ----------
        model : torch.nn.Module

        optimizer : torch.optim.Optimizer

        features : BaseFeature

        scoring_method : function
            Method that maps users to the scores for all the items
        """
        self.model_ = model
        self._optimizer = optimizer
        self._features = features
        if scoring_method is None:
            self._scoring_method = lambda users: self._torch_scores(
                users=np.repeat(users, self._features.n_items),
                items=np.asarray(len(users) * range(self._features.n_items))
            ).cpu().detach().numpy()
        elif callable(scoring_method):
            self._scoring_method = scoring_method.__get__(self)
        else:
            raise ValueError("'scoring_method' is not a function!")

    def fit(self, train_set, model, optimizer, features, scoring_method=None):
        """Trains the model on train_set.

        Parameters
        ----------
        train_set : UserItemSet

        model : torch.nn.Module

        optimizer : torch.optim.Optimizer

        features : BaseFeature

        scoring_method : function
            Method that maps users to the scores for all the items
        """
        self._init_model_vars(model, optimizer, features, scoring_method)
        self._fit(train_set)
        self.prefit = False

        return self

    def _fit(self, train_set):
        """Trains the model on train_set.

        Parameters
        ----------
        train_set : UserItemSet
        """
        total_num_items = len(train_set.items_inner)
        for epoch in range(self.num_epochs):
            sum_loss = 0.0

            batched_user_item_selections = dataloader(
                dataset=train_set.user_item_selections_inner,
                batch_size=self.batch_size,
                shuffle=True
            )
            for user_item_selections in batched_user_item_selections:
                users = user_item_selections[:, 0]
                items = user_item_selections[:, 1]
                num_batch_users = len(users)

                # 1. Compute scores for items that users chose
                positive_user_item_scores = torch.clamp(
                    self._torch_scores(users=users,
                                       items=items),
                    min=-self.max_abs_score
                )  # shape = [num_batch_users, 1]

                # 2. Compute scores for sampled items users did not choose
                sampled_items = np.random.choice(
                    total_num_items,
                    size=(num_batch_users, self.num_negative_samples),
                    replace=True
                )  # shape = [num_batch_users, self.num_negative_samples]

                sampled_user_item_scores = (
                    self._torch_scores(
                        users=np.repeat(users, self.num_negative_samples),
                        items=np.reshape(sampled_items, -1)
                    )
                ).view(num_batch_users, self.num_negative_samples)
                # shape = [num_batch_users, self.num_negative_samples]

                # 0 for item in train_set and 1 otherwise
                sampled_negative_item_mask = self._torch.FloatTensor(
                    [np.isin(sampled_items[row, :],
                             train_set.user2items_inner[user],
                             invert=True)
                     for row, user in enumerate(users)]
                )  # shape = [num_batch_users, self.num_negative_samples]

                # positive items scores are masked as 0.0
                bias_mask_negative_item_scores = (
                    sampled_negative_item_mask
                    * (sampled_user_item_scores + self.max_abs_score)
                )  # shape = [num_batch_users, self.num_negative_samples]

                # extracting maximum scores for negative items
                max_negative_item_scores = torch.max(
                    bias_mask_negative_item_scores,
                    dim=1
                )[0] - self.max_abs_score  # shape = [num_batch_users,]

                activated_negative_samples_sum = torch.sum(
                    sampled_negative_item_mask
                    * torch.gt(
                        bias_mask_negative_item_scores + self.gap,
                        positive_user_item_scores
                        + self.max_abs_score  # broadcast
                    ).type(self._torch.FloatTensor),
                    dim=1
                )
                activated_selections = torch.gt(
                    activated_negative_samples_sum,
                    0.0
                )
                activated_selections_sum = int(torch.sum(activated_selections))

                num_negative_items = self._torch.FloatTensor(
                    [total_num_items - len(train_set.user2items_inner[user])
                     for user in users]
                )
                hinge_loss_weights = torch.log(
                    num_negative_items[activated_selections]
                    * activated_negative_samples_sum[activated_selections]
                    / torch.sum(sampled_negative_item_mask,
                                dim=1)[activated_selections]
                )
                loss = _hinge_loss(
                    positive_user_item_scores[activated_selections],
                    max_negative_item_scores[activated_selections],
                    hinge_loss_weights,
                    self.gap
                )

                if activated_selections_sum > 0:
                    sum_loss += float(loss) * activated_selections_sum
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

            if self.verbose:
                print("Epoch : %d | Loss : %.4e" %
                      (epoch,
                       sum_loss / len(train_set.user_item_selections_inner)))
