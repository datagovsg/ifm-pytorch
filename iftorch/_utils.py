# coding=utf-8
"""The _utils module contains various utility functions."""
from __future__ import absolute_import, division, print_function
import math

from future.standard_library import install_aliases
import numpy as np
import six

install_aliases()


def dataloader(dataset, batch_size, shuffle=True):
    """Yield batches of the dataset.

    Parameters
    ----------
        dataset : list | tuple | numpy.ndarray

        batch_size : int

        shuffle : bool

    Yields
    ------
        : list | numpy.ndarray
    """
    size_dataset = len(dataset)
    indices = np.arange(size_dataset)
    if shuffle:
        np.random.shuffle(indices)
    num_batches = math.ceil(size_dataset / batch_size)

    is_np_ndarray = isinstance(dataset, np.ndarray)
    for batch in range(num_batches):
        start_index = batch * batch_size
        end_index = start_index + batch_size
        if is_np_ndarray:
            yield dataset[indices[start_index:end_index]]
        else:
            yield [dataset[index]
                   for index in indices[start_index:end_index]]


def reverse_mapping(original_dict):
    """Reverses the mapping between keys and values for a dict

    Parameters
    ----------
    original_dict : dict[str | int, str | int]
        Original dictionary

    Returns
    -------
    dict[str | int, str | int]
        Dictionary with its keys and values reversed
    """
    return dict((value, key) for key, value in six.iteritems(original_dict))


def recommended_items(item_scores,
                      items_to_filter,
                      max_score_to_filter=None,
                      num_recommendations=150):
    """Returns the recommended items based on the item scores.

    Parameters
    ----------
    item_scores : numpy.ndarray[np.float64]
        Scores for each item where the index is the item (inner id).

    items_to_filter : set[int]
        Items (inner id) to filter from the recommendation

    max_score_to_filter : float, optional
        Maximum score of item to filter from the recommendation

    num_recommendations : int, optional

    Returns
    -------
    recommended : list[int]
    """
    recommended = []
    for item in np.argsort(-item_scores):  # index is the inner item id
        if item not in items_to_filter:
            if max_score_to_filter is not None:
                if item_scores[item] <= max_score_to_filter:
                    break
            recommended.append(item)
            if len(recommended) == num_recommendations:
                break
    return recommended
