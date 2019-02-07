"""
The :mod:`iftorch.algorithms` module includes various algorithms
for recommenders with implicit feedback.
"""
from .collaborative_filtering import CollaborativeRecommender
from .matrix_factorization import SVDRecommender
from .user_item_embeddings_feature import UserItemEmbeddingsFeature

__all__ = ['CollaborativeRecommender',
           'SVDRecommender',
           'UserItemEmbeddingsFeature']
