"""Test PopularityRecommender module."""
from iftorch.dataset import UserItemSet
from iftorch.mapping import RawInnerMap
from iftorch.algorithms.popularity import PopularityRecommender


def test_popularity_recommendation_without_grouping():
    """Test recommendations made based on popularity
    when users and items have no groups."""
    users_raw_train = ['user_0', 'user_0', 'user_0',
                       'user_1',
                       'dummy_user_2', 'dummy_user_2',
                       'dummy_user_3']
    items_raw_train = ['item_0', 'item_1', 'item_2',
                       'item_0',
                       'item_1', 'item_2',
                       'item_2']
    users_raw_test = ['user_0',
                      'user_1']
    items_raw_test = ['dummy_item_3',
                      'dummy_item_4']
    raw_inner_map = RawInnerMap(
        raw2inner_user={'user_0': 0, 'user_1': 1,
                        'dummy_user_2': 2, 'dummy_user_3': 3},
        raw2inner_item={'item_0': 0, 'item_1': 1, 'item_2': 2,
                        'dummy_item_3': 3, 'dummy_item_4': 4})

    train_set = UserItemSet.load_cls_from_raw(users_raw_train,
                                              items_raw_train,
                                              raw_inner_map)
    test_set = UserItemSet.load_cls_from_raw(users_raw_test,
                                             items_raw_test,
                                             raw_inner_map)

    popularity_recommender = PopularityRecommender().fit(train_set)
    user2recommended_items = popularity_recommender.predict(test_set,
                                                            train_set)

    assert user2recommended_items[0] == [], \
        "User should not be recommended items already applied for in Train."
    assert user2recommended_items[1] == [2, 1], \
        "Items were not recommended according to popularity."


def test_popularity_recommendation_with_grouping():
    """Test recommendations made based on popularity
    when users and items have groups."""
    users_raw_train = ['user_0',
                       'user_1', 'user_1']
    items_raw_train = ['cluster_0_item_0',
                       'cluster_1_item_1', 'cluster_1_item_2']
    users_raw_test = ['user_0', 'user_1']
    items_raw_test = ['dummy_item_2', 'dummy_item_2']
    raw_inner_map = RawInnerMap(
        raw2inner_user={'user_0': 0, 'user_1': 1},
        raw2inner_item={'cluster_0_item_0': 0,
                        'cluster_1_item_1': 1, 'cluster_1_item_2': 2,
                        'dummy_item_2': 3})

    train_set = UserItemSet.load_cls_from_raw(users_raw_train,
                                              items_raw_train,
                                              raw_inner_map)
    test_set = UserItemSet.load_cls_from_raw(users_raw_test,
                                             items_raw_test,
                                             raw_inner_map)

    user_grouping = {"user_0": 0, "user_1": 0}
    item_grouping = {"cluster_0_item_0": 0,
                     "cluster_1_item_1": 1, "cluster_1_item_2": 1,
                     "dummy_item_2": 2}

    inner_user_grouping = {}
    for user, cluster in user_grouping.items():
        inner_user_grouping.update(
            {raw_inner_map.to_user_inner(user): cluster}
        )

    inner_item_grouping = {}
    for item, cluster in item_grouping.items():
        inner_item_grouping.update(
            {raw_inner_map.to_item_inner(item): cluster})

    popularity_recommender = (
        PopularityRecommender(user_grouping=inner_user_grouping,
                              item_grouping=inner_item_grouping)
        .fit(train_set)
    )
    user2recommended_items = (popularity_recommender
                              .predict(test_set=test_set,
                                       train_set=train_set,
                                       max_score_to_filter=0))

    assert user2recommended_items[0] == [], \
        "Items from a different group to user should have a score of 0."
    assert user2recommended_items[1] == [0],\
        "Items from same group as user should be recommended"


def test_popularity_recommendation_with_missing_grouping():
    users_raw_train = ['user_0',
                       'user_1']
    items_raw_train = ['item_1',
                       'item_0']
    users_raw_test = ['user_0', 'user_1']
    items_raw_test = ['dummy_item_2', 'dummy_item_2']
    raw_inner_map = RawInnerMap(
        raw2inner_user={'user_0': 0, 'user_1': 1},
        raw2inner_item={'item_0': 0, 'item_1': 1, 'dummy_item_2': 2})

    train_set = UserItemSet.load_cls_from_raw(users_raw_train,
                                              items_raw_train,
                                              raw_inner_map)
    test_set = UserItemSet.load_cls_from_raw(users_raw_test,
                                             items_raw_test,
                                             raw_inner_map)

    user_grouping = {"user_0": 0}
    item_grouping = {"item_0": 0}

    inner_user_grouping = {}
    for user, cluster in user_grouping.items():
        inner_user_grouping.update(
            {raw_inner_map.to_user_inner(user): cluster})

    inner_item_grouping = {}
    for item, cluster in item_grouping.items():
        inner_item_grouping.update(
            {raw_inner_map.to_item_inner(item): cluster})

    popularity_recommender = (
        PopularityRecommender(user_grouping=inner_user_grouping,
                              item_grouping=inner_item_grouping)
        .fit(train_set)
    )
    user2recommended_items = (popularity_recommender
                              .predict(test_set=test_set,
                                       train_set=train_set,
                                       max_score_to_filter=0))

    assert user2recommended_items[0] == [0], \
        "User should be recommended item from same group."
    assert user2recommended_items[1] == [], \
        "User should not be recommended items when they have no group."
