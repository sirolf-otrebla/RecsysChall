import pandas as pd
import numpy as np
import scipy.sparse as sps
import cython
from scipy.sparse import hstack
from MF_RMSE import *

def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


# We pass as paramether the recommender class

def evaluate_algorithm(userList_unique, URM_test, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in userList_unique:

        relevant_items = URM_test[user_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)

class MFGDRecsys():

    def __init__(self, URM_train, k=100, shrink=0):
        self._URM_train = URM_train
        self._ICM = ICM
        self._k = k
        self._shrink = shrink

    def fit(self, num_factors=50,
                 learning_rate=0.01,
                 reg=0.015,
                 epochs=50,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):

        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

        self.U, self.V = FunkSVD_sgd(self._URM_train, self.num_factors, self.learning_rate, self.reg, self.epochs, self.init_mean,
                                     self.init_std,
                                     self.lrate_decay, self.rnd_seed)
        self._estimated_ratings = np.dot(self.U, self.V.T)


    def recommend(self, user_id, at=10):
        user_real = self._URM_train.getrow(user_id).toarray().squeeze()
        user_estimated = self._estimated_ratings[user_id, :]

        user_real = np.argwhere(user_real > 0)

        user_estimated_sorted = np.argsort(-user_estimated)
        recommendation = [x for x in user_estimated_sorted[0,:] if x not in user_real]

        debug = recommendation[0:at]

        return debug

    def recommendALL(self, userList, at=10):
        res = np.array([])
        for i in userList:
            recList = self.recommend(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res

if __name__ == '__main__':
    URM_text = np.loadtxt('../../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    URM = sps.csr_matrix((rating_list, (user_list, item_list)))

    train_test_split = 0.8
    numInteractions = URM.nnz
    train_mask = np.random.choice([True, False], numInteractions, p=[train_test_split, 1 - train_test_split])

    userList = np.array(user_list)
    itemList = np.array(item_list)
    ratingList = np.array(rating_list)

    URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
    test_mask = np.logical_not(train_mask)
    URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))

    ICM_text = np.loadtxt('../../data/tracks.csv', delimiter=',', skiprows=1, dtype=int)

    tracks_list, album_list, artist_list, duration_list = zip(*ICM_text)

    ratings = np.ones(len(album_list), dtype=int)

    ICM_album = sps.csc_matrix((ratings, (tracks_list, album_list)))
    ICM_artist = sps.csc_matrix((ratings, (tracks_list, artist_list)))

    duration_class_list = []

    for index in range(len(tracks_list)):
        if duration_list[index] < 106:
            duration_class_list.append(0)

        elif duration_list[index] >= 106 and duration_list[index] < 212:
            duration_class_list.append(1)

        elif (duration_list[index] >= 212 and duration_list[index] < 318):
            duration_class_list.append(2)

        else:
            duration_class_list.append(3)

    ICM_duration = sps.csc_matrix((ratings, (tracks_list, duration_class_list)))
    ICM_partial = hstack((ICM_album, ICM_artist))

    ICM = hstack((ICM_partial, ICM_duration))
    R = URM_train.tocsr()
    cf = MFGDRecsys(R, 50)
    cf.fit()
    target = pd.read_csv('../../data/target_playlists.csv', index_col=False)
    recommended = cf.recommendALL(target.values)

    playlists = recommended[:, 0]
    recommended = np.delete(recommended, 0, 1)
    i = 0
    res_fin = []
    for j in recommended:
        res = ''
        for k in range(0, len(j)):
            res = res + '{0} '.format(j[k])
        res_fin.append(res)
        i = i + 1
    d = {'playlist_id': playlists, 'track_ids': res_fin}
    df = pd.DataFrame(data=d, index=None)
    df.to_csv("../../results/recommendedMFGD.csv", index=None)
