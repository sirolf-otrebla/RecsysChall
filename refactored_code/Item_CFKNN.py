from refactored_code.cosine_similarity import Cosine_Similarity
import numpy as np


class ItemCFKNN:

    def __init__(self, URM_train, k=100, shrink=0):
        self._URM_train = URM_train.tocsr()
        self._k = k
        self._shrink = shrink

    def fit(self):
        self._similarity_matrix = Cosine_Similarity(self._URM_train.tocsc(), self._k, self._shrink, normalize=False, mode='cosine').compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        scores = user_profile.dot(self._similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = int(self._URM_train.indptr[user_id])
        end_pos = int(self._URM_train.indptr[user_id + 1])

        user_profile = self._URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommendALL(self, userList, at=10):
        res = np.array([])
        n=0
        for i in userList:
            n+=1
            recList = self.recommend(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res
