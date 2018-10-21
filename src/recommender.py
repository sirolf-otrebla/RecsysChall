import scipy
import numpy as np
import src.similarities as sim
import src.build_data as bd

class TopPop(object):

    def fit(self, URM_train):

        itemPopularity = (URM_train>0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self._popularItems = np.argsort(itemPopularity)
        self._popularItems = np.flip(self._popularItems, axis = 0)

    def recommend(self, user_ID, at=10):
        recommended = self._popularItems[0:at]
        return recommended

    def recommendAll(self, userList, at=10):
        res = np.array([])

        for i in userList:
            recList = self.recommend(i, 10)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res


class CBF_Item_Naive(object):
    def __init__(self, k):
        self._cosine = sim.Cosine_Similarity(bd.build_icm(), k)
        self.URM = bd.build_urm()
        self._k = k


    def fit(self, URM_train):
        self._cosine.compute()
        S_knn = self._cosine.topK(self._k)
        DEBUG = S_knn.getrow(13757).toarray()
        print(DEBUG)
        self._S_knn = S_knn
        self.estimated_ratings = self.URM.dot(S_knn.transpose())


    def recommend(self, user_ID, at = 10):
        user_ID = int(user_ID)
        user_estimated = self.estimated_ratings.getrow(user_ID).toarray().squeeze()
        user_real = self.URM.tocsr().getrow(user_ID).toarray().squeeze()

        user_real = np.argwhere(user_real > 0)

        user_estimated_sorted = np.argsort(-user_estimated)
        DEBUGuser_estimated_values = np.sort(-user_estimated)
        DEBUGurm_row13759 = self.URM.getrow(13759).toarray()
        DEBUGsknn_row13759 = self._S_knn.getrow(13759).toarray()
        recommendation = [x for x in user_estimated_sorted if x not in user_real]
        return recommendation[0:at]

    def recommendAll(self, userList, at = 10):
        res = np.array([])
        for i in userList:
            recList = self.recommend(i, 10)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res


class CBF_coldstart(object):

    def __init__(self, k,  coldstart=10):

        self._coldstart = coldstart
        self._cbf = CBF_Item_Naive(k)
        self._URM = bd.build_urm().tocsr()
        self._toppop = TopPop()

    def fit(self):

        self._toppop.fit(self._URM)
        self._cbf.fit(self._URM)

    def recommend(self, userID, at=10, limit=10):

        row_sum = self._URM.getrow(userID).sum()
        if row_sum > limit :
            return self._cbf.recommend(userID, at)


        else:
            return self._toppop.recommend(userID, at)

    def recommendAll(self, userList, at=10, limit=10):
        res = np.array([])
        for i in userList:
            recList = self.recommend(i, 10, limit)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res
