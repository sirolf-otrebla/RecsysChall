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
        self._cosine = sim.Cosine_Similarity(bd.build_ICM(), k)
        self._k = k


    def fit(self, URM_train):
        S = self._cosine.compute()

    def recommend(self, user_ID, at = 10):

        d = 1

    def recommendALL(selfself, userList, at = 10):
        d = 1


# class CBF_User_Naive(object):
