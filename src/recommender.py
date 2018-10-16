import scipy
import numpy as np

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


class Knn(object):
    def __init__(self, k):

        self._k = k


    def fit(self, URM_train):

        d = 1

    def recommend(self, user_ID, at = 10):

        d = 1

    def recommendALL(selfself, userList, at = 10):
        d = 1


