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
        self.estimated_ratings = self.URM.dot(S_knn)


    def recommend(self, user_ID, at = 10):
        user_estimated = self.estimated_ratings.getrow(user_ID).toarray().squeeze()
        user_real = self.URM.tocsr().getrow(user_ID).toarray().squeeze()

        user_real = np.argwhere(user_real > 0)

        user_estimated_sorted = np.argsort(-user_estimated)

        recommendation = [x for x in user_estimated_sorted if x not in user_real]
        return recommendation[0:at]

    def recommendALL(self, userList, at = 10):
        for user in userList:
            print(self.recommend(user, at))

if __name__ == '__main__':
    a = CBF_Item_Naive(10)
    a.fit(a)
    print(a.recommend(0))