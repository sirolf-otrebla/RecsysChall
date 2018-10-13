import pandas as p
import scipy as sp
import scipy.sparse as sps
import numpy as np

class Data_manager:

    def __init__(self, URM, ICM = 0):

        self._cooURM = URM
        self._csrURM = URM.tocsr()
        self._cscURM = URM.tocsc()

    def getURM_CSR(self):
        return  self._csrURM

    def getURM_CSC(self):
        return  self._cscURM

    def getURM_COO(self):
        return  self._cooURM

class TopPop(object):

    def fit(self, URM_train):
        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self._popularItems = np.argsort(itemPopularity)
        self._popularItems = np.flip(self._popularItems, axis=0)

    def recommend(self, user_ID, at=10):
        recommended = self._popularItems[0:at]
        return recommended

def build():
    datafile = np.loadtxt('../../data/train.csv', delimiter=',', skiprows=1, dtype=int)
    userList, itemList = zip(*datafile)
    ratings = np.ones(1211791, dtype=int)
    URM_all = sps.coo_matrix((ratings, (userList, itemList)))
    dm = Data_manager(URM_all)
    return  dm

dm = build()
rec = TopPop()
rec.fit(dm.getURM_COO())
recommended = rec.recommend(1,10)
print(recommended)