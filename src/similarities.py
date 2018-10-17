import numpy as np
import time, sys
import scipy.sparse as sps
from src import utils

check_matrix = utils.check_matrix

class Cosine_Similarity(object):

    def __init__(self, ICM, k=100):
        self.topK = 100
        self.ICM = ICM.copy()
        self._S = None

    def compute(self):
        check_matrix(self.ICM, 'csc')
        S = np.dot(self.ICM, self.ICM)
        np.fill_diagonal(S, 0)
        self._S = S
        return S

    def assign_weights(self, w):
        self._weighted_S = np.dot(self._S, w)
        return  self._weighted_S

    def topK(self, topK):

        topk_matrix = []

        if (self._weighted_S != None):
            for row in range(self._weighted_S.shape[0]):
                item_data = self.ICM[row, :]
                item_data = item_data.toarray.squeeze()

                # partition row placing at the k-th position element
                # that would occupy that position in an ordered array.
                # then, move all elements greater or equal than that
                # to the left partition and elements smaller to the
                # right partition. since we are interested only about
                # the top k elements, e.g. the left part of the array
                #  we want to select only those using [0:topK]

                topK_items = row.argpartition(topK-1)[0:topK]

                # now we want to order the topK_items we found before
                # so that we can check the most similar items in order
                topK_items_sorted = np.argsort(row[topK_items])
                topk_matrix.append(topK_items_sorted)

        return topk_matrix
