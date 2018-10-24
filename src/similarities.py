import numpy as np
import time, sys
import scipy.sparse as sps
from src import utils

check_matrix = utils.check_matrix



class Cosine_Similarity(object):

    def __init__(self, ICM, k=100):
        self.diop = k
        self.ICM = ICM.copy()
        self._S = None

    def compute(self):
        check_matrix(self.ICM, 'csr')
        S = self.ICM.dot(self.ICM.transpose()).tocsr()
        S.setdiag(0)
        self._S = S
        self._weighted_S = S
        return S

    def assign_weights(self, w):
        self._weighted_S = np.dot(self._S, w)
        return  self._weighted_S

    def topK(self, k):

        topk_matrix = []
        # values = sps.lil_matrix((self._weighted_S.shape[0], self._weighted_S.shape[1]))
        values = sps.lil_matrix((self._weighted_S.shape[0], self._weighted_S.shape[1]))
        topK_items_sorted = []
        d = []
        if (self._weighted_S != None):
            for row_index in range(self._weighted_S.shape[0]):
                row = self._weighted_S.getrow(row_index).toarray().squeeze()
                #item_data = self.ICM[row, :]
                #item_data = item_data.toarray.squeeze()

                # partition row placing at the k-th position element
                # that would occupy that position in an ordered array.
                # then, move all elements greater or equal than that
                # to the left partition and elements smaller to the
                # right partition. since we are interested only about
                # the top k elements, e.g. the left part of the array
                #  we want to select only those using [0:topK]

                # D I O K A N 3  I L  C A Z Z O   D I   M E N O   AAAAAAAAAAAA
                topK_items = np.argpartition(-row, k-1)[0:k]

                # now we want to order the topK_items we found before
                # so that we can check the most similar items in order
                d = row[topK_items]
                partition_sorting = np.argsort(-row[topK_items])
                topK_items_sorted = topK_items[partition_sorting]
                topk_matrix.append(topK_items_sorted)

            for topk_row_idx in range(len(topk_matrix)):
                for element in topk_matrix[topk_row_idx]:
                    values[topk_row_idx, element] = 1
                    #Check this matrix, it should be an elementwise matrix

        # S_knn = np.dot(self._weighted_S, values)
        S_knn = np.multiply(self._weighted_S, values)
        return S_knn

