import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import hstack
import time, sys
from src.new_utils import utils

def gen_k_folds_matrix(URM, n):
    for i in range(0,n):
        URM_train, URM_test = utils.train_test_holdout(URM, 0.95)
        if URM_train.shape[0] == URM.shape[0] and URM_train.shape[1] == URM.shape[1]\
                and URM_test.shape[0] == URM.shape[0] and URM_test.shape[1] == URM.shape[1]:
            sps.save_npz("../data/validation_mat/TRAIN_{0}".format(i), URM_train)
            sps.save_npz("../data/validation_mat/TEST_{0}".format(i), URM_test)
        else:
            i -= 1



if __name__ == '__main__':
    URM_text = np.loadtxt('../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    URM = sps.csr_matrix((rating_list, (user_list, item_list)))
    gen_k_folds_matrix(URM, 1000)
