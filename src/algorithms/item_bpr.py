from Base.Cython.cosine_similarity import Cosine_Similarity

from refactored_code.IALS_numpy import IALS_numpy
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np



class BMussoliniEnsemble:

    def __init__(self, urm_train, urm_test, icm):

        self.train = urm_train.tocsr()
        self.test = urm_test.tocsr()
        self.icm = icm.tocsr()

        self.initialize_components()

    def initialize_components(self):

        self.cbf_bpr_recommender = SLIM_BPR_Cython(self.icm.T, positive_threshold=0)

    def fit(self):
        self.cbf_bpr_w = self.cbf_bpr_recommender.fit(epochs=10, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)

    def recommend(self, user_id, combiner, at=10):
        user_profile = self.train[user_id, :]
        cbf_bpr_r = user_profile.dot(self.cbf_bpr_w).toarray().ravel()

        scores = [
            [cbf_bpr_r, "1", "CBF_BPR"]
        ]

        for r in scores:
            self.filter_seen(user_id, r[0])

        return combiner.combine(scores, at)

    def filter_seen(self, user_id, scores):

        start_pos = int(self.train.indptr[user_id])
        end_pos = int(self.train.indptr[user_id + 1])

        user_profile = self.train.indices[start_pos:end_pos]

        scores[user_profile] = -1000000 #-np.inf
        return scores

    def recommend_batch(self, user_list, combiner, at=10):
        res = np.array([])
        n=0
        for i in user_list:
            recList = self.recommend(i, combiner, at).T
            tuple = np.concatenate(([i], recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res

    def get_component_data(self):

        cbf_bpr_rating = 1*self.train.dot(self.cbf_bpr_w)
        cbf_bpr = {

                "min": cbf_bpr_rating.min(),
                "max": cbf_bpr_rating.max(),
                "mean": cbf_bpr_rating.mean(),
            }
        del cbf_bpr_rating
        return {
            "CBF_BPR" : cbf_bpr
        }
