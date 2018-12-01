from Base.Cython.cosine_similarity import Cosine_Similarity

from Base.Evaluation.Evaluator import SequentialEvaluator
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from sklearn import preprocessing
import numpy as np



class BMussoliniEnsemble:

    def __init__(self, urm_train, urm_test, icm, parameters=None):

        if parameters is None:
            parameters = {
                "USER_CF" : 0.8,
                "USER_BPR" : 0.7,
                "ITEM_CF" : 0.8,
                "ITEM_BPR" : 0.8,
                "CBF" : 0.3,
                "IALS" : 0.6
            }

        self.ensemble_weights = parameters
        self.train = urm_train.tocsr()
        self.test = urm_test.tocsr()
        self.icm = icm.tocsr()

        self.initialize_components()


    def initialize_components(self):

        self.item_cosineCF_recommender = Cosine_Similarity(self.train, topK=200, shrink=15, normalize=True, mode='cosine')
        self.user_cosineCF_recommender = Cosine_Similarity(self.train.T, topK=200, shrink=15, normalize=True, mode='cosine')
        self.item_bpr_recommender = SLIM_BPR_Cython(self.train, positive_threshold=0)
        self.user_bpr_recommender = SLIM_BPR_Cython(self.train.T, positive_threshold=0)
        self.cbf_recommender = Cosine_Similarity(self.icm.t, topK=50, shrink=10, normalize=True, mode='cosine')
        self.ials_recommender = IALS_numpy()

    def fit(self):

        self.item_bpr_w = self.item_bpr_recommender.fit(epochs=10, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)
        self.user_bpr_w = self.user_bpr_recommender.fit(epochs=10, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)
        self.item_cosineCF_w = self.item_cosineCF_recommender.compute_similarity()
        self.user_cosineCF_w = self.user_cosineCF_recommender.compute_similarity()
        self.cbf_w = self.cbf_recommender.compute_similarity()
        self.ials_latent_x, self.ials_latent_y = self.ials_recommender.fit(R=self.train)

    def recommend(self, user_id, combiner, at=10):
        user_profile = self.train[user_id]

        item_bpr_r = user_profile.dot(self.item_bpr_w).toarray().ravel()
        user_bpr_r = self.user_bpr_w[user_id].dot(self.train).toarray().ravel()
        item_cosineCF_r = user_profile.dot(self.item_cosineCF_w).toarray().ravel()
        user_cosineCF_r = self.user_cosineCF_w[user_id].dot(self.train).toarray().ravel()
        cbf_r = self.cbf_w.dot(self.icm.T).toarray().ravel()
        ials_r = np.dot(self.ials_latent_x, self.ials_latent_y.T).ravel()

        scores = [
            [item_bpr_r, self.ensemble_weights["ITEM_BPR"], "ITEM_BPR" ],
            [user_bpr_r, self.ensemble_weights["USER_BPR"], "USER_BPR" ],
            [item_cosineCF_r, self.ensemble_weights["ITEM_CF"], "ITEM_CF" ],
            [user_cosineCF_r, self.ensemble_weights["USER_CF"], "USER_CF" ],
            [ials_r, self.ensemble_weights["IALS"], "IALS" ],
            [cbf_r, self.ensemble_weights["CBF"], "CBF" ],
        ]

        for r in scores:
            self.filter_seen(user_id, r[0])

        return combiner.combine(at, scores)

    def filter_seen(self, user_id, scores):

        start_pos = int(self.train.indptr[user_id])
        end_pos = int(self.train.indptr[user_id + 1])

        user_profile = self.train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommend_batch(self, user_list, combiner, at=10):
        res = np.array([])
        n=0
        for i in user_list:
            recList = self.recommend(i, combiner, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res
