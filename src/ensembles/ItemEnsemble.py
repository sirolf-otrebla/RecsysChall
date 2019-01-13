from Base.Cython.cosine_similarity import Cosine_Similarity
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from implicit.als import AlternatingLeastSquares as IALS_CG
from Base.Evaluation.Evaluator import SequentialEvaluator
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from implicit.bpr import BayesianPersonalizedRanking as BPR_matrix_factorization
from sklearn import preprocessing
import numpy as np
from lightfm import LightFM



class ItemEnsemble:

    def __init__(self, urm_train, urm_test, icm, parameters=None):

        if parameters is None:
            parameters = {
                "ITEM_CF" : 1,
                "ITEM_BPR" : 1,
                "P3_ALPHA": 1,
                "ITEM_RP3B": 1
            }

        self.ensemble_weights = parameters
        self.train = urm_train.tocsr()
        self.test = urm_test.tocsr()
        self.icm = icm.tocsr()

        self.initialize_components()


    def initialize_components(self):

        self.item_cosineCF_recommender = Cosine_Similarity(self.train, topK=200, shrink=15, normalize=True, mode='cosine')
        self.item_bpr_recommender = SLIM_BPR_Cython(self.train, positive_threshold=0)
        self.item_rp3b_recommender = RP3betaRecommender(self.train)
        self.p3alpha_recommender = P3alphaRecommender(self.train)

    def fit(self):

        self.item_cosineCF_w = self.item_cosineCF_recommender.compute_similarity()
        self.item_rp3b_w = self.item_rp3b_recommender.fit()
        self.p3alpha_w = self.p3alpha_recommender.fit()
        self.item_bpr_w = self.item_bpr_recommender.fit(epochs=15, topK=200, batch_size=200, sgd_mode='adagrad',
                                                        learning_rate=1e-2)

    def recommend(self, user_id, combiner, at=10):
        user_profile = self.train[user_id, :]

        item_bpr_r = user_profile.dot(self.item_bpr_w).toarray().ravel()
        item_cosineCF_r = user_profile.dot(self.item_cosineCF_w).toarray().ravel()
        item_rp3b_r = user_profile.dot(self.item_rp3b_w).toarray().ravel()
        p3alpha_r = user_profile.dot(self.p3alpha_w).toarray().ravel()

        scores = [
            [item_bpr_r, self.ensemble_weights["ITEM_BPR"], "ITEM_BPR" ],
            [item_cosineCF_r, self.ensemble_weights["ITEM_CF"], "ITEM_CF" ],
            [item_rp3b_r, self.ensemble_weights["ITEM_RP3B"], "ITEM_RP3B"],
            [p3alpha_r, self.ensemble_weights["P3_ALPHA"], "P3_ALPHA"]
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
        item_cf_rating = self.ensemble_weights["ITEM_CF"]*self.train.dot(self.item_cosineCF_w)

        item_cf = {

                "min" : item_cf_rating.min(),
                "max" : item_cf_rating.max(),
                "mean" : item_cf_rating.mean(),

            }
        del item_cf_rating

        return {
            "ITEM_CF" : item_cf
        }
