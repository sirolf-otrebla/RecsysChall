from Base.Cython.cosine_similarity import Cosine_Similarity
from GraphBased.RP3betaRecommender import RP3betaRecommender
from implicit.als import AlternatingLeastSquares as IALS_CG
from src.combiners.probabilisticCombiner import ProbabilisticCombiner
from Base.Evaluation.Evaluator import SequentialEvaluator
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from implicit.bpr import BayesianPersonalizedRanking as BPR_matrix_factorization
from sklearn import preprocessing
import numpy as np

from src.combiners.linearCombiner import linearCombiner
from src.ensembles.BMussoliniEnsemble import BMussoliniEnsemble


class NTeslaEnsemble:

    def __init__(self, urm_train, urm_test, icm, parameters=None):

        if parameters is None:
            parameters = {
                "USER_CF" : 0.8,
                "USER_BPR" : 0.7,
                "ITEM_CF" : 1,
                "ITEM_BPR" : 0.8,
                "CBF" : 0.3,
                "IALS" : 1.0,
                "CBF_BPR" : 1
            }

        self.ensemble_weights = parameters
        self.train = urm_train.tocsr()
        self.test = urm_test.tocsr()
        self.icm = icm.tocsr()

        self.initialize_components()


    def initialize_components(self):
        self.bm_recommender = BMussoliniEnsemble(urm_train=self.train, urm_test=self.test,
                                                 icm=self.icm, parameters=self.ensemble_weights)
        self.item_bpr_recommender = SLIM_BPR_Cython(self.train, positive_threshold=0.1)
        self.user_bpr_recommender = SLIM_BPR_Cython(self.train.T, positive_threshold=0.1)


    def fit(self):
        self.bm_recommender.fit()
        self.item_bpr_w = self.item_bpr_recommender.fit(epochs=15, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)
        self.user_bpr_w = self.user_bpr_recommender.fit(epochs=10, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)

    def recommend(self, user_id, combiner, at=10):
        user_profile = self.train[user_id, :]

        bm_list = self.bm_recommender.recommend(user_id, linearCombiner(), at=40)
        item_bpr_r = user_profile.dot(self.item_bpr_w).toarray()
        user_bpr_r = self.user_bpr_w[user_id].dot(self.train).toarray()
        # item_bpr_r = preprocessing.normalize(user_profile.dot(self.item_bpr_w).toarray(), norm='l2').ravel()
        # user_bpr_r = preprocessing.normalize(self.user_bpr_w[user_id].dot(self.train).toarray(), norm='l2').ravel()

        bpr_scores = [
            [item_bpr_r, self.ensemble_weights["ITEM_BPR"], "ITEM_BPR"],
            [user_bpr_r, self.ensemble_weights["USER_BPR"], "USER_BPR"]
        ]

        for r in bpr_scores:
            self.filter_seen(user_id, r[0])

        bpr_list = (linearCombiner()).combine(bpr_scores, at=40)
        lists = [
            [bm_list],
            [bpr_list]
            ]

        return (ProbabilisticCombiner()).combine(scores=lists, alpha=self.ensemble_weights["ALPHA"], at=at)

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
            recList = self.recommend(i, combiner, at)
            tuple = np.concatenate(([i], recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res

    def get_component_data(self):

        return {
            "ITEM_CF" : 1,
            "USER_CF": 1 ,
            # "ITEM_BPR" : item_bpr ,
            "IALS" : 1,
            "CBF" : 1,
            "CBF_BPR" : 1
        }
