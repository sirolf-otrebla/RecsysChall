from Base.Cython.cosine_similarity import Cosine_Similarity
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Base.Evaluation.Evaluator import SequentialEvaluator
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from implicit.bpr import BayesianPersonalizedRanking as BPR_matrix_factorization
from sklearn import preprocessing
import numpy as np



class SStorinoEnsemble:

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

        self.user_bpr_recommender = SLIM_BPR_Cython(self.train.T, positive_threshold=0)

    def fit(self):

        self.user_bpr_w = self.user_bpr_recommender.fit(epochs=6, topK=200,
                                                        lambda_i=0.0, lambda_j=0.0, # lambda_j=0.0005
                                                        gamma=0.9,
                                                        beta_1=0.00099,
                                                        beta_2=0.00099,
                                                        batch_size=20000, sgd_mode='adagrad', learning_rate=1e-2)


    def recommend(self, user_id, combiner, at=10):
        user_profile = self.train[user_id, :]
        user_bpr_r = self.user_bpr_w[user_id].dot(self.train).toarray().ravel()

        scores = [
            [user_bpr_r, self.ensemble_weights["USER_BPR"], "USER_BPR" ],
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
        print('сука')
