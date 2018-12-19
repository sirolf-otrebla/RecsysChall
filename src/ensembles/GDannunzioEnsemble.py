from Base.Cython.cosine_similarity import Cosine_Similarity

from Base.Evaluation.Evaluator import SequentialEvaluator
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from sklearn import preprocessing
import numpy as np



class GDannunzioEnsemble:

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

        self.item_cosineCF_recommender = Cosine_Similarity(self.train, topK=200, shrink=15, normalize=True, mode='cosine')
        self.user_cosineCF_recommender = Cosine_Similarity(self.train.T, topK=200, shrink=15, normalize=True, mode='cosine')
        self.item_bpr_recommender = SLIM_BPR_Cython(self.train, positive_threshold=0)
        self.user_bpr_recommender = SLIM_BPR_Cython(self.train.T, positive_threshold=0)
        self.cbf_bpr_recommender = SLIM_BPR_Cython(self.icm.T, positive_threshold=0)
        self.cbf_recommender = Cosine_Similarity(self.icm.T, topK=50, shrink=10, normalize=True, mode='cosine')
        if self.ensemble_weights["IALS"] == 0:
            self.ials_recommender = IALS_numpy(iters=0)
        else:
            self.ials_recommender = IALS_numpy()

    def fit(self):


        self.item_bpr_w = self.item_bpr_recommender.fit(epochs=10, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)
        self.user_bpr_w = self.user_bpr_recommender.fit(epochs=10, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)
        self.cbf_bpr_w = self.cbf_bpr_recommender.fit(epochs=10, topK=200, batch_size=200, sgd_mode='adagrad', learning_rate=1e-2)
        self.item_cosineCF_w = self.item_cosineCF_recommender.compute_similarity()
        self.user_cosineCF_w = self.user_cosineCF_recommender.compute_similarity()
        self.cbf_w = self.cbf_recommender.compute_similarity()
        self.ials_latent_x, self.ials_latent_y = self.ials_recommender.fit(R=self.train)
        self.min_ials = np.dot(self.ials_latent_x, self.ials_latent_y.T).min()


    def recommend(self, user_id, combiner, at=10):
        user_profile = self.train[user_id, :]

        item_bpr_r = preprocessing.normalize(user_profile.dot(self.item_bpr_w).toarray(), norm='l2').ravel()
        user_bpr_r = preprocessing.normalize(self.user_bpr_w[user_id].dot(self.train).toarray(), norm='l2').ravel()
        item_cosineCF_r = preprocessing.normalize(user_profile.dot(self.item_cosineCF_w).toarray(), norm='l2').ravel()
        user_cosineCF_r = preprocessing.normalize(self.user_cosineCF_w[user_id].dot(self.train).toarray(), norm='l2').ravel()
        cbf_r = preprocessing.normalize(user_profile.dot(self.cbf_w).toarray(), norm='l2').ravel()
        cbf_bpr_r = preprocessing.normalize(user_profile.dot(self.cbf_bpr_w).toarray(), norm='l2').ravel()
        ials_r = preprocessing.normalize(np.dot(self.ials_latent_x[user_id], self.ials_latent_y.T + self.min_ials).reshape(1,-1), norm='l2').ravel()

        scores = [
            [item_bpr_r, self.ensemble_weights["ITEM_BPR"], "ITEM_BPR" ],
            [user_bpr_r, self.ensemble_weights["USER_BPR"], "USER_BPR" ],
            [item_cosineCF_r, self.ensemble_weights["ITEM_CF"], "ITEM_CF" ],
            [user_cosineCF_r, self.ensemble_weights["USER_CF"], "USER_CF" ],
            [ials_r, self.ensemble_weights["IALS"], "IALS" ],
            [cbf_r, self.ensemble_weights["CBF"], "CBF" ],
            [cbf_bpr_r, self.ensemble_weights["CBF_BPR"], "CBF_BPR"]
        ]

        for r in scores:
            self.filter_seen(user_id, r[0])

        return combiner.combine(scores, at)

    def filter_seen(self, user_id, scores):

        start_pos = int(self.train.indptr[user_id])
        end_pos = int(self.train.indptr[user_id + 1])

        user_profile = self.train.indices[start_pos:end_pos]

        scores[user_profile] = -1000000000

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

        user_cf_rating = self.ensemble_weights["USER_CF"]*self.user_cosineCF_w.dot(self.train)

        user_cf = {
                "min": user_cf_rating.min(),
                "max": user_cf_rating.max(),
                "mean": user_cf_rating.mean(),
            }
        del user_cf_rating
        user_bpr_rating = self.ensemble_weights["USER_BPR"]*self.user_bpr_w.dot(self.train)

        user_bpr = {

                "min": user_bpr_rating.min(),
                "max": user_bpr_rating.max(),
                "mean": user_bpr_rating.mean(),
            }
        del user_bpr_rating
        item_bpr_rating =self.ensemble_weights["ITEM_BPR"]*self.train.dot(self.item_bpr_w)
        item_bpr = {
                "min": item_bpr_rating.min(),
                "max": item_bpr_rating.max(),
                "mean": item_bpr_rating.mean(),
            }
        del item_bpr_rating
        ials_rating =  self.ensemble_weights["IALS"]*(np.dot(self.ials_latent_x, self.ials_latent_y.T)+self.min_ials)

        ials = {

                "min": ials_rating.min(),
                "max": ials_rating.max(),
                "mean": np.mean(ials_rating),
            }
        del ials_rating
        cbf_rating = self.ensemble_weights["CBF"]*self.train.dot(self.cbf_w)
        cbf = {

                "min": cbf_rating.min(),
                "max": cbf_rating.max(),
                "mean": cbf_rating.mean(),
            }
        del cbf_rating
        cbf_bpr_rating = self.ensemble_weights["CBF_BPR"]*self.train.dot(self.cbf_bpr_w)
        cbf_bpr = {

                "min": cbf_bpr_rating.min(),
                "max": cbf_bpr_rating.max(),
                "mean": cbf_bpr_rating.mean(),
            }
        del cbf_bpr_rating
        return {
            "ITEM_CF" : item_cf,
            "USER_CF": user_cf ,
            "ITEM_BPR" : item_bpr ,
            "USER_BPR" : user_bpr,
            "IALS" : ials,
            "CBF" : cbf,
            "CBF_BPR" : cbf_bpr
        }
