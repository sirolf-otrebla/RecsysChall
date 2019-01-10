from Base.Cython.cosine_similarity import Cosine_Similarity
from GraphBased.RP3betaRecommender import RP3betaRecommender
from implicit.als import AlternatingLeastSquares as IALS_CG
from Base.Evaluation.Evaluator import SequentialEvaluator
from MatrixFactorization.PureSVD import PureSVDRecommender
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from implicit.bpr import BayesianPersonalizedRanking as BPR_matrix_factorization
from sklearn import preprocessing
import numpy as np
import src.load_train_sequential as load_sequential
from lightfm_test import LightFM_Recommender

POPULARITY_SCALING_EXP = .0353


class BMussoliniEnsemble:

    def __init__(self, urm_train, urm_test, icm, parameters=None):

        if parameters is None:
            parameters = {
                "USER_CF" : 7,
                "SVD" : 26,
                "ITEM_CF" : 0,
                "ITEM_BPR" : 16,
                "CBF" : 7,
                "IALS" : 26,
                "CBF_BPR" : 64,
                "BPR_MF": 6,
                "ITEM_RP3B": 16,
                "USER_RP3B": 0,
                "FM": 10
            }

        self.ensemble_weights = parameters
        self.train = urm_train.tocsr()
        self.test = urm_test.tocsr()
        self.icm = icm.tocsr()
        self.sequential_playlists = None
        self.sequential_playlists = load_sequential.load_train_sequential()
        self.initialize_components()


    def initialize_components(self):

        self.train = self.rescale_wrt_insertion_order(self.train)

        self.item_cosineCF_recommender = Cosine_Similarity(self.train, topK=200, shrink=15, normalize=True, mode='cosine')
        self.user_cosineCF_recommender = Cosine_Similarity(self.train.T, topK=200, shrink=15, normalize=True, mode='cosine')
        self.svd_recommender = PureSVDRecommender(self.train)
        self.cbf_bpr_recommender = SLIM_BPR_Cython(self.icm.T, positive_threshold=0)
        self.cbf_recommender = Cosine_Similarity(self.icm.T, topK=50, shrink=10, normalize=True, mode='cosine')
        self.item_rp3b_recommender = RP3betaRecommender(self.train)
        self.user_rp3b_recommender = RP3betaRecommender(self.train.T)
        self.bpr_mf = BPR_matrix_factorization(factors=800, regularization=0.01, learning_rate=0.01, iterations=300)
        self.ials_cg_mf = IALS_CG(iterations=15, calculate_training_loss=True, factors=500, use_cg=True, regularization=1e-3)
        self.lightfm = LightFM_Recommender(self.train, self.icm, no_components=200)

    def fit(self):

        self.svd_latent_x, self.svd_latent_y = self.svd_recommender.fit(num_factors=500)
        self.min_svd = np.dot(self.svd_latent_x, self.svd_latent_y).min()
        self.cbf_bpr_w = self.cbf_bpr_recommender.fit(epochs=10, topK=200, batch_size=20, sgd_mode='adagrad', learning_rate=1e-2)
        self.item_cosineCF_w = self.item_cosineCF_recommender.compute_similarity()
        self.user_cosineCF_w = self.user_cosineCF_recommender.compute_similarity()
        self.cbf_w = self.cbf_recommender.compute_similarity()
        self.item_rp3b_w = self.item_rp3b_recommender.fit()
        self.user_rp3b_w = self.user_rp3b_recommender.fit()
        self.ials_cg_mf.fit(40*self.train.T)
        self.ials_latent_x = self.ials_cg_mf.user_factors.copy()
        self.ials_latent_y = self.ials_cg_mf.item_factors.copy()
        self.min_ials = np.dot(self.ials_latent_x, self.ials_latent_y.T).min()
        self.bpr_mf.fit(self.train.T.tocoo())
        self.bpr_mf_latent_x = self.bpr_mf.user_factors.copy()
        self.bpr_mf_latent_y = self.bpr_mf.item_factors.copy()
        self.lightfm.fit(100)


    def recommend(self, user_id, combiner, at=10):
        user_profile = self.train[user_id, :]

        svd_r = self.svd_latent_x[user_id, :].dot(self.svd_latent_y)
        item_cosineCF_r = user_profile.dot(self.item_cosineCF_w).toarray().ravel()
        user_cosineCF_r = self.user_cosineCF_w[user_id].dot(self.train).toarray().ravel()
        cbf_r = user_profile.dot(self.cbf_w).toarray().ravel()
        cbf_bpr_r = user_profile.dot(self.cbf_bpr_w).toarray().ravel()
        ials_r = np.dot(self.ials_latent_x[user_id], self.ials_latent_y.T + self.min_ials).ravel()
        bpr_mf_r = np.dot(self.bpr_mf_latent_x[user_id], self.bpr_mf_latent_y.T).ravel()
        item_rp3b_r = user_profile.dot(self.item_rp3b_w).toarray().ravel()
        user_rp3b_r = self.user_rp3b_w[user_id].dot(self.train).toarray().ravel()
        lightfm_r = self.lightfm.scores(user_id)

        scores = [
            # [item_bpr_r, self.ensemble_weights["ITEM_BPR"], "ITEM_BPR" ],
            # [user_bpr_r, self.ensemble_weights["USER_BPR"], "USER_BPR" ],
            [svd_r, self.ensemble_weights["SVD"], "SVD"],
            [item_cosineCF_r, self.ensemble_weights["ITEM_CF"], "ITEM_CF" ],
            [user_cosineCF_r, self.ensemble_weights["USER_CF"], "USER_CF" ],
            [ials_r, self.ensemble_weights["IALS"], "IALS" ],
            [cbf_r, self.ensemble_weights["CBF"], "CBF" ],
            [cbf_bpr_r, self.ensemble_weights["CBF_BPR"], "CBF_BPR"],
            [bpr_mf_r, self.ensemble_weights["BPR_MF"], "BPR_MF"],
            [item_rp3b_r, self.ensemble_weights["ITEM_RP3B"], "ITEM_RP3B"],
            [user_rp3b_r, self.ensemble_weights["USER_RP3B"], "USER_RP3B"],
            [lightfm_r, self.ensemble_weights["FM"], "FM"]
            ]

        for r in scores:
            self.filter_seen(user_id, r[0])

        R = combiner.combine(scores, at)
        return R

    def rescale_wrt_insertion_order(self, R):
        R = R.copy()
        R = R.tolil()
        R = R*0.8
        for i in self.sequential_playlists:
            pl = i["id"]
            k = 1
            for j in i["songs"]:
                factor = 1/(k**POPULARITY_SCALING_EXP)
                R[pl, j] = factor*(R[pl,j] + 0.2)
                k += 1
        R = R.tocsr()
        return R
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

        user_cf_rating = self.ensemble_weights["USER_CF"]*self.user_cosineCF_w.dot(self.train)

        user_cf = {
                "min": user_cf_rating.min(),
                "max": user_cf_rating.max(),
                "mean": user_cf_rating.mean(),
            }
        del user_cf_rating
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
        svd_ratings = self.ensemble_weights["SVD"] * (np.dot(self.svd_latent_x, self.svd_latent_y) + self.min_svd)

        svd = {

            "min": svd_ratings.min(),
            "max": svd_ratings.max(),
            "mean": svd_ratings.mean(),
        }
        del svd_ratings


        return {
            "ITEM_CF" : item_cf,
            "USER_CF": user_cf ,
            "SVD" : svd ,
            "IALS" : ials,
            "CBF" : cbf,
            "CBF_BPR" : cbf_bpr
        }
