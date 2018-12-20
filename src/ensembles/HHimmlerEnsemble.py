from implicit.bpr import BayesianPersonalizedRanking as BPR_matrix_factorization
from implicit.als import AlternatingLeastSquares as IALS_CG
import numpy as np



class HHimmlerEnsemble:

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
        self.bpr_mf = BPR_matrix_factorization(factors=200, regularization=0.00000, learning_rate=0.01, iterations=65)
        self.ials_cg_mf = IALS_CG(iterations=15, calculate_training_loss=True, factors=500, use_cg=True, regularization=1e-3)

    def fit(self):
        self.bpr_mf.fit(self.train.T.tocoo())
        self.ials_cg_mf.fit(40*self.train.T)
        self.bpr_mf_latent_x = self.bpr_mf.user_factors.copy()
        self.bpr_mf_latent_y = self.bpr_mf.item_factors.copy()
        self.ials_cg_mf_latent_x = self.ials_cg_mf.user_factors.copy()
        self.ials_cg_mf_latent_y = self.ials_cg_mf.item_factors.copy()


    def recommend(self, user_id, combiner, at=10):
        bpr_mf_r = np.dot(self.bpr_mf_latent_x[user_id], self.bpr_mf_latent_y.T).ravel()
        ials_cg_mf_r = np.dot(self.ials_cg_mf_latent_x[user_id], self.ials_cg_mf_latent_y.T).ravel()


        scores = [
            # [bpr_mf_r, self.ensemble_weights["BPR_MF"], "BPR_MF"],
            [ials_cg_mf_r, 1, "IALS_CG"]
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
            bpr = self.bpr_mf.recommend(user_items=self.train, userid=i, N=at, recalculate_user=False)
            ials = self.ials_cg_mf.recommend(userid=i, user_items=self.train, N=10)
            list = [x[0] for x in ials]
            recList = np.array(list)
            tuple = np.concatenate(([i], recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res

    def get_component_data(self):
        print('cyka')
