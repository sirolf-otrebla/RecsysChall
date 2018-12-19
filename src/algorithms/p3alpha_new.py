from GraphBased.P3alphaRecommender import P3alphaRecommender
import numpy as np



class P3ALPHANEW:

    def __init__(self, urm_train, urm_test):

        self.train = urm_train.tocsr()
        self.test = urm_test.tocsr()

        self.initialize_components()


    def initialize_components(self):

        self.p3alphaUU = P3alphaRecommender(self.train.T)

    def fit(self):

        self.p3alphaUU.fit(topK=200, alpha=0.3)
        self.p3alphaUU_w = self.p3alphaUU.W_sparse

    def recommend(self, user_id, combiner, at=10):

        p3alphaUU_r = self.p3alphaUU_w[user_id].dot(self.train).toarray().ravel()

        scores = [
            [p3alphaUU_r, 1, "P3_ALPHA"]
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
        p3_alpha_UU_rating = 1*self.p3alphaUU_w.dot(self.train)

        p3_alpha = {

                "min" : p3_alpha_UU_rating.min(),
                "max" : p3_alpha_UU_rating.max(),
                "mean" : p3_alpha_UU_rating.mean(),

            }
        del p3_alpha_UU_rating

        return {
            "P£_ALPHA" : p3_alpha
        }
