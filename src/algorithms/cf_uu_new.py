from Base.Similarity.Cython.Compute_Similarity_Cython import Cosine_Similarity
import numpy as np



class P3ALPHANEW:

    def __init__(self, urm_train, urm_test):

        self.train = urm_train.tocsr()
        self.test = urm_test.tocsr()

        self.initialize_components()


    def initialize_components(self):

        self.user_cosineCF_recommender = Cosine_Similarity(self.train.T, topK=200, shrink=5, normalize=True,
                                                           mode='cosine')

    def fit(self):

        self.user_cosineCF_w = self.user_cosineCF_recommender.compute_similarity()

    def recommend(self, user_id, combiner, at=10):

        user_cosineCF_r = self.user_cosineCF_w[user_id].dot(self.train).toarray().ravel()

        scores = [
            [user_cosineCF_r, 1, "USER_CF"]
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
        user_cf_rating = 1 * self.user_cosineCF_w.dot(self.train)

        user_cf = {
            "min": user_cf_rating.min(),
            "max": user_cf_rating.max(),
            "mean": user_cf_rating.mean(),
        }
        del user_cf_rating

        return {
            "USER_CF" : user_cf
        }
