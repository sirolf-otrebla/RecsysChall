from lightfm import LightFM
from lightfm.evaluation import auc_score
from refactored_code.utils import train_test_holdout, load_urm, load_icm, evaluate_csv
import numpy as np
import pandas as pd


class LightFM_Recommender:

    def __init__(self, train):
        self.train = train
        self.icm = load_icm()
        self.model = LightFM(loss='warp', k=5, n=10, item_alpha=0, user_alpha=0)

        self.pid_array = np.arange(train.shape[1], dtype=np.int32)

    def fit(self):
        self.model.fit(epochs=300, interactions=self.train, item_features=self.icm, verbose=True)

    def filter_seen(self, user_id, scores):

        start_pos = int(self.train.indptr[user_id])
        end_pos = int(self.train.indptr[user_id + 1])

        user_profile = self.train.indices[start_pos:end_pos]

        scores[user_profile] = -1000000 #-np.inf
        return scores

    def recommend(self, user_id, at=10):
        scores = self.model.predict(user_id, self.pid_array, item_features=self.icm)
        scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def recommendALL(self, userList, at=10):
        res = np.array([])
        n=0
        for i in userList:
            n+=1
            recList = self.recommend(i[0], at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res


def before():
    model = LightFM(loss='warp')
    train, test = train_test_holdout(load_urm())
    model.fit(train, epochs=3, num_threads=2)

    pid_array = np.arange(test.shape[1], dtype=np.int32)
    a = model.predict(7, pid_array)
    print(a)
    test_auc = auc_score(model, test, train_interactions=train, num_threads=2).mean()
    print(test_auc)


if __name__ == '__main__':
    train, test = train_test_holdout(load_urm())
    recsys = LightFM_Recommender(load_urm())
    target = pd.read_csv('./data/target_playlists.csv', index_col=False)

    recsys.fit()

    recommended = recsys.recommendALL(target.values)

    playlists = recommended[:, 0]
    recommended = np.delete(recommended, 0, 1)
    i = 0
    res_fin = []
    for j in recommended:
        res = ''
        for k in range(0, len(j)):
            res = res + '{0} '.format(j[k])
        res_fin.append(res)
        i = i + 1
    d = {'playlist_id': playlists, 'track_ids': res_fin}
    df = pd.DataFrame(data=d, index=None)
    df.to_csv("./results/lightfm_test.csv", index=None)

    evaluate_csv(test, "./results/lightfm_test.csv")


