from Base.Evaluation.Evaluator import SequentialEvaluator
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from refactored_code.utils import load_urm, new_train_test_holdout, load_icm, evaluate_csv
import numpy as np
import pandas as pd

if __name__ == '__main__':
    urm_train, urm_test = new_train_test_holdout(load_urm())
    slim = SLIM_BPR_Cython(urm_train, positive_threshold=0, symmetric=True)

    slim.fit(epochs=10, topK=300, batch_size=500, sgd_mode='adagrad', learning_rate=1e-4)

    target = pd.read_csv('./data/target_playlists.csv', index_col=False)
    recommended = slim.recommendALL(userList=target.values)

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
    df.to_csv("./results/results_als_tuning.csv", index=None)

    evaluate_csv(urm_test, "./results/results_als_tuning.csv")
