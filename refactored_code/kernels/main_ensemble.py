import pandas as pd
import numpy as np
from refactored_code.utils import load_icm, evaluate_csv, load_random_urms
from refactored_code.general_ensemble import GeneralEnsemble


def main(alpha, beta, URM_train, URM_test):

    ICM = load_icm()

    cf = GeneralEnsemble(URM_train, ICM, 50, epsilon=beta)
    cf.fit(alpha)

    target = pd.read_csv('../../data/target_playlists.csv', index_col=False)
    recommended = cf.recommendALL(target.values)

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
    df.to_csv("./results/resultsEnsembleAll.csv", index=None)

    return evaluate_csv(URM_test, "./results/resultsEnsembleAll.csv")


if __name__ == '__main__':
    train, test = load_random_urms()
    main(0.03, 0.6, train, test)
