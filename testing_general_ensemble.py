from refactored_code.general_ensemble import GeneralEnsemble
from refactored_code.utils import load_random_urms, load_icm, evaluate_csv, load_urm
import pandas as pd
import numpy as np


if __name__ == '__main__':
    urm_train, urm_test = load_random_urms()
    icm = load_icm()
    general = GeneralEnsemble(urm_train, urm_test, icm, ro=0.6)
    general.fit(0.03)

    target = pd.read_csv('./data/target_playlists.csv', index_col=False)
    recommended = general.recommendALL(target.values)

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
    df.to_csv("./results/TESTING_GENERAL_ENSEMBLE.csv", index=None)

    evaluate_csv(urm_test, "./results/TESTING_GENERAL_ENSEMBLE.csv")
