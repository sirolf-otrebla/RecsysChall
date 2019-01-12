from src.ensembles.BMussoliniEnsemble import BMussoliniEnsemble
from src.combiners.linearCombiner import linearCombiner
from refactored_code.utils import load_random_urms, load_icm, evaluate_csv, load_urm, new_train_test_holdout, train_test_holdout
import json
import io
import pandas as pd
import numpy as np
from scipy import sparse as sps
import gc

MODE = "TEST"
def load_data():
    URM_text = np.loadtxt('../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    URM = sps.csr_matrix((rating_list, (user_list, item_list)))
    urm_train, urm_test = train_test_holdout(URM)
    icm = load_icm()
    return { "URM_complete" : URM, "URM_test" : urm_test, "URM_train" : urm_train, "ICM" : icm}

def load_parameters(path):
    file = open(path, mode='r')
    json_parameters = json.load(file)
    file.close()
    return  json_parameters

def run_algorithm(data, parameters):
    if MODE == "SUBMIT":
        recommender = BMussoliniEnsemble(data["URM_complete"], data["URM_test"], icm=data["ICM"], parameters=parameters)
    else:
        recommender = BMussoliniEnsemble(data["URM_train"], data["URM_test"], icm=data["ICM"], parameters=parameters)
    recommender.fit()
    target = pd.read_csv('../data/target_playlists.csv', index_col=False)
    target  = target['playlist_id']
    recommended = recommender.recommend_batch(user_list=target, combiner=linearCombiner())
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
    distrib_data = recommender.get_component_data()
    del recommender
    gc.collect()
    return [recommended, df, distrib_data]

def write_results(id, urm_test, dict, data_frame, component_attributes, encoder=None):
    if encoder is None:
        encoder = json.JSONEncoder()
    filename = "../results/BMussolini/result_{0}.csv".format(id)
    data_frame.to_csv(filename, index=None)
    eval = evaluate_csv(urm_test, filename)
    eval_json = encoder.encode(eval)
    param_json = encoder.encode(dict)
    param_file = open('../results/BMussolini/params_{0}.json'.format(id), mode='w')
    param_file.write(param_json)
    param_file.close()
    eval_file = open('../results/BMussolini/eval_{0}.json'.format(id), mode='w')
    eval_file.write(eval_json)
    eval_file.close()
    return  eval

def print_info(dict, eval):
    print("\n \n PARAMETERS : \n " + str(dict) + "\n ############################################################ \n")
    print("\n \n EVALUATION : \n " + str(eval) + "\n ############################################################ \n")

def main():
    data = load_data()
    param_list = load_parameters('./parameters/BMussolini.json')
    encoder = json.JSONEncoder()
    i = 0
    for dict in param_list:
        result = run_algorithm(data, dict)
        eval = write_results(i, data['URM_test'], dict, result[1], result[2], encoder)
        i += 1
        print_info(dict, eval)



if __name__ == "__main__":
    MODE = "TEST"
    main()
