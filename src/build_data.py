import pandas as p
import scipy as s

import data


def build():
    train_df = p.read_csv('../data/train.csv')
    train_df.head()
    train_df['rate'] = 1
    URM = train_df.pivot(index="playlist_id", columns="track_id", values="rate").fillna(0).to_sparse(fill_value=0)
    sparseURM = URM.to_sparse()
    cooURM = URM.to_coo()
    dm = data.Data_manager(cooURM);
    return  dm



