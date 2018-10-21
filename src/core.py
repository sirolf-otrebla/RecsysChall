from src import build_data
from src import data
from src import recommender as r
import pandas
import numpy

dm = build_data.build_urm()
target = build_data.loadTarget()
rec = r.CBF_coldstart(k=50)
rec.fit()
recommended = rec.recommendAll(target, 10)
playlists = recommended[:,0]
recommended = numpy.delete(recommended, 0, 1)
i = 0
res_fin = []
for j in recommended:
    res = ''
    for k in range(0,len(j)):
        res = res + '{0} '.format(j[k])
    res_fin.append(res)
    i = i+1
d = {'playlist_id': playlists, 'track_ids' : res_fin}
df = pandas.DataFrame(data=d,index=None)
df.to_csv("../results/recommended7.csv", index=None)
i = 1+1
