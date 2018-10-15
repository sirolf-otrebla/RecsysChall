from src import build_data
from src import data
from src import recommender as r
import pandas

dm = build_data.build_urm()
target = build_data.loadTarget()
rec = r.TopPop()
rec.fit(dm.getURM_COO())
recommended = rec.recommendAll(target, 10)
#df = pandas.DataFrame([], header=['playlist_id', 'track_ids'])
df = pandas.DataFrame(recommended)
df.to_csv("../results/recommended.csv", index=None)
print(recommended)
