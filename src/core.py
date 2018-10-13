from src import build_data
from src import data
from src import recommender as r
import pandas

dm = build_data.build()
target = build_data.loadTarget()
rec = r.TopPop()
rec.fit(dm.getURM_COO())
recommended = rec.recommendAll(target, 10)
df = pandas.DataFrame([ 'playlist_id', 'track_ids'])
df.append(pandas.DataFrame(recommended))
df.to_csv("../results/recommended.csv")
print(recommended);
