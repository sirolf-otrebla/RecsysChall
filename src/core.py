from src import build_data
from src import data
from src import recommender as r


dm = build_data.build()
rec = r.TopPop()
rec.fit(dm.getURM_COO())
recommended = rec.recommend(1,10)
print(recommended);
