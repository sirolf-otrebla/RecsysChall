from src import build_data
from src.similarities import Cosine_Similarity
from src import recommender

porcodio = build_data.build_icm()

sim = Cosine_Similarity(porcodio, 100)
cristo_il_porco = sim.compute()

a = recommender.CBF_Item_Naive(10)
a.fit(a)
print(a.recommend(0))
