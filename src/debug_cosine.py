from src import build_data
from src.similarities import Cosine_Similarity

porcodio = build_data.build_icm()

sim = Cosine_Similarity(porcodio, 100)
cristo_il_porco = sim.compute()

sim.topK(10)
