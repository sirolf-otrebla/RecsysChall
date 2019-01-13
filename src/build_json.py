import json

CFI =   [1, 2, 5, 10, 12, 15, 20]                        #[16.0]
CFU =   [7]                         #[7.0]
BPRI=   [1, 2, 5, 10, 12, 15, 20]                        #[16.0]
BPRU=   [8]                         #[8.0]
BPRC=   [64]                        #[64]
CBF =   [7]                         #[7]
IALS=   [26]                        #[14]
BPRM =  [6]
IRP3B = [1, 2, 5, 10, 12, 15, 20]                        #0.87
URP3B = [0]                         #
alpha = [1]
P3ALPHA = [1, 2, 5, 10, 12, 15, 20]


arr = []

for c in CFI:
    for d in CFU:
        for e in BPRI:
            for f in BPRU:
                for g in BPRC:
                    for h in CBF:
                        for i in IALS:
                            for j in BPRM:
                                for k in IRP3B:
                                    for l in URP3B:
                                        for m in alpha:
                                            for n in P3ALPHA:
                                                arr.append({
                                #"USER_CF" : d,
                                #"USER_BPR" : f,
                                "ITEM_CF" : c,
                                "ITEM_BPR" : e,
                                #"CBF" : h,
                                #"IALS" : i,
                                #"CBF_BPR" : g,
                                #"BPR_MF" : j,
                                "ITEM_RP3B": k,
                                #"USER_RP3B" : l,
                                #'ALPHA' : m,
                                "P3_ALPHA": n
                            })
encoder = json.JSONEncoder()


json = encoder.encode(arr)
file = open("./parameters/ItemEnsemble.json", mode='w')
file.write(json)
file.close()
