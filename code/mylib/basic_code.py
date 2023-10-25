import os
import ot
import random

import pandas as pd
import scipy as sp

from sklearn.metrics import accuracy_score

def function_distance(i, models):
    model1 = pd.read_csv("classes/" + models[i] + "_object.csv", index_col=0)
    coef1 = sp.spatial.distance.cdist(model1.values, model1.values, metric="euclidean")
    coef1 /= coef1.max()  
    for j in range(len(models)):
        if i == j:
            return 0
        else:
            if coef1 is None:
                model1 = pd.read_csv("classes/" + models[i] + "_object.csv", index_col=0)
                coef1 = sp.spatial.distance.cdist(model1.values, model1.values, metric="euclidean")
                coef1 /= coef1.max()    
            if coef2 is None:
                model2 = pd.read_csv("classes/" + models[j] + "_object.csv", index_col=0)
                coef2 = sp.spatial.distance.cdist(model2.values, model2.values, metric="euclidean")
                coef2 /= coef2.max()    
            p = ot.unif()
            q = ot.unif()
            _, log0 = ot.gromov.gromov_wasserstein(coef1, coef2, p, q, 'square_loss', log=True)
            return log0["gw_dist"]

models = [scale for scale in os.listdir("objects/") if "csv" in scale and not "_bin" in scale]
similarities = []
for i in range(len(models)):
    similarities.append(function_distance(i, models))
similarities = pd.DataFrame(similarities, columns=models, index=models)
similarities.to_csv("imagenet_gromov.csv")

pairwise_fidelity = []
for m1 in models:
    arr = []
    for m2 in models:
        pred1 = pd.read_csv("classes/" + m1 + "_pred.csv", index_col=0).values
        pred2 = pd.read_csv("classes/" + m2 + "_pred.csv", index_col=0).values
        arr.append(accuracy_score(pred1, pred2))
    pairwise_fidelity.append(arr)
fidelity = pd.DataFrame(pairwise_fidelity, columns=models, index=models)
fidelity.to_csv("imagenet_sim.csv")
