import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score


def classify(o, model_w):
    cosine = [np.dot(o,model_w.loc[f]) / (np.linalg.norm(o) * np.linalg.norm(model_w.loc[f])) for f in model_w.index]
    return model_w.index[cosine.index(max(cosine))]

def pred(model_o, model_w):
    pred = []
    for o in model_o:
        cosine = [np.dot(o,model_w.loc[f]) / (np.linalg.norm(o) * np.linalg.norm(model_w.loc[f])) for f in model_w.index]
        pred.append(model_w.index[cosine.index(max(cosine))])
    return pred

def fidelity(model_o, model_w, pred):
    model_pred_idx = pred["0"].apply(lambda i : model_w.index[i])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(model_w.values, model_w.index.values)
    knn_class_pred = knn.predict(model_o.values)
    knn_acc = accuracy_score(model_pred_idx,knn_class_pred)
    knn_f1 = f1_score(model_pred_idx,knn_class_pred,average="weighted")
    print(f"Euclidean Accuracy {knn_acc} KNN F1 score {knn_f1}")

    dt = DecisionTreeClassifier()
    dt.fit(model_w.values, model_w.index.values)
    dt_class_pred = dt.predict(model_o.loc[:,model_o.columns != "class"].values)
    dt_acc = accuracy_score(model_pred_idx,dt_class_pred)
    dt_f1 = f1_score(model_pred_idx,dt_class_pred,average="weighted")
    print(f"DTree Accuracy {dt_acc} KNN F1 score {dt_f1}")

    cos_pred = pred(model_o.values,model_w)
    cos_acc = accuracy_score(model_pred_idx,cos_pred)
    cos_f1 = f1_score(model_pred_idx,cos_pred,average="weighted")
    print(f"COS Acc {cos_acc} f1 {cos_f1}")
    return knn_acc, cos_acc

def seperation(model_w):
    unique = len({tuple(model_w.loc[i].values.tolist()) for i in model_w.index})
    separation = unique / model_w.shape[0]
    print(f"{unique} of {model_w.shape[0]} are distinct: separation: {separation} ")
    return separation
