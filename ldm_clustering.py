from sklearn.cluster import DBSCAN
import new_ldm_inductive as ldm_inductive

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import new_ldm_inductive as ldm_inductive
import pandas as pd

dataset = pd.read_csv("EEG_Eye_State.csv", header = None)
values = dataset.values
X, y = values[:, :-1], values[:, -1]
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
adaboostClassifier = AdaBoostClassifier()
randomForest = RandomForestClassifier()
decisionTreeClassifier = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)
classes=[0,1]
num_datasets=205
proportion_of_dataset=0.04

#get a list of PDs for different algorithms
def get_PD(list_of_clf, sgt=True, use_predict_proba=False):
    list_PD = []
    for clf in list_of_clf:
        if use_predict_proba:
            sgt=False
            LDM = ldm_inductive.getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets = num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=False)
        else:
            LDM = ldm_inductive.getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets = num_datasets, proportion_of_dataset=proportion_of_dataset)
               
        if sgt:
            Pd = ldm_inductive.simpleGoodTuring(LDM)
        else:
            Pd = ldm_inductive.computePD(LDM)

        list_PD.append(Pd)
    
    return list_PD

def get_cluster_labels(clustering_method, X):
    clustering = clustering_method().fit(X)
    return clustering.labels_

# Noisy samples are given the label -1.
def cluster(clustering_method, list_of_clf, sgt=True, use_predict_proba=False):
    return get_cluster_labels(clustering_method, get_PD(list_of_clf, sgt=sgt, use_predict_proba=use_predict_proba))


list_of_clf=[model]*10 + [model3]*10 
print(cluster(DBSCAN, list_of_clf))