import itertools
import random
import math
import numpy as np
from scipy.sparse import data
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
# import simple_good_turing
from sklearn.metrics import mean_squared_error
from statistics import mean
import os

import itertools
import random
import math
import numpy as np
import pandas as pd
from scipy.sparse import data
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statistics import mean

#import clean_data
import csv

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import DBSCAN, KMeans, MeanShift, AgglomerativeClustering

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

KNN1 = KNeighborsClassifier(n_neighbors=1)
KNN3 = KNeighborsClassifier(n_neighbors=3)
KNN11 = KNeighborsClassifier(n_neighbors=11)

randomForest1 = RandomForestClassifier(n_estimators=1)
randomForest5 = RandomForestClassifier(n_estimators=5)
randomForest10 = RandomForestClassifier(n_estimators=10)
randomForest25 = RandomForestClassifier(n_estimators=25)
randomForest100 = RandomForestClassifier(n_estimators=100)

naiveBayesClassifier = GaussianNB()
adaboostClassifier = AdaBoostClassifier()
gradientBoostingClassifier = GradientBoostingClassifier()
decisionTreeClassifier = DecisionTreeClassifier()
quadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()
logisticRegression = LogisticRegression(max_iter=500)
SGDClassifier_hinge = SGDClassifier()
SGDClassifier_log = SGDClassifier(loss='log')
SVC_linear_kernel = SVC(kernel="linear")
SVC_linear = LinearSVC()
SVC_rbf = SVC()
MLPclf_1 = MLPClassifier(max_iter=500)
MLPclf_3 = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=500)

from new_ldm_inductive import *
import server_setup as setup


def get_LDM_PD(list_of_clf, X_train, X_test, y_train, classes=[0,1], num_datasets=5, num_repeat=1, proportion_of_dataset=0.1, sparse=True, data_generation=random_uniform):
    LDM_l = []
    PD_l = []
    for clf in list_of_clf:
        LDM = getLDM(clf, X_train, X_test, y_train, classes = classes, num_datasets=num_datasets, num_repeat=num_repeat, proportion_of_dataset=proportion_of_dataset, sparse=sparse, data_generation=data_generation)
        LDM_l.append(LDM)
        PD_l.append(computePD(LDM, sparse=sparse))
    return LDM_l, PD_l
dim_reduc_l = ["PCA", "UMAP-n_neighbors=20", "UMAP-n_neighbors=15", "UMAP-n_neighbors=10", "UMAP-n_neighbors=5"]
cluster_alg_l = ["DBSCAN-eps=0.35-min_samples=3", "DBSCAN-eps=0.25-min_samples=3", "DBSCAN-eps=0.5-min_samples=3", "DBSCAN-eps=0.10-min_samples=3"
                    , "AgglomerativeClustering-n_clusters=2", "AgglomerativeClustering-n_clusters=4", "AgglomerativeClustering-n_clusters=8",
                    "MeanShift"]
def generate_paths(dim_reduc_l, cluster_alg_l):
    paths = []
    for dim_reduc_raw in dim_reduc_l:
        dim_reduc = dim_reduc_raw.split("-")
        dim_reduc_function = dim_reduc[0]
        dim_reduc_parameters = dim_reduc[1:] #might give an error
        dim_reduc_parameters = {parameter.split("=")[0]:parameter.split("=")[1] for parameter in dim_reduc_parameters}
        print(dim_reduc_parameters)
        for cluster_alg_raw in cluster_alg_l:
            cluster_alg = cluster_alg_raw.split("-")
            cluster_alg_function = cluster_alg[0]
            cluster_alg_parameters = cluster_alg[1:]
            cluster_alg_parameters = {parameter.split("=")[0]:parameter.split("=")[1] for parameter in cluster_alg_parameters}
            print(cluster_alg_parameters)
            paths.append(dim_reduc_raw + "|" + cluster_alg_raw)
    return paths

def convert_str(n):
    if "." in n:
        return float(n)
    else:
        return int(n)

def generate_plots(pD_vectors,clf_names, list_of_clf, paths, base, markers, dim_reduc_function_dict, cluster_alg_function_dict):
    for path in paths:
        dim_reduc = path.split("|")[0]
        dim_reduc = dim_reduc.split("-")
        dim_reduc_function = dim_reduc[0]
        dim_reduc_parameters = dim_reduc[1:]
        print(dim_reduc_parameters)
        dim_reduc_parameters = {parameter.split("=")[0]:convert_str(parameter.split("=")[1]) for parameter in dim_reduc_parameters}

        cluster_alg = path.split("|")[1]
        cluster_alg = cluster_alg.split("-")
        cluster_alg_function = cluster_alg[0]
        cluster_alg_parameters = cluster_alg[1:]
        cluster_alg_parameters = {parameter.split("=")[0]:convert_str(parameter.split("=")[1]) for parameter in cluster_alg_parameters}
        
        labels, list_of_PD = setup.cluster(pD_vectors, cluster_alg_function_dict[cluster_alg_function], list_of_clf, cluster_alg_parameters)
        visualDim = 2
        reduceDim = True
        dim_reduc_parameters = {**{"n_components":2} , **dim_reduc_parameters}
        setup.cluster_plot(list_of_PD, clf_names, os.path.join(base, path).replace("|", "-") , markers, labels, visualDim, reduceDim, dim_reduc_function_dict[dim_reduc_function], dim_reduc_parameters)
        setup.analyzeResults(os.path.join(base, path).replace("|", "-"), labels, clf_names) #might need to save this to some sort of doc



def main():
    current_folders = os.listdir()
    runs = list(filter(lambda name: name[:3]=="run", current_folders))
    if len(runs)==0:
        run_path = "run0"
    else:
        runs = [int(run[3:]) for run in runs]
        run_path = "run" + str(max(runs)+1)
    os.mkdir(run_path)
    cluster_path = os.path.join(run_path, "clusters")
    os.mkdir(cluster_path)
    dataset = pd.read_csv("EEG_Eye_State.csv")
    values = dataset.values
    X, y = values[:, :-1], values[:, -1]
    markers = {'KNN1': "$a$", 'KNN3': "$b$", 'KNN11': "$c$", 'randomForest1': "$d$", 'randomForest5': "$e$", 
            'randomForest10': "$f$", 'randomForest25': "$g$",'randomForest100': "$h$",'naiveBayesClassifier': "$i$",
            'adaboostClassifier': "$j$",'gradientBoostingClassifier': "$k$", 'decisionTreeClassifier': "$l$", 
            'quadraticDiscriminantAnalysis': "$m$", 'logisticRegression': "$n$", 'SGDClassifier_hinge': "$o$", 
            'SGDClassifier_log': "$p$", 'MLPclf_1': "$q$", 'MLPclf_3': "$r$", "SVC_linear": "$s$", "SVC_rbf": "$t$" }
    dim_reduc_function_dict = {"PCA": PCA, "UMAP": UMAP}
    cluster_alg_function_dict = {"DBSCAN":DBSCAN, "AgglomerativeClustering":AgglomerativeClustering, "MeanShift":MeanShift}
    num_repeats = 3
    list_of_clf=[decisionTreeClassifier]*num_repeats + [KNN1]*num_repeats + [KNN11]*num_repeats
    clf_names = ["decisionTreeClassifier"]*num_repeats + ["KNN1"]*num_repeats + ["KNN11"]*num_repeats

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)
    LDM_l, PD_l = get_LDM_PD(list_of_clf, X_train, X_test, y_train, num_datasets=505, num_repeat=1, proportion_of_dataset=0.04)
    
    with open(os.path.join(run_path, 'ldm.txt'), 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(LDM_l)

    with open(os.path.join(run_path, 'pd.txt'), 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(PD_l)
    
    dim_reduc_l = ["PCA", "UMAP-n_neighbors=20", "UMAP-n_neighbors=15", "UMAP-n_neighbors=10", "UMAP-n_neighbors=5"]
    cluster_alg_l = ["DBSCAN-eps=0.35-min_samples=3", "DBSCAN-eps=0.25-min_samples=3", "DBSCAN-eps=0.5-min_samples=3", 
                    "DBSCAN-eps=0.10-min_samples=3", "AgglomerativeClustering-n_clusters=2", 
                    "AgglomerativeClustering-n_clusters=4", "AgglomerativeClustering-n_clusters=8","MeanShift"]
    paths = generate_paths(dim_reduc_l, cluster_alg_l)
    generate_plots(PD_l,clf_names, list_of_clf, paths, cluster_path, markers, dim_reduc_function_dict, cluster_alg_function_dict)


if __name__ == "__main__":
    main()