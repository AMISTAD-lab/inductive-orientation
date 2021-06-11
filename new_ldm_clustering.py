from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, MeanShift

import new_ldm_inductive as ldm_inductive

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:gray', 'tab:red',
         'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan']
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

# DBSCAN gives noisy samples the label -1.
'''
cluster
Assumes that dimensionality of the PD vectors will be reduced
Note: Dimensionality of PD vectors will NOT always be reduced to 2 dimensions...
Parameters:
    1. param_dimReduc = a dictionary containing key-value pairs for dimensionality reduction 
    2. param_clustering = a dictionary containing key-value pairs for clustering algorithm
'''
def cluster(clustering_method, dimReduc_method, list_of_clf, param_clustering = {}, param_dimReduc = {},
            sgt=True, use_predict_proba=False):
    
    print("param_dimReduc", param_dimReduc)
    print("param_clustering", param_clustering)
 
    list_of_PD = get_PD(list_of_clf, sgt=sgt, use_predict_proba=use_predict_proba)
    
    #reduce the dimensions of PD vector
    array_of_PD_reduced = dimReduc_method(**param_dimReduc).fit_transform(np.array(list_of_PD))
    
    #cluster lower dimensional PD vectors 
    clustering = clustering_method(**param_clustering).fit(array_of_PD_reduced)
    labels = clustering.labels_
    
    return labels, array_of_PD_reduced
 
'''
cluster_plot
Parameters:
    1. X = embedded_PD's
    2. clf_names = names of the classifiers
    3. labels = different clusters formed
    4. visualDim = # of dimensions that clusters + Pd vectors will be visualized as (either 2 or 3)
Note: X may not be in 2D, 3D
    5. reduceDim = False if X = embedded_PD's do NOT need to be reduced 
                 = True if X = embedded_PD's need to be reduced 
    6. *args = dim_Reduction_method 
    7. *kwargs = key-value parameters for dim_Reduction 
'''
def cluster_plot(X, clf_names, labels = [], visualDim = 2, reduceDim = False, *args, **kwargs):
    
    #X needs to be reduced to a lower dimension 2D
    print("args: ",  args[0])
    print("type args: ",  type(args[0]))
    if reduceDim:
        X = args[0](**kwargs).fit_transform(np.array(X))
    
    #check that the dimensions of the reduced PD vectors is equal to the dimension
    if len(X[0]) != visualDim:
        raise Exception("Dimension of Reduced PD vectors must equal to the dimension being visualized!")
        
    if visualDim == 2:
        
        if len(labels) != 0:
            df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'label': labels, 'clf_name': clf_names})
            print(df)
            sns.scatterplot(data=df, x="x1", y="x2", hue="label", style="clf_name",  legend = "full", s = 100)
        else:
            df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'clf_name': clf_names})
            print(df)
            sns.scatterplot(data=df, x="x1", y="x2", style="clf_name",  legend = "full", s =100)  
        plt.legend(bbox_to_anchor=(1.5, 1), borderaxespad=0)

    elif visualDim == 3:
        
        #allTypesOfMarkers = dictionary
        allTypesOfMarkers = list(mpl.markers.MarkerStyle.markers.keys())
        #unique classifers: unique_clfs
        unique_clfs = np.unique(clf_names)
        markerStyles = allTypesOfMarkers[:len(unique_clfs)]
        
        #map classifier to unique marker
        mapClfsToMarker = {}
        for c, m in zip(unique_clfs, markerStyles):
            mapClfsToMarker[c] = m
            
        print("Map Classifier to Marker: ", mapClfsToMarker)
        
        clfMarkerStyle = []
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        
        if len(labels) != 0:
            color_label = colors[:len(np.unique(labels))] 
             #print("labels:",  labels)
        
            mapLabelsToColor = {}
            for l, c in zip(np.unique(labels), color_label):
                mapLabelsToColor[l] = c
        
            print("Map Labels to Color: ", mapLabelsToColor)
        
            clfColor = []
            for c, l in zip(clf_names, labels):
                clfMarkerStyle.append(mapClfsToMarker[c])
                clfColor.append(mapLabelsToColor[l])
        
            #print("clfColor", clfColor)

            for i in range(len(clf_names)):
                ax.scatter(x[i], y[i], z[i], c = clfColor[i], marker= clfMarkerStyle[i], s = 100)
        else:
            for c in clf_names:
                clfMarkerStyle.append(mapClfsToMarker[c])
            
            for i in range(len(clf_names)):
                ax.scatter(x[i], y[i], z[i], marker= clfMarkerStyle[i], color = "red", s = 100)
                
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    else:
        print(visualDim, "canNOT be visualized.")
    
if __name__ == "__main__":
    #would it make a difference if took dimReduc from high -> low dim and cluster or 
    # if PCA from high to med dim cluster -> PCA for visual
    '''
    list_of_clf=[model]*10 + [model3]*10 
    clf_names = ["KNN1"]*10 + ["KNN3"]*10
    labels, list_of_PD = cluster(KMeans, PCA, list_of_clf, {"n_components": 4}, {"n_clusters":2, "random_state": 0})
    print("Was dimensionality reduced:", len(list_of_PD[0]))
    cluster_plot(list_of_PD, labels, clf_names, 3, True, PCA, n_components = 3)
    '''

    # # only visualization in 3d
    # list_of_clf=[model]*3 + [model3]*3 + [model10]*3
    # clf_names = ["KNN1"]*3 + ["KNN3"]*3 + ["KNN10"]*3
    # list_of_PD = get_PD(list_of_clf)
    # visualDim=3
    # reduceDim=True
    # labels=[]
    # cluster_plot(list_of_PD, clf_names, labels, visualDim, reduceDim, PCA, n_components=visualDim)

    # # only visualization in 2d
    # list_of_clf=[model]*3 + [model3]*3 + [model10]*3
    # clf_names = ["KNN1"]*3 + ["KNN3"]*3 + ["KNN10"]*3
    # list_of_PD = get_PD(list_of_clf)
    # visualDim=2
    # reduceDim=True
    # labels=[]
    # cluster_plot(list_of_PD, clf_names, labels, visualDim, reduceDim, PCA, n_components=visualDim)

    # #clustering + visualization in 3d
    # list_of_clf=[model]*3 + [model3]*3 + [model10]*3
    # clf_names = ["KNN1"]*3 + ["KNN3"]*3 + ["KNN10"]*3
    # param_clustering={"n_clusters":3, "random_state": 0}
    # param_dimReduc={"n_components": 4}
    # labels, list_of_PD = cluster(KMeans, PCA, list_of_clf, param_clustering, param_dimReduc)
    # print("Was dimensionality reduced:", len(list_of_PD[0]))
    # visualDim=3
    # reduceDim=True
    # labels=labels
    # cluster_plot(list_of_PD, clf_names, labels, visualDim, reduceDim, PCA, n_components = visualDim)

    # #clustering + visualization in 2d
    # list_of_clf=[model]*3 + [model3]*3 + [model10]*3
    # clf_names = ["KNN1"]*3 + ["KNN3"]*3 + ["KNN10"]*3
    # param_clustering={"n_clusters":3, "random_state": 0}
    # param_dimReduc={"n_components": 4}
    # labels, list_of_PD = cluster(KMeans, PCA, list_of_clf, param_clustering, param_dimReduc)
    # print("Was dimensionality reduced:", len(list_of_PD[0]))
    # visualDim=2
    # reduceDim=True
    # labels=labels
    # cluster_plot(list_of_PD, clf_names, labels, visualDim, reduceDim, PCA, n_components = visualDim)