from sklearn import datasets
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math

'''set dataset'''
dataset = pd.read_csv("datasets/EEG_Eye_State.csv", header = None)
values = dataset.values
X, y = values[:, :-1], values[:, -1]

classes=[0,1]
num_datasets=205
proportion_of_dataset=0.04

# DBSCAN gives noisy samples the label -1.
'''
cluster
Assumes that dimensionality of the PD vectors will be reduced
Note: Dimensionality of PD vectors will NOT always be reduced to 2 dimensions...
Parameters:
    1. param_dimReduc = a dictionary containing key-value pairs for dimensionality reduction 
    2. param_clustering = a dictionary containing key-value pairs for clustering algorithm
'''
def cluster(list_of_PD, clustering_method, list_of_clf, param_clustering = {}, dimReduc_method=None, param_dimReduc = {},
            sgt=False, use_predict_proba=False):
    
    if dimReduc_method != None:
        print("param_dimReduc", param_dimReduc)
        print("param_clustering", param_clustering)
        #reduce the dimensions of PD vector
        list_of_PD = dimReduc_method(**param_dimReduc).fit_transform(np.array(list_of_PD))
    
    #cluster lower dimensional PD vectors 
    clustering = clustering_method(**param_clustering).fit(list_of_PD)
    labels = clustering.labels_
    
    return labels, list_of_PD
 
'''
cluster_plot
Parameters:
    1. X = embedded_PD's
    2. clf_names = names of the classifiers
    3. filepath = where to save the pdf
    3. labels = different clusters formed
    4. visualDim = # of dimensions that clusters + Pd vectors will be visualized as (either 2 or 3)
Note: X may not be in 2D, 3D
    5. reduceDim = False if X = embedded_PD's do NOT need to be reduced 
                 = True if X = embedded_PD's need to be reduced 
    6. *args = dim_Reduction_method 
    7. *kwargs = key-value parameters for dim_Reduction 
'''
def cluster_plot(X, clf_names, dataset_name, filepath, markers, labels = [], visualDim = 2, reduceDim = False, dimReduc_method=None, param_dimReduc = {}):   
    #fixes the color for the labels
    palette = ["#F23030","#F2762E","#F2D338","#25C7D9","#248EA6","#324DBE","#800080","#3C005A","#DB7093"]
    palette = itertools.cycle(palette)

    #X needs to be reduced to a lower dimension 2D
    if reduceDim:
        print("args: ",  dimReduc_method)
        print("type args: ",  type(dimReduc_method))
        print("param_dimReduc: ", param_dimReduc)
        X = dimReduc_method(**param_dimReduc).fit_transform(np.array(X))
    
    #check that the dimensions of the reduced PD vectors is equal to the dimension
    print("X", X)
    print("visualDim", visualDim)
    print("len(X[0])", len(X[0]))
    if len(X[0]) != visualDim:
        raise Exception("Dimension of Reduced PD vectors must equal to the dimension being visualized!")
        
    if visualDim == 2:
        
        if len(labels) != 0:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.set_position([0.1,0.1,0.8,0.8])
            df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'label': labels, 'clf_name': clf_names})
            df["clf_name_label"] = df["clf_name"] +" cluster: " + df['label'].astype("str")
            df["markers"] = df["clf_name"].replace(markers)
            plot_markers = {df["clf_name_label"][i]: df["markers"][i] for i in range(len(df))}
            sns.scatterplot(data=df, x="x1", y="x2", hue="clf_name_label",  style="clf_name_label", legend = "full", s = 100, ax=ax, markers=plot_markers, palette=palette)
            plt.legend(bbox_to_anchor=(1.05, 0.1), borderaxespad=0, loc="lower left")
            plt.title(dataset_name)
            plt.savefig(filepath + ".pdf", bbox_inches="tight", pad_inches=0)
            
            
            #commented out the labels specific plots; problem with len of platte
            #for label in np.unique(np.array(labels)):
              #print(label)
              #df_with_label = df[df["label"] == label]
              #fig, ax = plt.subplots(figsize=(5,5))
              #ax.set_position([0.1,0.1,0.8,0.8])
              #sns.scatterplot(data=df_with_label, x="x1", y="x2", hue="label", legend="full", style="clf_name", s = 100, ax=ax, markers=markers, palette=palette)
              # handles, labels = plt.gca().get_legend_handles_labels()
              # by_label = dict(zip(labels, handles))
              # plt.legend(by_label.values(), by_label.keys())
              #plt.legend(bbox_to_anchor=(1.05, 0.1), borderaxespad=0, loc="lower left")
              #plt.savefig(filepath + "-" + str(label) +".pdf", bbox_inches="tight", pad_inches=0)
        else:
            raise Exception("This part of the code is not modified")
            fig, ax = plt.subplots(figsize=(10,10))
            ax.set_position([0.1,0.1,0.5,0.8])
            df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'clf_name': clf_names})
            #print(df)
            sns.scatterplot(data=df, x="x1", y="x2", style="clf_name",  legend = "full", s =100, ax=ax)  
            plt.legend(bbox_to_anchor=(1, 0.5), borderaxespad=0)
            plt.savefig(filepath + ".pdf", bbox_inches="tight", pad_inches=0)
            for label in np.unique(np.array(labels)):
              df_with_label = df[df["label"] == label]
              sns.scatterplot(data=df_with_label, x="x1", y="x2", style="clf_name",  legend = "full", s = 100)
              plt.legend(bbox_to_anchor=(1, 0.5), borderaxespad=0)
              plt.savefig(filepath + str(label) +".pdf", bbox_inches="tight", pad_inches=0)

    elif visualDim == 3:
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:gray', 'tab:red',
         'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan']

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

def analyzeResults(path, label, clfNames):
  tableClusters = pd.DataFrame({'Labels': label, 'Algorithms': clfNames})
  tableClusters = tableClusters.groupby(['Labels','Algorithms'])['Algorithms'].count()
  tableClusters.to_csv(path, index=True)
    
def plotPairwiseDist_max(list_of_PDs, clf_Names, filepath):
  temp_list_of_PDs = list_of_PDs[:]
  searchSpaceSize = len(temp_list_of_PDs[0])
  uniform_PD = [(1 / searchSpaceSize) for i in range(searchSpaceSize)]
  temp_list_of_PDs.append(uniform_PD)
  temp_clf_Names = clf_Names + ["UniformRandomSampling"]
  temp_list_of_PDs = np.array(temp_list_of_PDs)
  pairwise_Dist = np.sum((temp_list_of_PDs[:, np.newaxis, :] - temp_list_of_PDs[np.newaxis, :, :])**2, axis = -1)
  df_pairwise_Dist = pd.DataFrame((pairwise_Dist / np.max(pairwise_Dist)), index = temp_clf_Names, columns = temp_clf_Names).round(3)
  fig, ax = plt.subplots(figsize=(20,20))
  sns.heatmap(df_pairwise_Dist, annot = True, cmap="YlGnBu", ax=ax, vmax=1)
  #plt.xticks(rotation=0) 
  plt.yticks(rotation=0) 
  plt.savefig(filepath + ".pdf", bbox_inches="tight", pad_inches=0)
  df_pairwise_Dist.to_csv(filepath + ".csv")

def plotPairwiseDist(list_of_PDs, clf_Names, filepath):
  temp_list_of_PDs = list_of_PDs[:]
  searchSpaceSize = len(temp_list_of_PDs[0])
  uniform_PD = [(1 / searchSpaceSize) for i in range(searchSpaceSize)]
  temp_list_of_PDs.append(uniform_PD)
  temp_clf_Names = clf_Names + ["UniformRandomSampling"]
  temp_list_of_PDs = np.array(temp_list_of_PDs)
  pairwise_Dist = np.sum((temp_list_of_PDs[:, np.newaxis, :] - temp_list_of_PDs[np.newaxis, :, :])**2, axis = -1)
  df_pairwise_Dist = pd.DataFrame(pairwise_Dist, index = temp_clf_Names, columns = temp_clf_Names).round(3)
  fig, ax = plt.subplots(figsize=(20,20))
  sns.heatmap(df_pairwise_Dist, annot = True, cmap="YlGnBu", ax=ax)
  #plt.xticks(rotation=0) 
  plt.yticks(rotation=0) 
  plt.savefig(filepath + ".pdf", bbox_inches="tight", pad_inches=0)
  df_pairwise_Dist.to_csv(filepath + ".csv")



  
def plot_pairwise_angles(list_of_pDs, clf_Names): 
  index = np.arange(len(list_of_pDs))
  pairwise_combos = np.array(list(itertools.combinations(index, 2)))

  angles = []
  for i in range(len(pairwise_combos)):
    tuple_index = pairwise_combos[i]
    dot_product = 0
    len_a = 0
    len_b = 0
    for a, b in zip(list_of_pDs[tuple_index[0]], list_of_pDs[tuple_index[1]]):
      dot_product += a * b
      len_a += a * a
      len_b += b * b 
    angles.append(math.acos(dot_product / (math.sqrt(len_a) * math.sqrt(len_b))))

  pairwise_angles = np.zeros((len(list_of_pDs), len(list_of_pDs)))
  for j in range(len(angles)):
    a = pairwise_combos[j][0]
    b = pairwise_combos[j][1]
    pairwise_angles[a][b] = angles[j]
    pairwise_angles[b][a] = angles[j]

  df_pairwise_Angles = pd.DataFrame(pairwise_angles, index = clf_Names, columns = clf_Names)
  sns.heatmap(df_pairwise_Angles, annot = True, cmap="YlGnBu")