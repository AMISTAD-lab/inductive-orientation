from sklearn import datasets
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.cluster import AgglomerativeClustering

import new_ldm_inductive as ldm_inductive

dataset = pd.read_csv("EEG_Eye_State.csv", header = None)
values = dataset.values
X, y = values[:, :-1], values[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)
classes=[0,1]
num_datasets=500
proportion_of_dataset=0.04


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

KNN1 = KNeighborsClassifier(n_neighbors=1)
KNN7 = KNeighborsClassifier(n_neighbors=7)
KNN3 = KNeighborsClassifier(n_neighbors=3)
KNN9 = KNeighborsClassifier(n_neighbors=9)
KNN5 = KNeighborsClassifier(n_neighbors=5)
KNN11 = KNeighborsClassifier(n_neighbors=11)
KNN17 = KNeighborsClassifier(n_neighbors=17)
KNN15 = KNeighborsClassifier(n_neighbors=15)
KNN19 = KNeighborsClassifier(n_neighbors=19)
KNN21 = KNeighborsClassifier(n_neighbors=21)
KNN23 = KNeighborsClassifier(n_neighbors=23)
KNN25 = KNeighborsClassifier(n_neighbors=25)
adaboostClassifier = AdaBoostClassifier()
gradientBoostingClassifier = GradientBoostingClassifier()
randomForest = RandomForestClassifier()
decisionTreeClassifier = DecisionTreeClassifier()
naiveBayesClassifier = GaussianNB()
quadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()
logisticRegression = LogisticRegression(random_state=0, max_iter=500)
linearRegression = LinearRegression()
linearDiscriminantAnalysis = LinearDiscriminantAnalysis()


def new_trial(N, clf, X_train, X_test, y_train, classes=[0,1], num_datasets=500, proportion_of_dataset=0.04):
  list_of_LDM = ldm_inductive.computeNLDM(N, clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset)
  sparse_Pd_l = [ldm_inductive.computePD(LDM) for LDM in list_of_LDM]
  predict_proba_Pd_l = ldm_inductive.computeNPD(N, clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=False)
  list_of_LDM_SGT = ldm_inductive.computeNLDM(N, clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset)
  SGT_Pd_l = [ldm_inductive.simpleGoodTuring(LDM) for LDM in list_of_LDM_SGT]
  fig = plt.figure(figsize = (20,20)) # width x height
  ax1 = fig.add_subplot(3, 3, 1) # row, column, position
  ax2 = fig.add_subplot(3, 3, 2)
  ax3 = fig.add_subplot(3, 3, 3)
  ax1.set_title("sparse")
  ax2.set_title("predict_Pd")
  ax3.set_title("SGT")
  sparse_Pd_lT = [list(i) for i in zip(*sparse_Pd_l)]
  predict_proba_Pd_lT = [list(i) for i in zip(*predict_proba_Pd_l)]
  SGT_Pd_lT = [list(i) for i in zip(*SGT_Pd_l)]
  sns.heatmap(data=sparse_Pd_lT, ax=ax1,cmap='coolwarm')
  sns.heatmap(data=predict_proba_Pd_lT, ax=ax2, cmap='coolwarm')
  sns.heatmap(data=SGT_Pd_lT, ax=ax3, cmap='coolwarm')
  return sparse_Pd_l, predict_proba_Pd_l, SGT_Pd_l
  
import plotly.graph_objects as go
def plot_list(list_of_vectors, list_of_names):
  fig = go.Figure()
  i=0
  for vector in list_of_vectors:
    fig.add_trace(go.Scatter(x=list(range(len(vector))), y=vector, mode='lines', name=list_of_names[i]))
    i +=1
  fig.show()

def create_index(repeat, list_of_names):
  list_of_indexed_names = []
  for name in list_of_names:
    for i in range(1, repeat+1):
      list_of_indexed_names.append(name + str(i))
  return list_of_indexed_names

def side_by_side(list_of_vectors, list_of_names):
  for i in range(len(list_of_vectors[0])):
    name=0
    for vector in list_of_vectors:
      print(list_of_names[name]+ " ", i, " ", vector[i])
      name += 1
    print()

if __name__ == "__main__":
    #generate a list of 10 sparse, predict_proba, and SGT PD vectors + heatmap
    sparse_Pd_l, predict_proba_Pd_l, SGT_Pd_l = new_trial(10, KNN11, X_train, X_test, y_train)
    print("sparse ", sparse_Pd_l)
    print("predict_proba ", predict_proba_Pd_l)
    print("SGT ", SGT_Pd_l)

    #uses plotly to plot the variance
    list_of_vectors = [ldm_inductive.computeVariance(sparse_Pd_l), ldm_inductive.computeVariance(predict_proba_Pd_l), ldm_inductive.computeVariance(SGT_Pd_l)]
    list_of_names = ["variance of sparse", "variance of predict_proba", "variance of SGT"]
    plot_list(list_of_vectors, list_of_names)

    #uses plotly to plot the PD's
    list_of_vectors = sparse_Pd_l + predict_proba_Pd_l + SGT_Pd_l
    list_of_names = create_index(10, ["sparse", "predict_proba", "SGT"])
    plot_list(list_of_vectors, list_of_names)