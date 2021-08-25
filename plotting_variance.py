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

dataset = pd.read_csv("datasets/EEG_Eye_State.csv", header = None) #can be changed depending on the dataset
values = dataset.values
X, y = values[:, :-1], values[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)
classes=[0,1]
proportion_of_dataset=0.04 #needs to be changed

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
KNN15 = KNeighborsClassifier(n_neighbors=15)
KNN19 = KNeighborsClassifier(n_neighbors=19)
KNN25 = KNeighborsClassifier(n_neighbors=25)
KNN49 = KNeighborsClassifier(n_neighbors=49)
KNN51 = KNeighborsClassifier(n_neighbors=51)

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
logisticRegression = LogisticRegression(max_iter=2000)
SGDClassifier_hinge = SGDClassifier(max_iter=2000)
SGDClassifier_log = SGDClassifier(loss='log', max_iter=2000)
SVC_linear_kernel = SVC(kernel="linear", max_iter=2000)
SVC_linear = LinearSVC(max_iter=2000)
SVC_rbf = SVC(max_iter=2000)
MLPclf_1 = MLPClassifier(max_iter=2000)
MLPclf_3 = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=2000)


def calculate_variance(N, clf, X_train, X_test, y_train, num_datasets_list, num_repeat=1, classes=[0,1], proportion_of_dataset=0.04, data_generation=ldm_inductive.random_uniform):
  list_of_LDM = [ldm_inductive.getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets=num, num_repeat=num_repeat, 
                                      proportion_of_dataset=proportion_of_dataset, data_generation=data_generation) for num in num_datasets_list]
  sparse_Pd_l = [ldm_inductive.computePD(LDM) for LDM in list_of_LDM]
  list_of_variance = []
  index = 0
  for i in range(len(num_datasets_list)//N):
    list_of_variance.append(ldm_inductive.computeVariance(sparse_Pd_l[index:index+N]))
    index += N
  list_of_mean_variance = [np.mean(variance) for variance in list_of_variance]
  print("sparse_Pd_l: ", sparse_Pd_l)
  print("list_of_variance: ", list_of_variance)
  print("list_of_mean_variance: ", list_of_mean_variance)
  return sparse_Pd_l, list_of_variance, list_of_mean_variance

# def plot_variance(num_datasets, mean_variances, axes):
#   num_datasets = np.array(num_datasets)
#   _, idx = np.unique(num_datasets, return_index=True)
#   num_datasets = num_datasets[np.sort(idx)]
#   axes.plot(num_datasets, mean_variances)
#   axes.set_xlabel('number of randomly selected training subsets')
#   axes.set_ylabel('variance')
#   fig.show()

def plot_mult_variance(num_datasets, mean_variances_l, clf_names, path, title):
  palette = ["#F23030", "#F2762E", "#F2D338", "#25C7D9", "#248EA6", "#324DBE"]
  plt.style.use("ggplot")
  fig = plt.figure()
  axes = fig.add_axes([0.15, 0.13, 0.8, 0.8])
  axes.set_xlabel('number of randomly selected training subsets')
  axes.set_ylabel('variance')
  for i, mean_variances in enumerate(mean_variances_l):
    axes.plot(num_datasets, mean_variances, label=clf_names[i], marker="o", c=palette[i])
  axes.legend()
  axes.ticklabel_format(axis="y", style="scientific", scilimits = (0,0))
  plt.title(title)
  plt.savefig(path)

if __name__ == "__main__":
    num_datasets_list = [100]*3+[200]*3+[300]*3+[400]*3+[500]*3+[600]*3+[700]*3+[800]*3+[900]*3+[1000]*3+[1500]*3+[2000]*3
    KNN1_sparse_Pd_l, KNN1_variance, KNN1_mean_variance = calculate_variance(3, KNN1, X_train, X_test, y_train, num_datasets_list, num_repeat=20)
    KNN3_sparse_Pd_l, KNN3_variance, KNN3_mean_variance = calculate_variance(3, KNN3, X_train, X_test, y_train, num_datasets_list, num_repeat=20)
    KNN11_sparse_Pd_l, KNN11_variance, KNN11_mean_variance = calculate_variance(3, KNN11, X_train, X_test, y_train, num_datasets_list, num_repeat=20)
    KNN19_sparse_Pd_l, KNN19_variance, KNN19_mean_variance = calculate_variance(3, KNN19, X_train, X_test, y_train, num_datasets_list, num_repeat=20)
    KNN51_sparse_Pd_l, KNN51_variance, KNN51_mean_variance = calculate_variance(3, KNN51, X_train, X_test, y_train, num_datasets_list, num_repeat=20)


    p1_num_datasets_list = [100]+[200]+[300]+[400]+[500]+[600]+[700]+[800]+[900]+[1000]+[1500]+[2000]
    p1_clf_names = ["KNN1", "KNN3", "KNN11", "KNN19", "KNN51"]
    p1_mean_variances_l = [KNN1_mean_variance, KNN3_mean_variance, KNN11_mean_variance, KNN19_mean_variance, KNN51_mean_variance]
    plot_mult_variance(p1_num_datasets_list, p1_mean_variances_l, p1_clf_names, "./variances/plot1.pdf", "KNN") #make sure you are saving the file in the right place