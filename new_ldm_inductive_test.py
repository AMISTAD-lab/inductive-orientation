from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import new_ldm_inductive as ldm_inductive
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import itertools

# Initialize model specific variables
#dataset = datasets.load_iris()
dataset = pd.read_csv("EEG_Eye_State.csv", header = None)
values = dataset.values
X, y = values[:, :-1], values[:, -1]
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
adaboostClassifier = AdaBoostClassifier()
randomForest = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42) #gives us about 5 things in the holdout set

target = ldm_inductive.getTarget(X_test, y_test, 5)

num_holdout_samples = len(X_test)
all_labels = list(itertools.product([0,1], repeat=num_holdout_samples))
model3.fit(X_train, y_train)
sparse = ldm_inductive.getSimplex(model3, X_test, [0,1], all_labels, True)
LDM = ldm_inductive.getLDM(model3, X_train, X_test, y_train, classes=[0,1], num_datasets=5, num_repeat=5)
PD = ldm_inductive.computePD(LDM)

list_LDM = ldm_inductive.computeNLDM(10, model3, X_train, X_test, y_train, num_repeat=5, classes=[0,1], num_datasets=5, proportion_of_dataset=0.3)











# holdout_set_percentage = 0.003 #i think this is useless now rt
# num_datasets = 5
# proportion_of_dataset = 0.3
# classes = [0,1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.003) #gives us about 5 things in the holdout set
#X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)
# classes=[0,1]
# num_datasets=200
# proportion_of_dataset=0.3
# N = 10

# def trial(clf):
#     list_of_LDM = ldm_inductive.computeNLDM(N, clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset)
#     sparse_Pd_l = [ldm_inductive.computePD(LDM) for LDM in list_of_LDM]
#     predict_proba_Pd_l = ldm_inductive.computeNPD(N, clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=False)
#     SGT_Pd_l = [ldm_inductive.simpleGoodTuring(LDM) for LDM in list_of_LDM]

#     print("example inductive orientation vector -------------------------------------------------------------")

#     sparse_Pd = sparse_Pd_l[0]
#     print("example sparse PD:", sparse_Pd)
#     print()
#     predict_proba_Pd = predict_proba_Pd_l[0]
#     print("example predict_proba PD:", predict_proba_Pd)
#     print()
#     SGT_Pd = SGT_Pd_l[0]
#     print("example SGT PD:", SGT_Pd)
#     print()

#     print("variance -------------------------------------------------------------")

#     sparse_var = ldm_inductive.computeVariance(sparse_Pd_l)
#     print("sparse PD variance: ", sparse_var)
#     print("mean sparse PD variance: ", mean(sparse_var))
#     print()
#     predict_proba_var = ldm_inductive.computeVariance(predict_proba_Pd_l)
#     print("predict_proba PD variance: ", predict_proba_var)
#     print("mean predict_proba PD variance: ", mean(predict_proba_var))
#     print()
#     SGT_var = ldm_inductive.computeVariance(SGT_Pd_l)
#     print("SGT adjusted PD variance: ", SGT_var)
#     print("mean SGT adjusted PD variance: ", mean(SGT_var))
#     print()

#     print("angle -------------------------------------------------------------")

#     print("angle sparse, SGT: ", ldm_inductive.computeAngle(sparse_Pd, SGT_Pd))
#     print("angle sparse, predict_proba: ", ldm_inductive.computeAngle(sparse_Pd, predict_proba_Pd))
#     print("angle predict_proba, SGT: ", ldm_inductive.computeAngle(predict_proba_Pd, SGT_Pd))
#     print()
    
#     print("difference -------------------------------------------------------------")
    
#     print("difference sparse, SGT: ", (sparse_Pd - SGT_Pd))
#     print()
#     print("difference sparse, predict_proba: ", (sparse_Pd - predict_proba_Pd))
#     print()
#     print("difference predict_proba, SGT: ", (predict_proba_Pd - SGT_Pd))
#     print()

#     print("mean square error -------------------------------------------------------------")
    
#     print("mean square error sparse, SGT: ", mean_squared_error(sparse_Pd, SGT_Pd, squared=False))
#     print("mean square error sparse, predict_proba: ", mean_squared_error(sparse_Pd, predict_proba_Pd, squared=False))
#     print("mean square error predict_proba, SGT: ", mean_squared_error(predict_proba_Pd, SGT_Pd, squared=False))

# #trial(model)
# target, location = ldm_inductive.getTarget(X_test, y_test, classes=classes)
