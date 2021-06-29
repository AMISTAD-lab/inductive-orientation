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

import setup2
#import clean_data
import csv

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


def get_LDM_PD(list_of_clf, X_train, X_test, y_train, classes=[0,1], num_datasets=5, num_repeat=1, proportion_of_dataset=0.1, sparse=True, data_generation=random_uniform):
    LDM_l = []
    PD_l = []
    for clf in list_of_clf:
        LDM = getLDM(clf, X_train, X_test, y_train, classes = classes, num_datasets=num_datasets, num_repeat=num_repeat, proportion_of_dataset=proportion_of_dataset, sparse=sparse, data_generation=data_generation)
        LDM_l.append(LDM)
        PD_l.append(computePD(LDM, sparse=sparse))
    return LDM_l, PD_l



def main():
    dataset = pd.read_csv("EEG_Eye_State.csv")
    values = dataset.values
    X, y = values[:, :-1], values[:, -1]

    
    num_repeats = 3
    list_of_clf=[decisionTreeClassifier]*num_repeats 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)
    LDM_l, PD_l = get_LDM_PD(list_of_clf, X_train, X_test, y_train, num_datasets=505, num_repeat=1, proportion_of_dataset=0.04)
    

    with open('ldm.txt', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(LDM_l)

    with open('pd.txt', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(PD_l)



if __name__ == "__main__":
    main()