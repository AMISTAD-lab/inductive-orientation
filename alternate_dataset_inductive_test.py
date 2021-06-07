from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import new_ldm_inductive as ldm_inductive
import matplotlib.pyplot as plt
import numpy as np

# Initialize model specific variables
dataset = datasets.load_iris()
dataset2 = datasets.fetch_covtype()
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
adaboostClassifier = AdaBoostClassifier()
holdout_set_percentage = 0.03
num_datasets = 5
proportion_of_dataset = 0.1

num_classes = 7
classes = [i for i in range(num_classes)]

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

LDM = ldm_inductive.getLDM(adaboostClassifier, X_train, X_test, y_train, sparse=False, proportion_of_dataset=proportion_of_dataset)
ldm_inductive.computeGoodTuring(LDM)
# # print(LDM)
# ldm_inductive.varianceUpToN(LDM)
# Pd = ldm_inductive.computePD(LDM)

# list_of_PD_sparse = ldm_inductive.computeNPD(5, model, X_train, X_test, y_train)
# list_of_PD = ldm_inductive.computeNPD(5, model, X_train, X_test, y_train, sparse=False)


# print("sparse")
# ldm_inductive.varianceUpToN(list_of_PD_sparse)
# print("not sparse")
# ldm_inductive.varianceUpToN(list_of_PD)

# data = dataset2.data[:50]
# target = dataset2.target[:50]
# X_train, X_test, y_train, y_test = train_test_split(dataset2.data, dataset2.target, test_size = holdout_set_percentage)

# n=500

# LDM = ldm_inductive.getLDM(model, X_train[:n], X_test[:int(holdout_set_percentage * n)], y_train[:n], classes=classes)
# Pd = ldm_inductive.computePD(LDM)

# list_of_PD_sparse = ldm_inductive.computeNPD(5, model, X_train[:n], X_test[:int(holdout_set_percentage * n)], y_train[:n], classes=classes)
# list_of_PD = ldm_inductive.computeNPD(5, model, X_train[:n], X_test[:int(holdout_set_percentage * n)], y_train[:n], sparse=False, classes=classes)

# print("sparse")
# ldm_inductive.varianceUpToN(list_of_PD_sparse)
# print("not sparse")
# ldm_inductive.varianceUpToN(list_of_PD)

# X_train, X_test, y_train, y_test = train_test_split(dataset2.data, dataset2.target, test_size = holdout_set_percentage)

# n=50000


# print("sparse LDM")
# LDM_sparse = ldm_inductive.getLDM(adaboostClassifier, X_train[:n], X_test[:5], y_train[:n], classes=classes)
# ldm_inductive.varianceUpToN(LDM_sparse)

# print("not sparse LDM")
# LDM = ldm_inductive.getLDM(adaboostClassifier, X_train[:n], X_test[:5], y_train[:n], classes=classes, sparse=False)
# ldm_inductive.varianceUpToN(LDM)

# Pd = ldm_inductive.computePD(LDM)

# list_of_PD_sparse = ldm_inductive.computeNPD(5, adaboostClassifier, X_train[:n], X_test[:5], y_train[:n], classes=classes, proportion_of_dataset=proportion_of_dataset)
# list_of_PD = ldm_inductive.computeNPD(5, adaboostClassifier, X_train[:n], X_test[:5], y_train[:n], sparse=False, classes=classes, proportion_of_dataset=proportion_of_dataset)

# print("sparse")
# ldm_inductive.varianceUpToN(list_of_PD_sparse)
# print("not sparse")
# ldm_inductive.varianceUpToN(list_of_PD)
