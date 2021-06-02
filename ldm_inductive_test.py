# AMISTAD Lab - Overfitting/Underfitting Team
# Code for Labelling Distribution Matrix Tests

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import ldm_inductive
import numpy as np

# Initialize model specific variables
dataset = datasets.load_iris()
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
holdout_set_percentage = 0.03
num_datasets = 5

def computeLdm(model, dataset, holdout_set_percentage, num_datasets):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

    matrix = ldm_inductive.getLdm(model, X_train, X_test, y_train, [0, 1, 2], num_datasets)

    return matrix

#print("LDM: ", computeLdm(model, dataset, holdout_set_percentage, num_datasets))

def computeEntropy(model, dataset, holdout_set_percentage, num_datasets):

    matrix = computeLdm(model, dataset, holdout_set_percentage, num_datasets)

    entropy_list = []

    for i in range(len(matrix)):
        current_entropy = entropy(matrix[i])
        entropy_list.append(current_entropy)

    return entropy_list

LDM = computeLdm(model, dataset, holdout_set_percentage, num_datasets)
print("LDM: ", LDM)
print("numer of columns :", len(LDM[0]))


def computePD(model, dataset, holdout_set_percentage, num_datasets):
    LDM = computeLdm(model, dataset, holdout_set_percentage, num_datasets)
    return np.mean(LDM, axis = 0)

print()
PD = computePD(model, dataset, holdout_set_percentage, num_datasets)
print ("PD: ", PD)
print("length PD: ", len(PD))

# returns angle in radians
def computeAngle(PD1, PD2):
    unit_vector_1 = PD1/np.linalg.norm(PD1)
    unit_vector_2 = PD2/np.linalg.norm(PD2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    
    if dot_product >= 1:
        return 0
    
    angle = np.arccos(dot_product)
    angle = round(angle, 7)
    return angle

PD3 = computePD(model3, dataset, holdout_set_percentage, num_datasets)
PD10 = computePD(model10, dataset, holdout_set_percentage, num_datasets)
print("Angle of alignment 1,3: ", computeAngle(PD, PD3))
print("Angle of alignment 1,10: ", computeAngle(PD, PD10))
print("Angle of alignment 3, 10: ", computeAngle(PD3, PD10))

# print("Entropy List: ", computeEntropy(model, dataset, holdout_set_percentage, num_datasets))

# find the variance of a sequence of PD's
def computeVariance(num_PD):
    list_of_PD = []
    for i in range(num_PD):
        list_of_PD.append(computePD(model, dataset, holdout_set_percentage, num_datasets))
    # list_of_PD.append(PD3)
    # list_of_PD.append(PD10)
    variance = np.var(list_of_PD)
    return variance

print("Variance of KNN1 after 5 runs: ", computeVariance(5))
# Variance of KNN1 after 3 runs:  6.774035123372077e-06
# Variance of KNN1 after 5 runs:  6.7740351233720765e-06
# Variance of KNN1 after 3 runs, KNN3, KNN10:  7.27654940851929e-06