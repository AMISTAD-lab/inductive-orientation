import itertools
import random
import math
import numpy as np
from scipy.stats import entropy
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# AMISTAD Lab - Overfitting/Underfitting Team
# Code for Labelling Distribution Matrix Tests


'''
getSimplex()) takes in a classifier (clf), a set of test features (X_test), and a list of possible classes
to be classified into (classes), and returns a normalized probability distribution (a simplex vector)
as a numpy array
'''
def getSimplex(clf, X_test, classes):

    num_holdout_samples = len(X_test)
    num_classes = len(classes)

    # Initialize a list of all possible labellings
    all_labels = list(itertools.product(classes, repeat = num_holdout_samples))
    simplex_vector = []

    y_pred_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    # sparse_y_pred is matrix of sparse probabilities in the same form as y_pred_prob
    sparse_y_pred = [[0 for i in range(num_classes)] for j in range(num_holdout_samples)]

    for i in range(len(y_pred)):
        sparse_y_pred[i][y_pred[i]] = 1


    sum_probs = 0

    # Iterate through all_labels and compute probabilities for simplex_vector
    for i in range(len(all_labels)):
        # Initialize current_prob with a small value (since we're going to take
        # products)
        current_prob = 0
        # Iterate through the current combination of labels
        for j in range(len(all_labels[i])):
            for class_index in classes:
                if ((all_labels[i][j] == class_index)): # and (class_index < len(y_pred_prob[j]))):
                    # If the current probability is 0, then just add 0 to current prob
                    if y_pred_prob[j][class_index] == 0:
                        current_prob += 0
                    elif y_pred_prob[j][class_index] == 1:
                        current_prob += -math.log10(y_pred_prob[j][class_index] - 0.00001)
                    else:
                        current_prob += -math.log10(y_pred_prob[j][class_index])

        # Compute sum used for normalization at the end
        sum_probs += current_prob
        simplex_vector.append(current_prob)

    # Normalization step
    # If sum_probs == 0, divide by 1 instead
    if sum_probs == 0:
        simplex_vector = np.array(simplex_vector)
    else:
        simplex_vector = np.array(simplex_vector) / sum_probs
    # simplex_vector = np.array(simplex_vector)

    sum = 0
    for i in range(simplex_vector.size):
        sum += simplex_vector[i]
    #print("SUM" , sum)

    return simplex_vector


def getSparseSimplex(clf, X_test, classes):

    num_holdout_samples = len(X_test)
    num_classes = len(classes)

    # Initialize a list of all possible labellings
    all_labels = list(itertools.product(classes, repeat = num_holdout_samples))
    simplex_vector = []

    y_pred_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    # sparse_y_pred is matrix of sparse probabilities in the same form as y_pred_prob
    sparse_y_pred = [[0 for i in range(num_classes)] for j in range(num_holdout_samples)]

    for i in range(len(y_pred)):
        sparse_y_pred[i][y_pred[i]] = 1


    sum_probs = 0

    # Iterate through all_labels and compute probabilities for simplex_vector
    for i in range(len(all_labels)):
        # Initialize current_prob with a small value (since we're going to take
        # products)
        current_prob = 0
        # Iterate through the current combination of labels
        for j in range(len(all_labels[i])):
            for class_index in classes:
                if ((all_labels[i][j] == class_index)): # and (class_index < len(y_pred_prob[j]))):
                    # If the current probability is 0, then just add 0 to current prob
                    if sparse_y_pred[j][class_index] == 0:
                        current_prob += 0
                    elif sparse_y_pred[j][class_index] == 1:
                        current_prob += -math.log10(sparse_y_pred[j][class_index] - 0.00001)
                    else:
                        current_prob += -math.log10(sparse_y_pred[j][class_index])

        # Compute sum used for normalization at the end
        sum_probs += current_prob
        simplex_vector.append(current_prob)

    # Normalization step
    # If sum_probs == 0, divide by 1 instead
    if sum_probs == 0:
        simplex_vector = np.array(simplex_vector)
    else:
        simplex_vector = np.array(simplex_vector) / sum_probs
    # simplex_vector = np.array(simplex_vector)

    sum = 0
    for i in range(simplex_vector.size):
        sum += simplex_vector[i]
    #print("SUM" , sum)

    return simplex_vector

'''
getLdm() takes in a classifier (clf), a set of training features (X_train),
a set of test features (X_test), a set of training labels (y_train), a list of
possible classes to be classified into (classes), and the number of information
resources to consider (num_columns), and returns a labelling distribution matrix
where every row corresponds to a different training set.
'''
def getLdm(clf, X_train, X_test, y_train, classes, num_columns, ):

    # Initialize a labelling distribution matrix to be constructed
    ldm = []

    # Iterate through all training sets (there is a total of num_columns training
    # sets)
    for _ in range(num_columns):
        # Shuffle the training labels randomly to generate different training
        # sets for each iteration
        # random.shuffle(y_train)
        clf.classes_ = classes
        # Train the model using the current training set
        num_entries = int(.5 * len(X_train))
        random_X, random_y = random_uniform(X_train, y_train, num_entries)

        clf.fit(random_X, random_y)
        # Obtain simplex vector for current training set
        current_simplex_vector = getSimplex(clf, X_test, classes)
        ldm.append(current_simplex_vector)

    return ldm


def getSparseLdm(clf, X_train, X_test, y_train, classes, num_columns, ):

    # Initialize a labelling distribution matrix to be constructed
    ldm = []

    # Iterate through all training sets (there is a total of num_columns training
    # sets)
    for _ in range(num_columns):
        # Shuffle the training labels randomly to generate different training
        # sets for each iteration
        # random.shuffle(y_train)
        clf.classes_ = classes
        # Train the model using the current training set
        num_entries = int(.5 * len(X_train))
        random_X, random_y = random_uniform(X_train, y_train, num_entries)

        clf.fit(random_X, random_y)
        # Obtain simplex vector for current training set
        current_simplex_vector = getSparseSimplex(clf, X_test, classes)
        ldm.append(current_simplex_vector)

    return ldm

#data generation
def random_uniform(X_train, y_train, num_entries):
    indices = np.arange(len(X_train))
    # print(type(indices))
    np.random.shuffle(indices)
    indices[:num_entries] #no replacement
    shuffled_X = [X_train[i] for i in indices]
    shuffled_y = [y_train[i] for i in indices]
    
    return shuffled_X, shuffled_y

def computeLdm(model, dataset, holdout_set_percentage, num_datasets):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

    matrix = getLdm(model, X_train, X_test, y_train, [0, 1, 2], num_datasets)

    return matrix

def computeSparseLdm(model, dataset, holdout_set_percentage, num_datasets):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

    matrix = getSparseLdm(model, X_train, X_test, y_train, [0, 1, 2], num_datasets)

    return matrix


def computeEntropy(model, dataset, holdout_set_percentage, num_datasets):

    matrix = computeLdm(model, dataset, holdout_set_percentage, num_datasets)

    entropy_list = []

    for i in range(len(matrix)):
        current_entropy = entropy(matrix[i])
        entropy_list.append(current_entropy)

    return entropy_list



def computePD(model, dataset, holdout_set_percentage, num_datasets):
    LDM = computeLdm(model, dataset, holdout_set_percentage, num_datasets)
    return np.mean(LDM, axis = 0)


def computeSparsePD(model, dataset, holdout_set_percentage, num_datasets):
    LDM = computeSparseLdm(model, dataset, holdout_set_percentage, num_datasets)
    return np.mean(LDM, axis = 0)


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

# create a list of N PD's
def computeNPD(num_PD, model, dataset, holdout_set_percentage, num_datasets):
    list_of_PD = []
    for i in range(num_PD):
        list_of_PD.append(computePD(model, dataset, holdout_set_percentage, num_datasets))
    return list_of_PD

# create a list of N PD's
def computeSparseNPD(num_PD, model, dataset, holdout_set_percentage, num_datasets):
    list_of_PD = []
    for i in range(num_PD):
        list_of_PD.append(computeSparsePD(model, dataset, holdout_set_percentage, num_datasets))
    return list_of_PD

# find the variance of a sequence of PD's
def computeVariance(list_of_PD):
    # list_of_PD.append(PD3)
    # list_of_PD.append(PD10)
    variance = np.var(list_of_PD)
    return variance

# Finds the variances for set of N inductive orientation vectors as N increases from 2 to max
def varianceUpToN(max,modelName, model, dataset, holdout_set_percentage, num_datasets):
    run_number = list(range(2,max))
    variance_per_run = []
    list_of_PD = computeNPD(1, model, dataset, holdout_set_percentage, num_datasets)
    for i in range(2,max):
        list_of_PD.append(computePD(model, dataset, holdout_set_percentage, num_datasets))
        current_variance = computeVariance(list_of_PD)
        variance_per_run.append(current_variance)
        #print("Variance of " + modelName + " after ", i, " runs: ", current_variance)
    return run_number, variance_per_run


# Finds the variances for set of N inductive orientation vectors as N increases from 2 to max
def varianceUpToN(max,modelName, model, dataset, holdout_set_percentage, num_datasets):
    run_number = list(range(2,max))
    variance_per_run = []
    list_of_PD = computeSparseNPD(1, model, dataset, holdout_set_percentage, num_datasets)
    for i in range(2,max):
        list_of_PD.append(computeSparsePD(model, dataset, holdout_set_percentage, num_datasets))
        current_variance = computeVariance(list_of_PD)
        variance_per_run.append(current_variance)
        #print("Variance of " + modelName + " after ", i, " runs: ", current_variance)
    return run_number, variance_per_run


def plotHeatMap(ldm):
    # Transpose LDM generated so that simplex vectors are column vectors
    ldmTransposed = [list(i) for i in zip(*ldm)]
    plt.imshow(ldmTransposed, cmap='hot', interpolation='nearest')
    plt.show()