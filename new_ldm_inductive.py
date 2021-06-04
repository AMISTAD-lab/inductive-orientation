import itertools
import random
import math
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#data generation

#random sampling
def random_uniform(X_train, y_train, num_entries, i=0):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    indices[:num_entries] #no replacement
    shuffled_X = [X_train[i] for i in indices]
    shuffled_y = [y_train[i] for i in indices]
    
    return shuffled_X, shuffled_y

#splitting the dataset - no overlap between columns
def split_dataset(X_train, y_train, num_entries, i=0):
    subset_X = X_train[i:i+num_entries]
    subset_y = y_train[i:i+num_entries]

    return subset_X, subset_y


# Helper function that takes in an input a model, a test dataset, and a list
# of classes for the given classification problem, and computes its simplex
# vector
def getSimplex(clf, X_test, classes):
    """
    clf: the classifier to use (must implement predict_proba)
    X_test: the holdout test data
    classes: a list of possible classes that are classified
    Returns a simplex vector as a numpy array, and the list of predicted labels
    """
    num_holdout_samples = len(X_test)
    num_classes = len(classes)
    y_pred_prob = clf.predict_proba(X_test)
        #print("y_pred_prob", y_pred_prob)
    # Generate a list of predicted labels to compute the differences
    # in label assignment relative to the label assignment of the training dataset
    predicted_labels_list = clf.predict(X_test)

    all_labels = list(itertools.product(classes, repeat=num_holdout_samples))
         #print("all labels: ", all_labels[:1])
    simplex_vector = []

    alpha = 0.000001 # Used for alpha smoothing
    sum_probs = 0    # Used for normailization later

    # Iterate through all_labels and calculate probabilities
    # based on y_pred_prob values
    for i in range(0, len(all_labels)):
        #current_prob = alpha
        current_prob = 1.0
        for j in range(0, len(all_labels[i])):
            for class_index in classes:
                if ((all_labels[i][j] == class_index) and (class_index < len(y_pred_prob[j]))):
                        current_prob *= (alpha if y_pred_prob[j][class_index] == 0 else y_pred_prob[j][class_index])
                        #current_prob = 0.000001 * 0.000001 if y_pred_prob[holdoutsamplei][class_index]
        sum_probs += current_prob
        simplex_vector.append(current_prob)

    #print("sum_probs", sum_probs)
    # No need to Normalize?
    #simplex_vector = np.array(simplex_vector) / sum_probs
    simplex_vector = np.array(simplex_vector)

    return simplex_vector, predicted_labels_list

'''
getLdm() takes in a classifier (clf), a set of training features (X_train),
a set of test features (X_test), a set of training labels (y_train), a list of
possible classes to be classified into (classes), and the number of information
resources to consider (num_columns), and returns a labelling distribution matrix
where every row corresponds to a different training set.
'''

def getLDM(clf, X_train, X_test, y_train, num_columns, classes=[0,1,2], proportion_of_dataset=0.3, sparse=True, data_generation=random_uniform):
    # Initialize a labelling distribution matrix to be constructed
    ldm = []
    # Iterate through all training sets (there is a total of num_columns training sets)

    for i in range(num_columns):
        # Shuffle the training labels randomly to generate different training
        # sets for each iteration
        # random.shuffle(y_train)
        clf.classes_ = classes
        # Train the model using the current training set
        if data_generation == random_uniform:
            num_entries = int(proportion_of_dataset * len(X_train))

        elif data_generation == split_dataset:
            num_entries = len(X_train)//num_columns

        subset_X, subset_y = data_generation(X_train, y_train, num_entries, i=i) 

        clf.fit(subset_X, subset_y)
        # Obtain simplex vector for current training set
        current_simplex_vector = getSimplex(clf, X_test, classes)
        ldm.append(current_simplex_vector)

    if sparse == True:
        for x in ldm:
            index = np.argmax(x)
            for i in range(len(x)):
                x[i] = 0
            x[index] = 1

    return ldm

def computeEntropy(LDM):
    entropy_list = []
    for i in range(len(LDM)):
        current_entropy = entropy(LDM[i])
        entropy_list.append(current_entropy)
    return entropy_list

def computePD(LDM):
    return np.mean(LDM, axis=0)

#returns angle in radians
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
def computeNPD(num_PD, clf, X_train, X_test, y_train, num_columns, classes=[0,1,2], proportion_of_dataset=0.3, sparse=True, data_generation=random_uniform):
    list_of_PD = []

    for i in range(num_PD):
        LDM = getLDM(clf, X_train, X_test, y_train, num_columns, classes=classes, proportion_of_dataset=proportion_of_dataset, sparse=sparse, data_generation=data_generation)
        list_of_PD.append(computePD(LDM))

    return list_of_PD

# find the variance of a sequence of PD's
def computeVariance(list_of_PD):
    variance = np.var(list_of_PD)
    return variance


# Finds the variances for set of N inductive orientation vectors as N increases from 2 to max
def varianceUpToN(list_of_PD):

    variance_per_run = []

    for i in range(1, len(list_of_PD)):
        current_variance = computeVariance(list_of_PD[:i])
        variance_per_run.append(current_variance)
        print("Variance after ", i, " runs: ", current_variance)
    return variance_per_run