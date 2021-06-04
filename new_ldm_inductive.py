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

'''
getLdm() takes in a classifier (clf), a set of training features (X_train),
a set of test features (X_test), a set of training labels (y_train), a list of
possible classes to be classified into (classes), and the number of information
resources to consider (num_columns), and returns a labelling distribution matrix
where every row corresponds to a different training set.
'''

def getLDM(clf, X_train, X_test, y_train, classes, num_columns, proportion_of_dataset=0.3, sparse=True, data_generation=random_uniform):
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
