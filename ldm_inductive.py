import itertools
import random
import math
import numpy as np
from scipy.stats import entropy

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

    sum_probs = 0

    # Iterate through all_labels and compute probabilities for simplex_vector
    for i in range(len(all_labels)):
        # Initialize current_prob with a small value (since we're going to take
        # products)
        current_prob = 0
        # Iterate through the current combination of labels
        for j in range(len(all_labels[i])):
            for class_index in classes:
                if ((all_labels[i][j] == class_index) and (class_index < len(y_pred_prob[j]))):
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
    print("SUM" , sum)

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
#data generation
def random_uniform(X_train, y_train, num_entries):
    indices = np.arange(len(X_train))
    # print(type(indices))
    np.random.shuffle(indices)
    indices[:num_entries] #no replacement
    shuffled_X = [X_train[i] for i in indices]
    shuffled_y = [y_train[i] for i in indices]
    
    return shuffled_X, shuffled_y
