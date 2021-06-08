import itertools
import random
import math
import numpy as np
from scipy.sparse import data
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import simple_good_turing

#data generation
'''
random_uniform takes in training data and produces one random subset of training data.
num_entries determines how many training datasets are in the subset of training data.
The random subset is selected by randomly drawing from the training data without replacement

inputs:
    X_train: a list of training features
    y_train: a list of labels corresponding to the training features
    num_entries: the number of data points to draw from X_train and y_train

returns:
    random_subset_X: a list of length num_entries of training features
    random_subset_y: a list of length num_entries with labels corresponding to training features
    in random_subset_y
'''
# def random_uniform(X_train, y_train, num_entries, i=0):
#     indices = np.arange(len(X_train))
#     np.random.shuffle(indices)
#     indices = indices[:num_entries] #no replacement
#     random_subset_X = [X_train[i] for i in indices]
#     random_subset_y = [y_train[i] for i in indices]
#     return random_subset_X, random_subset_y
def random_uniform(X_train, y_train, num_entries, i=0):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    indices = indices[:num_entries] #no replacement
    random_subset_X = X_train.iloc[indices]
    random_subset_y = y_train.iloc[indices]
    return random_subset_X, random_subset_y

'''splitting the dataset - no overlap between columns'''
def split_dataset(X_train, y_train, num_entries, i=0):
    start_index = i*num_entries
    subset_X = X_train[start_index:start_index+num_entries]
    subset_y = y_train[start_index:start_index+num_entries]
    return subset_X, subset_y

#Getting the LDM, and PD
'''
getSimplex()) takes in a classifier (clf), a set of test features (X_test), and a list of possible classes
to be classified into (classes), and returns a normalized probability distribution (a simplex vector)
as a numpy array

inputs:
    clf: an untrained classifier
    X_test: a list of test features
    classes: a list of possible classes, for example [0,1,2]

outputs: 
    simplex vector: a numpy array. each entry in the array includes the probability the classification 
    algorithm will give the corresponding sequence of labels as its prediction
'''
def getSimplex(clf, X_test, classes, all_labels):
    y_pred_prob = clf.predict_proba(X_test)
    # Generate a list of predicted labels to compute the differences
    # in label assignment relative to the label assignment of the training dataset
    # predicted_labels_list = clf.predict(X_test)
    #print("num_holdout_samples: ", num_holdout_samples)

    #computationally impossible for large numbers
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

    # No need to Normalize?
    #simplex_vector = np.array(simplex_vector) / sum_probs
    simplex_vector = np.array(simplex_vector)


    return simplex_vector

'''
getLDM() takes in a classifier (clf), a set of training features (X_train),
a set of test features (X_test), a set of training labels (y_train), a list of
possible classes to be classified into (classes), and the number of information
resources to consider (num_columns), and returns a labelling distribution matrix
where every row corresponds to a different training set.

inputs:
    clf: an untrained classifier
    X_train: a list of training feature
    X_test: a list of test features, also known as the holdout set
    y_train: a list of labels corresponding to training features
    classes: a list of possible classes, for example [0,1,2]
    num_datasets: an int that determines probability distribution (simplex) vectors to create
    proportion_of_dataset: a double determines what proportion of X_train and y_train should be used 
    to train the classifier to produce one probability distribution (simplex) vector
    sparse: a boolean that determines whether to smooth the probability dsitribution vector
            True: no smoothing
            False: currently using the LDM's predict_proba and logs

outpus:
    LDM: a 2D np array matrix, where LDM[i] gives a probability distribution vector trained on one particular subset of training data and tested on a fixed holdout set.
'''
def getLDM(clf, X_train, X_test, y_train, classes=[0,1,2], num_datasets=5, proportion_of_dataset=0.1, sparse=True, data_generation=random_uniform):
    # Initialize a labelling distribution matrix to be constructed
    LDM = []
    num_holdout_samples = len(X_test)
    all_labels = list(itertools.product(classes, repeat=num_holdout_samples))
    #print("this is all_labels ",all_labels)
    # Iterate through all training sets (there is a total of num_columns training
    # sets)
    for i in range(num_datasets):
        # Shuffle the training labels randomly to generate different training
        # sets for each iteration
        # random.shuffle(y_train)
        clf.classes_ = classes
        # Train the model using the current training set

        if data_generation == random_uniform:
            num_entries = int(proportion_of_dataset * len(X_train))

        elif data_generation == split_dataset:
            num_entries = len(X_train)//num_datasets

        subset_X, subset_y = data_generation(X_train, y_train, num_entries, i=i) 

        clf.fit(subset_X, subset_y)
        # Obtain simplex vector for current training set
        current_simplex_vector = getSimplex(clf, X_test, classes, all_labels)
        LDM.append(current_simplex_vector)

    if sparse:
        for x in LDM:
            index = np.argmax(x)
            for i in range(len(x)):
                x[i] = 0
            x[index] = 1

    return LDM

def computeNLDM(num_LDM, clf, X_train, X_test, y_train, classes=[0,1,2], num_datasets=5, proportion_of_dataset=0.3, sparse=True, data_generation=random_uniform):
    list_of_LDM=[]
    for i in range(num_LDM):
        LDM = getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=sparse, data_generation=data_generation)
        list_of_LDM.append(LDM)
    return list_of_LDM

'''
computePD finds the inductive orientation vector(PD) of a LDM by taking the average of all the probability distribution simplex vectors
input:
    LDM: an LDM matrix
return:
    PD: a np array that is an inductive orientation vector    
'''
def computePD(LDM):
    PD = np.mean(LDM, axis=0)
    return PD

'''
computeNPD creates a list of N number of inductive orientation vectors
inputs:
    num_PD: an int that represents the total number of inductive orientation vectors desired
output:
    list_of_PD: a list of num_PD number of inductive orientation vectors (each of which is a np array)
'''
def computeNPD(num_PD, clf, X_train, X_test, y_train, classes=[0,1,2], num_datasets=5, proportion_of_dataset=0.3, sparse=True, data_generation=random_uniform):
    list_of_PD = []

    for i in range(num_PD):
        LDM = getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=sparse, data_generation=data_generation)
        list_of_PD.append(computePD(LDM))

    return list_of_PD


# analysis
'''
computeEntropy finds the entropies for each sublist (pf, probability distribution simplex vector) of the LDM
input:
    LDM: an LDM matrix
return:
    entropy_list: a list of doubles that are the entropy values for the corresponding pf
'''
def computeEntropy(LDM):
    entropy_list = []
    for i in range(len(LDM)):
        current_entropy = entropy(LDM[i])
        entropy_list.append(current_entropy)
    return entropy_list

'''
computeVariance finds the variance of a list of probability distribution simplex vectors
input:
    list_of_simplex: a list of probability distribution simplex vectors
return:
    variance: a double that represents on average how much the each of the probability distribution simplex vectors 
    deviate from the average simplex vector
'''
def computeVariance(list_of_simplex):
    variance = np.var(list_of_simplex, axis=0)
    return variance

'''
varianceUptoN finds variance of a list of inductive orientation vectors PD's as the size of the list increases from 2 to max
inputs:
    max: an int that represents the maximum set of inductive orientation vectors PD's
    clfName: a string that represents the name of the classifer
    clf: an untrained classifier
    ... refer to getLDM
returns:
    run_number: a list that records the size of the inductive orientation vectors PD's
    variance_per_run: a list that gives the variance corresponding to the run_number
'''
def varianceUpToN(list_of_PD):

    variance_per_run = []

    for i in range(1, len(list_of_PD)):
        current_variance = computeVariance(list_of_PD[:i])
        variance_per_run.append(current_variance)
        print("Variance after ", i, " runs: ", current_variance)
    return variance_per_run


'''
variance_propdata finds the variance between columns of an LDM, not PD, as the proportion of dataset changes
inputs:
    percentages: a list of percentages in double 
returns:
    run_number: a list that records the size of the inductive orientation vectors PD's
    variance_per_run: a list that gives the variance corresponding to the run_number
'''
def variance_propdata(percentages, clf, X_train, X_test, y_train,classes=[0,1,2], num_datasets=5, sparse=True, data_generation=random_uniform):
    percentages_list = []
    variance_list = []
    for i in percentages:
        proportion_of_dataset = i
        percentages_list.append(i)
        current_ldm = getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=sparse, data_generation=data_generation)
        current_variance = computeVariance(current_ldm)
        variance_list.append(current_variance)
        print("proportion of dataset: ", i, " the variance of the Pf's in the LDM with: ", num_datasets, "number of datasets is: ", current_variance)
    return percentages_list, variance_list


'''
computeAngle finds the radian angle between two inductive orientation vector using dot product
input:
    PD1: a numpy array that represents an inductive orientation vector
    PD2: a numpy array that represents an inductive orientation vector
return:
    angle: a double that presents the angle between the two vectors in radians
'''
def computeAngle(PD1, PD2):
    unit_vector_1 = PD1/np.linalg.norm(PD1)
    unit_vector_2 = PD2/np.linalg.norm(PD2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    
    if dot_product >= 1:
        return 0
    
    angle = np.arccos(dot_product)
    angle = round(angle, 7)
    return angle

#compute Good Turing
'''
computeGoodTuring is currently the first part of smoothing by good turing. 
It tells us the frequency of each sequence. Each sequence is represented by their index in the inductive orientation simplex vector.
input: a sparse LDM
output: the Pd with its values adjusted according to the Good-Turing formula c_new = (c+1) N_c/N_c+1
'''
def computeGoodTuring(LDM):
    #convert ldm to numpy array
    LDM = np.array(LDM)
    num_cols_LDM = len(LDM)
 
    # adds up all the "columns" in the LDM
    sumColsLDM = np.sum(LDM, axis = 0)
 
    # values is an np array of unique values, 
    # counts is the number of times each of these values occurs in sumColsLDM
    values, counts = np.unique(sumColsLDM, return_counts = True)

    for i in range(len(sumColsLDM)):
        c = sumColsLDM[i]

        Nc = counts[np.where(values == c)]

        if c+1 in values:
            Nc1 = counts[np.where(values == c+1)]
        else: 
           Nc1 = 0

        c_new = (c+1) * Nc1/Nc
        sumColsLDM[i] = c_new

    # convert adjusted counts to an average
    adjustedPD = np.true_divide(sumColsLDM, num_cols_LDM)
    return adjustedPD

def simpleGoodTuring(LDM):
    #convert ldm to numpy array
    PD = computePD(LDM)
    if 1 in PD:
        raise ValueError("PD cannot be completely sparse")

    LDM = np.array(LDM)
    num_cols_LDM = len(LDM)
 
    # adds up all the "columns" in the LDM
    sumColsLDM = np.sum(LDM, axis = 0)

    count_dict = {}
    for i in range(len(sumColsLDM)):
        if sumColsLDM[i] != 0:
            count_dict[i] = int(sumColsLDM[i])

    max_ = max(count_dict.values())
    L = [i for i in range(len(LDM[0]))]
    
    SGT = simple_good_turing.SimpleGoodTuring(count_dict, max_)
    new_prop = SGT.run_sgt(L)

    # sum_=0
    # for x in new_prop.values():
    #     sum_ +=x
    # print("sum ", sum_)

    PD = [[] for i in range(len(sumColsLDM))]

    for i in new_prop:
        PD[i] = new_prop[i]

    return PD


def plotHeatMap(LDM):
    # Transpose LDM generated so that simplex vectors are column vectors
    LDMTransposed = [list(i) for i in zip(*LDM)]
    map = sns.heatmap(LDMTransposed, linewidth=0.)
    plt.show()