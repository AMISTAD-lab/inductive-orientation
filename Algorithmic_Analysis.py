"""
Functions performing

"""

import itertools
import numpy as np
from scipy.stats import entropy
import json
import Inductive_Generator
import os
import pandas as pd
import pdb
def getNTargets(y_tests, min_num_accurate, classes=[0,1]):
    targets = []
    for y_test in y_tests:
        targets.append(getTarget(y_test, min_num_accurate, classes))
    return targets

def getTarget(y_test, min_num_accurate, classes=[0,1]):
    """
    getTarget produces a numpy array with all 0's execept a 1 at the index representing the correct sequence of labeling/
    inputs:
        X_test: a pandas dataframe representing the features of the test data
        y_test: an numpy array representing the correct labels of the test data
        classes: a list represnting the possible classes
        min_num_accurate: the number of elements in the holdout set that must be identified correctly. <= |X_holdout|
    output:
        target: a numpy array  with all 0's expect a 1 at the correct index
    """
    y_test = y_test.astype("int") # true holdout set labeling
    num_holdout_samples = len(y_test)
    all_labels = np.array(list(itertools.product(classes, repeat=num_holdout_samples))) # all possible holdout set labelings of holdout set(??)
    testForCorrectLabels = all_labels == y_test # creates boolean 2D array of whether or not each entry matches y_test
    # print("all_labels: ", all_labels)
    # print("y_test: ", y_test)
    # print("testForCorrectLabels: ", testForCorrectLabels)
    countNumCorrect = np.sum(testForCorrectLabels, axis = 1)
    target = 1 * (countNumCorrect >= min_num_accurate)
    return target


def computeAlgorithmicBias(target, pD_vector):
    '''
    computeAlgorithmicBias takes in a target vector and a pD_vector (both numpy arrays)
    Parameters:
    1. target = k-hot vector (acceptable labelings/classification)
    2. pD_vector

    output:
    algorithmic Bias
    '''
    if len(target) != len(pD_vector):
        raise Exception("Length of target vector and pD_vector does not match")
    #k = number of acceptable labelings/classification
    k = np.sum(target)

    #len(target) = size of search space
    # k / len(target) = probability of success when uniformly randomly sampling
    algorithmicBias = np.dot(target, pD_vector) - (k / len(target)) # np.sum(target * pD_vector) - (k / len(target))
    return algorithmicBias


def computeEntropy(PD):
    '''
    computeEntropy finds the Entropic Expresstivity for a PD vector
    input:
        PD: an inductive orientation vector
    return:
        The Entropic Expresstivity for a PD vector
    '''
    return entropy(PD, base=2)


def computeEntropyLDM(LDM):
    '''
    computeEntropyLDM finds the entropies for each sublist (pf, probability distribution simplex vector) of the LDM
    input:
        LDM: an LDM matrix
    return:
        entropy_list: a list of doubles that are the entropy values for the corresponding pf
    '''
    def toDistribution(x):
        _, counts = np.unique(x, return_counts=True)
        return counts / sum(counts)
    return np.array([computeEntropy(toDistribution(x)) for x in LDM])


def computeAlgorithmicCapacity(ldm, pD_vector):
    '''
    Algorithmic Capacity = Entropy of PD vector - Expected value of (Entropy of each individual PF vector)
    Parameters: 
        1. ldm = list of Pf vectors
        2. pD_vector = PD vector calculated from ldm
    '''
    #entropy_of_Pfs is a list containing the entropy of each individual Pf vector in ldm
    entropy_of_Pfs = np.array(computeEntropyLDM(ldm))
    print("entropy_of_Pfs: ", entropy_of_Pfs)

    #expected value/average of the entropies of each Pf vector
    expected_entropy_of_Pfs = np.mean(entropy_of_Pfs)
    print("Expected = mean of entropy: ", expected_entropy_of_Pfs)

    #calculate the entropy of a pD_vector
    entropy_pD = computeEntropy(pD_vector)
    print("Entropy of PD: ", entropy_pD)

    return entropy_pD - expected_entropy_of_Pfs


def computeVariance(list_of_simplex):
    '''
    computeVariance finds the variance of a list of probability distribution simplex vectors
    input:
        list_of_simplex: a list of probability distribution simplex vectors
    return:
        variance: a double that represents on average how much the each of the probability distribution simplex vectors 
        deviate from the average simplex vector
    '''
    variance = np.var(list_of_simplex, axis=0)
    return variance


def varianceUpToN(list_of_PD):
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
    variance_per_run = []

    for i in range(1, len(list_of_PD)):
        current_variance = computeVariance(list_of_PD[:i])
        variance_per_run.append(current_variance)
        print("Variance after ", i, " runs: ", current_variance)
    return variance_per_run


def singleAnalysis(saved_state, targets):
    '''
    singleAnalysis calculates Bias, Entropic Expressivity, and Algorithmic Capacity for a model
    inputs:
        file: the path to the json file storing the model's LDMs and PDs
        target: a list of target vectors to evaluate the model, each has corresponding PD and LDM. 
        If not given, then the function will not calculate Bias
    '''
    #print("Current file: ", file)
    bias = np.array([])
    expressivity = np.array([])
    capacity = np.array([])
    
    for i, PD in enumerate(saved_state["PDs"]):
        target = targets[i] # each PD has corresponding target
        # append bias
        if target is not None:
            bias_ = computeAlgorithmicBias(target, PD)
        else:
            bias_ = np.nan
        bias = np.append(bias, bias_)
       
        # append expressivity
        expressivity_ = computeEntropy(PD)
        expressivity = np.append(expressivity, expressivity_)

        # append capacity
        capacity_ = computeAlgorithmicCapacity(saved_state["LDMs"][i], PD)
        capacity = np.append(capacity, capacity_)
    #pdb.set_trace()

    return tuple([bias, expressivity, capacity])
    # with open(file) as logs:
    #     saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
    #     if type(target) != type(None):
    #         bias = computeAlgorithmicBias(target, saved_state["PD"])
    #         print(bias)
    #         analytics.append(bias)
    #     else:
    #         analytics.append(None)
    #     # expressivity = computeEntropy(saved_state["LDM"])
    #     # print(expressivity)
    #     expressivity = computeEntropy(saved_state["PD"])
    #     analytics.append(expressivity)
    #     capacity = computeAlgorithmicCapacity(saved_state["LDM"], saved_state["PD"])
    #     print(capacity)
    #     analytics.append(capacity)
    #return tuple(analytics)
        

def runAnalysis(file:str, targets:list=None, name_column=[], bias_column=[], entropic_expressivity_column = [], algorithmic_capacity_column = []):
    '''
    runAnalysis allows multiple calls to singleAnalysis
    inputs:
        file: the path to either a json file or a directry containing all json files
        target: the list of target vectors to evaluate the model. If not given, then
                the function will not calculate Bias
    '''
    if file[-4:] == "json":
        return singleAnalysis(file, targets)

    else:
        logs = os.listdir(file)
        logs = [os.path.join(file, log) for log in logs]
        # pdb.set_trace()
        for log_file in logs:
            with open(log_file) as log:
                saved_state = json.loads(log.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
            bias, entropic_expressivity, algorithmic_capacity = singleAnalysis(saved_state, targets)
            name_column.append(log_file.split("/")[-1].split(".")[0])
            bias_column.append(bias)
            entropic_expressivity_column.append(entropic_expressivity)
            algorithmic_capacity_column.append(algorithmic_capacity)
        summary = pd.DataFrame({"model_name":name_column, "algorithmic_bias" : bias_column, \
            "entropic_expressivity" :entropic_expressivity_column, "algorithmic_capacity" : algorithmic_capacity_column})
        summary = summary.sort_values(by= "model_name")
        return summary

# '''
# variance_propdata finds the variance between columns of an LDM, not PD, as the proportion of dataset changes
# inputs:
#     percentages: a list of percentages in double 
# returns:
#     run_number: a list that records the size of the inductive orientation vectors PD's
#     variance_per_run: a list that gives the variance corresponding to the run_number
# '''
# def variance_propdata(percentages, clf, X_train, X_test, y_train, num_repeat=1, classes=[0,1,2], num_datasets=5, sparse=True, data_generation=random_uniform):
#     percentages_list = []
#     variance_list = []
#     for i in percentages:
#         proportion_of_dataset = i
#         percentages_list.append(i)
#         current_ldm = getLDM(clf, X_train, X_test, y_train, num_repeat=num_repeat,classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=sparse, data_generation=data_generation)
#         current_variance = computeVariance(current_ldm)
#         variance_list.append(current_variance)
#         print("proportion of dataset: ", i, " the variance of the Pf's in the LDM with: ", num_datasets, "number of datasets is: ", current_variance)
#     return percentages_list, variance_list


def computeAngle(PD1, PD2):
    '''
    computeAngle finds the radian angle between two inductive orientation vector using dot product
    input:
        PD1: a numpy array that represents an inductive orientation vector
        PD2: a numpy array that represents an inductive orientation vector
    return:
        angle: a double that presents the angle between the two vectors in radians
    '''
    unit_vector_1 = PD1/np.linalg.norm(PD1)
    unit_vector_2 = PD2/np.linalg.norm(PD2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    
    if dot_product >= 1:
        return 0
    
    angle = np.arccos(dot_product)
    angle = round(angle, 7)
    return angle