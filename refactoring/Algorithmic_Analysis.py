import itertools
import numpy as np
from scipy.stats import entropy
import json
import Inductive_Generator
import os

# analysis
"""
getTarget produces a numpy array with all 0's execept a 1 at the index representing the correct sequence of labeling/
inputs:
    X_test: a pandas dataframe representing the features of the test data
    y_test: an numpy array representing the correct labels of the test data
    classes: a list represnting the possible classes
    min_num_accurate: number of elements in the holdout set = X_test that should be identified correctly to be considered as a target
output:
    target: a numpy array  with all 0's expect a 1 at the correct index
"""
def getTarget(X_test, y_test, min_num_accurate, classes=[0,1]):
    y_test = y_test.astype("int")
    num_holdout_samples = len(X_test)
    all_labels = np.array(list(itertools.product(classes, repeat=num_holdout_samples)))
    testForCorrectLabels = all_labels == y_test
    # print("all_labels: ", all_labels)
    # print("y_test: ", y_test)
    # print("testForCorrectLabels: ", testForCorrectLabels)
    countNumCorrect = np.sum(testForCorrectLabels, axis = 1)
    target = 1 * (countNumCorrect >= min_num_accurate)
    return target

'''
computeAlgorithmicBias takes in a target vector and a pD_vector (both numpy arrays)
Parameters:
1. target = k-hot vector (acceptable labelings/classification)
2. pD_vector

output:
  algorithmic Bias
'''
def computeAlgorithmicBias(target, pD_vector):
  if len(target) != len(pD_vector):
    raise Exception("Length of target vector and pD_vector does not match")
  #k = number of acceptable labelings/classification
  k = np.sum(target)

  #len(target) = size of search space
  # k / len(target) = probability of success when uniformly randomly sampling
  algorithmicBias = np.sum(target * pD_vector) - (k / len(target))
  return algorithmicBias

'''
computeEntropy finds the Entropic Expresstivity for a PD vector
input:
    PD: an inductive orientation vector
return:
    The Entropic Expresstivity for a PD vector
'''
def computeEntropy(PD):
    return entropy(PD, base=2)

'''
computeEntropyLDM finds the entropies for each sublist (pf, probability distribution simplex vector) of the LDM
input:
    LDM: an LDM matrix
return:
    entropy_list: a list of doubles that are the entropy values for the corresponding pf
'''
def computeEntropyLDM(LDM):
    def toDistribution(x):
        _, counts = np.unique(x, return_counts=True)
        return counts / sum(counts)
    return np.array([computeEntropy(toDistribution(x)) for x in LDM])

'''
Algorithmic Capacity = Entropy of PD vector - Expected value of (Entropy of each individual PF vector)
Parameters: 
    1. ldm = list of Pf vectors
    2. pD_vector = PD vector calculated from ldm
'''
def computeAlgorithmicCapacity(ldm, pD_vector):

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
singleAnalysis calculates Bias, Entropic Expressivity, and Algorithmic Capacity for a model
inputs:
    file: the path to the json file storing the model's LDM and PD
    target: the target vector to evaluate the model. If not given, then
            the function will not calculate Bias
'''
def singleAnalysis(file, target = None):
    print("Current file: ", file)
    with open(file) as logs:
        saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
        if type(target) != type(None):
            bias = computeAlgorithmicBias(target, saved_state["PD"])
            print(bias)
        expressivity = computeEntropy(saved_state["LDM"])
        print(expressivity)
        capacity = computeAlgorithmicCapacity(saved_state["LDM"], saved_state["PD"])
        print(capacity)

'''
runAnalysis allows multiple calls to singleAnalysis
inputs:
    file: the path to either a json file or a directry containing all json files
    target: the target vector to evaluate the model. If not given, then
            the function will not calculate Bias
'''
def runAnalysis(file, target=None):
    if file[-4:] == "json":
        singleAnalysis(file, target)
    else:
        logs = os.listdir(file)
        logs = [os.path.join(file, log) for log in logs]
        for log_file in logs:
            singleAnalysis(log_file, target)

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