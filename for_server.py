import itertools
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


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

def random_uniform(X_train, y_train, num_entries, i=0):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    indices = indices[:num_entries] #no replacement
    random_subset_X = [X_train[i] for i in indices]
    random_subset_y = [y_train[i] for i in indices]
    return random_subset_X, random_subset_y
# def random_uniform(X_train, y_train, num_entries, i=0):
#     indices = np.arange(len(X_train))
#     np.random.shuffle(indices)
#     indices = indices[:num_entries] #no replacement
#     random_subset_X = X_train.iloc[indices]
#     random_subset_y = y_train.iloc[indices]
#     return random_subset_X, random_subset_y

'''splitting the dataset - no overlap between columns'''
def split_dataset(X_train, y_train, num_entries, i=0):
    start_index = i*num_entries
    subset_X = X_train[start_index:start_index+num_entries]
    subset_y = y_train[start_index:start_index+num_entries]
    return subset_X, subset_y

def fixed_dataset(X_train, y_train, num_entries, i=0):
    subset_X = X_train[:num_entries]
    subset_y = y_train[:num_entries]
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
def getSimplex(clf, X_test, classes, all_labels, sparse=True):
    #the sparse case
    if sparse:
        predicted_labels = clf.predict(X_test) # a list of predictions, one for each thing in X_test
        predicted_labels = [int(x) for x in predicted_labels]
        for i in range(0, len(all_labels)): #returns the index of all_labels that matches the predicted_labels
            if tuple(predicted_labels) == all_labels[i]:
                return i
    #predict proba case
    else:
        simplex_vector = []
        alpha = 0.000001 # Used for alpha smoothing
        sum_probs = 0    # Used for normailization later
        y_pred_prob = clf.predict_proba(X_test)
        # Iterate through all_labels and calculate probabilities
        # based on y_pred_prob values
        for i in range(0, len(all_labels)):
            #current_prob = alpha
            current_prob = 1.0

            for j in range(0, len(all_labels[i])):
                for class_index in classes:
                    if (all_labels[i][j] == class_index): #and (class_index < len(y_pred_prob[j]))):
                            current_prob *= (alpha if y_pred_prob[j][class_index] == 0 else y_pred_prob[j][class_index])
                            #current_prob = 0.000001 * 0.000001 if y_pred_prob[holdoutsamplei][class_index]
            sum_probs += current_prob
            simplex_vector.append(current_prob)
            simplex_vector = np.array(simplex_vector)
            return simplex_vector

def getLDM(clf, X_train, X_test, y_train, classes=[0,1], num_datasets=5, num_repeat=1, proportion_of_dataset=0.1, sparse=True, data_generation=random_uniform):
    # Initialize a labelling distribution matrix to be constructed
    num_holdout_samples = len(X_test)
    all_labels = list(itertools.product(classes, repeat=num_holdout_samples))
    #sparse case
    if sparse:
        LDM = [len(classes)**num_holdout_samples] #the first thing in LDM is the number of entries in the PD vector
        

        for i in range(num_datasets):
            if data_generation == random_uniform or data_generation == fixed_dataset:
                num_entries = int(proportion_of_dataset * len(X_train))

            elif data_generation == split_dataset or data_generation:
                num_entries = len(X_train)//num_datasets

            subset_X, subset_y = data_generation(X_train, y_train, num_entries, i=i)
            Pf = [] # shares the same training data, has length equal to num_repeat, average becomes P bar F/ one column of the LDM
            for repeat in range(num_repeat):
                clf.fit(subset_X, subset_y)
                outcome = getSimplex(clf, X_test, classes, all_labels, sparse) #outcome is just an index
                Pf.append(outcome)

            LDM.append(Pf)

    #predict_proba case
    else:
        LDM = []
        for i in range(num_datasets):
            # Shuffle the training labels randomly to generate different training
            # sets for each iteration
            clf.classes_ = classes
            # Train the model using the current training set
            if data_generation == random_uniform or data_generation == fixed_dataset:
                num_entries = int(proportion_of_dataset * len(X_train))

            elif data_generation == split_dataset or data_generation:
                num_entries = len(X_train)//num_datasets

            subset_X, subset_y = data_generation(X_train, y_train, num_entries, i=i) 

            averaged_simplex_vector = np.zeros(len(all_labels))
            for i in range(num_repeat):
                clf.fit(subset_X, subset_y)
                current_simplex_vector = getSimplex(clf, X_test, classes, all_labels,sparse)
                averaged_simplex_vector += current_simplex_vector
            averaged_simplex_vector /= num_repeat
            # Obtain simplex vector for current training set
            LDM.append(averaged_simplex_vector)
    return LDM


def computePD(LDM, sparse=True):
    #sprase case
    if sparse:
        PD_length = LDM[0]
        LDM = LDM[1:]
        values, counts = np.unique(LDM, return_counts=True)
        counts = counts / np.sum(counts) # to get the propobability distribution at each non-zero index
        PD = np.zeros(PD_length)
        for i, index in enumerate(values):
            PD[index] = counts[i]
    #predict_proba case
    else:
        PD = np.mean(LDM, axis=0)
    return PD

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