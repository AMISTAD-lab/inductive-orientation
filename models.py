"""
This file includes the setup functions for testing classification models. Each inputs one parameter and returns the specified version of the model.

We test: (1) ensemble methods (adaboost, random forest)
         (2) linear models (Logistic regression, passive aggressive)

*https://medium.com/geekculture/passive-aggressive-algorithm-for-big-data-models-8cd535ceb2e6#:~:text=Passive%2DAggressive%20algorithms%20are%20a,of%20the%20online%2Dlearning%20algorithms.
* get info: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
"""

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier


from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid

# ensemble methods
seed = 42 # set seed 

# ensemble methods
"""
The goal of ensemble methods is to combine the predictions of several base estimators built with a 
given learning algorithm in order to improve generalizability / robustness over a single estimator.

"""
def adaboost_n_estimators(n_estimators:int):
    "default n_estimators = 50"
    return AdaBoostClassifier(n_estimators=int(n_estimators)) 

def random_forest_n_estimators(n_estimators:int):
    return RandomForestClassifier(n_estimators=int(n_estimators))

def random_forest_depth_setup(max_depth:int):
    return RandomForestClassifier(max_depth=int(max_depth), n_estimators=10) #TODO: could also try max_features

def extra_trees_setup_n(n_estimators:int):
    return ExtraTreesClassifier(n_estimators=int(n_estimators))


# discriminant analysis (QDA has no metric)
def QDA():
    return QuadraticDiscriminantAnalysis()

# gaussian process
def gaussian_process_max_iter_predict(max_iter_predict:int):
    return GaussianProcessClassifier(max_iter_predict=int(max_iter_predict),random_state=seed)

# trees
def decision_tree_max_depth(max_depth:int):
    return DecisionTreeClassifier(max_depth=int(max_depth)) #random_state=42)

def knn_n_neighbors(n_neigbors:int):
    return KNeighborsClassifier(n_neighbors =int(n_neigbors))

def nearest_centroid(shrink_threshold:int):
    threshold = float(shrink_threshold/100) + 1e-7
    return NearestCentroid(shrink_threshold=threshold, metric='euclidean')

def radiusnn_radius(radius:float):
    return RadiusNeighborsClassifier(radius=radius)

# naive bayes (can try more, multinomail, etc.)
def nb_gaussian():
    return GaussianNB()

# linear models
def logistic_regression_max_iter(max_iter:int): # default = 100
    return LogisticRegression(max_iter=int(max_iter), random_state=seed)

def passive_aggressive(max_iter:int):
    PassiveAggressiveClassifier(max_iter=int(max_iter))


# SVC models
def linear_SVC_max_iter(max_iter:int):
    return LinearSVC(max_iter=int(max_iter), random_state=seed)

def c_SVC_max_iter(max_iter:int):
    return SVC(max_iter=int(max_iter), random_state=seed)

# MLP models
def MLP_hidden_layer(n_hidden_layers:int):
    return MLPClassifier(hidden_layer_sizes=int(n_hidden_layers), random_state=seed, max_iter=500)


