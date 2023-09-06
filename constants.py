"""Enum model naming system for consistent spelling"""

from enum import Enum

class ModelNamesMetrics(Enum):
    DECISION_TREE_MAX_DEPTH = "Decision_Tree_max_depth"
    KNN_NEIGHBORS = "KNN_neighbors"
    RANDOM_FOREST_DEPTH = "Random_Forest_depth"
    RANDOM_FOREST_ESTIMATORS = "Random_Forest_estimators"
    ADABOOST_ESTIMATORS = "Adaboost_estimators"
    GUASSIAN_PROCESS_MAX_ITER = "Guassian_Process_max_iter"
    LINEAR_SVC_MAX_ITER = "Linear_SVC_max_iter"
    C_SUPPORT_SVC_MAX_ITER = "C_Support_max_iter"
    LOGISTIC_REGRESSION_MAX_ITER = "Logistic_Regression_max_iter"

# TODO: add an Enum for metric types