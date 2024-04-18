"""Enum model naming system for consistent spelling"""

from enum import Enum

class ModelNamesMetrics(Enum):
    DECISION_TREE_MAX_DEPTH = "Decision_Tree_max_depth"
    KNN_NEIGHBORS = "KNN_neighbors"
    RANDOM_FOREST_DEPTH = "Random_Forest_depth"
    RANDOM_FOREST_ESTIMATORS = "Random_Forest_estimators"
    ADABOOST_ESTIMATORS = "Adaboost_estimators"
    GAUSSIAN_PROCESS_MAX_ITER = "Gaussian_Process_max_iter" # slow to run
    LINEAR_SVC_MAX_ITER = "Linear_SVC_max_iter"
    C_SUPPORT_SVC_MAX_ITER = "C_Support_max_iter"
    LOGISTIC_REGRESSION_MAX_ITER = "Logistic_Regression_max_iter"

class DatasetNames(Enum):
    EEG_EYE_STATE = "EEG_Eye_State"
    SEMIRANDOM = "SemiRandom"
    SHOPPER_INTENTION = "Shopper_Intention"
    SHOPPER_INTENTION_BALANCED = "Shopper_Intention_Balanced"

# TODO: add an Enum for metric types
class MetricNames(Enum):
    ALGORITHM_BIAS = "algorithmic_bias"
    ENTROPIC_EXPRESSIVITY = "entropic_expressivity"
    ALGORITHMIC_CAPACITY = "algorithmic_capacity"

ALGORITHM_BIAS = "algorithmic_bias"
RESULTS_FOLDER = "./../../../big/erchen/inductive_orientation"
# RESULTS_FOLDER = "."

MODEL_TO_TRIAL_NUMS = {ModelNamesMetrics.KNN_NEIGHBORS.value: {DatasetNames.EEG_EYE_STATE.value : 100,
                                                                         DatasetNames.SEMIRANDOM.value : 109, 
                                                                         DatasetNames.SHOPPER_INTENTION.value : 118,
                                                                         DatasetNames.SHOPPER_INTENTION_BALANCED.value : 200},

                      ModelNamesMetrics.GAUSSIAN_PROCESS_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : 101,
                                                                                    DatasetNames.SEMIRANDOM.value : 110, 
                                                                                    DatasetNames.SHOPPER_INTENTION.value : 119,
                                                                                    DatasetNames.SHOPPER_INTENTION_BALANCED.value : 201},

                      ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value: {DatasetNames.EEG_EYE_STATE.value : 102,
                                                                                  DatasetNames.SEMIRANDOM.value : 111, 
                                                                                  DatasetNames.SHOPPER_INTENTION.value : 120,
                                                                                  DatasetNames.SHOPPER_INTENTION_BALANCED.value : 202},

                      ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : 103,
                                                                              DatasetNames.SEMIRANDOM.value : 112, 
                                                                              DatasetNames.SHOPPER_INTENTION.value : 121,
                                                                              DatasetNames.SHOPPER_INTENTION_BALANCED.value : 203},

                      ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : 104,
                                                                                 DatasetNames.SEMIRANDOM.value : 113, 
                                                                                 DatasetNames.SHOPPER_INTENTION.value : 122,
                                                                                 DatasetNames.SHOPPER_INTENTION_BALANCED.value : 204},

                      ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : 105,
                                                                                       DatasetNames.SEMIRANDOM.value : 114, 
                                                                                       DatasetNames.SHOPPER_INTENTION.value : 123,
                                                                                       DatasetNames.SHOPPER_INTENTION_BALANCED.value : 205},

                      ModelNamesMetrics.RANDOM_FOREST_DEPTH.value: {DatasetNames.EEG_EYE_STATE.value : 106,
                                                                              DatasetNames.SEMIRANDOM.value : 115, 
                                                                              DatasetNames.SHOPPER_INTENTION.value : 124,
                                                                              DatasetNames.SHOPPER_INTENTION_BALANCED.value : 206},

                      ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value: {DatasetNames.EEG_EYE_STATE.value : 107,
                                                                                   DatasetNames.SEMIRANDOM.value : 116, 
                                                                                   DatasetNames.SHOPPER_INTENTION.value : 125,
                                                                                   DatasetNames.SHOPPER_INTENTION_BALANCED.value : 207},

                      ModelNamesMetrics.ADABOOST_ESTIMATORS.value: {DatasetNames.EEG_EYE_STATE.value : 108,
                                                                              DatasetNames.SEMIRANDOM.value : 117, 
                                                                              DatasetNames.SHOPPER_INTENTION.value : 126,
                                                                              DatasetNames.SHOPPER_INTENTION_BALANCED.value : 208}}