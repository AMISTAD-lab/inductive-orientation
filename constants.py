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
    ABALONE = "Abalone"
    BANK_MARKETING = "Bank_Marketing"
    CAR_EVALUATION = "Car_Evaluation"
    LETTER_RECOGNITION = "Letter_Recognition"
    OBESITY = "Obesity"
    SPAM = "Spam"
    WINE_QUALITY = "Wine_Quality"

# TODO: add an Enum for metric types
class MetricNames(Enum):
    ALGORITHM_BIAS = "algorithmic_bias"
    ENTROPIC_EXPRESSIVITY = "entropic_expressivity"
    ALGORITHMIC_CAPACITY = "algorithmic_capacity"
    ALGORITHM_BIAS_SIZE_1 = "algorithmic_bias_size_1"
    ALGORITHM_BIAS_SIZE_2 = "algorithmic_bias_size_2"
    ALGORITHM_BIAS_SIZE_3 = "algorithmic_bias_size_3"
    ALGORITHM_BIAS_SIZE_4 = "algorithmic_bias_size_4"
    ALGORITHM_BIAS_SIZE_5 = "algorithmic_bias_size_5"


ALGORITHM_BIAS = "algorithmic_bias"
RESULTS_FOLDER = "./../../../big/erchen/inductive_orientation"
# RESULTS_FOLDER = "."

MODEL_TO_TRIAL_NUMS = {ModelNamesMetrics.KNN_NEIGHBORS.value: {DatasetNames.EEG_EYE_STATE.value : [100, 300],
                                                                         DatasetNames.SEMIRANDOM.value : [109, 308], 
                                                                         DatasetNames.SHOPPER_INTENTION.value : [118, 316],
                                                                         DatasetNames.SHOPPER_INTENTION_BALANCED.value : [200, 324]},

                      ModelNamesMetrics.GAUSSIAN_PROCESS_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : [101],
                                                                                    DatasetNames.SEMIRANDOM.value : [110], 
                                                                                    DatasetNames.SHOPPER_INTENTION.value : [119],
                                                                                    DatasetNames.SHOPPER_INTENTION_BALANCED.value : [201] },

                      ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value: {DatasetNames.EEG_EYE_STATE.value : [102, 301],
                                                                                  DatasetNames.SEMIRANDOM.value : [111, 309], 
                                                                                  DatasetNames.SHOPPER_INTENTION.value : [120, 317],
                                                                                  DatasetNames.SHOPPER_INTENTION_BALANCED.value : [202, 325]},

                      ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : [103, 302],
                                                                              DatasetNames.SEMIRANDOM.value : [112, 310], 
                                                                              DatasetNames.SHOPPER_INTENTION.value : [121, 318],
                                                                              DatasetNames.SHOPPER_INTENTION_BALANCED.value : [203, 326]},

                      ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : [104, 303],
                                                                                 DatasetNames.SEMIRANDOM.value : [113, 311], 
                                                                                 DatasetNames.SHOPPER_INTENTION.value : [122, 319],
                                                                                 DatasetNames.SHOPPER_INTENTION_BALANCED.value : [204, 327]},

                      ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value: {DatasetNames.EEG_EYE_STATE.value : [105, 304],
                                                                                       DatasetNames.SEMIRANDOM.value : [114, 312], 
                                                                                       DatasetNames.SHOPPER_INTENTION.value : [123, 320],
                                                                                       DatasetNames.SHOPPER_INTENTION_BALANCED.value : [205, 328]},

                      ModelNamesMetrics.RANDOM_FOREST_DEPTH.value: {DatasetNames.EEG_EYE_STATE.value : [106, 305],
                                                                              DatasetNames.SEMIRANDOM.value : [115, 313], 
                                                                              DatasetNames.SHOPPER_INTENTION.value : [124, 321],
                                                                              DatasetNames.SHOPPER_INTENTION_BALANCED.value : [206, 329]},

                      ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value: {DatasetNames.EEG_EYE_STATE.value : [107, 306],
                                                                                   DatasetNames.SEMIRANDOM.value : [116, 314], 
                                                                                   DatasetNames.SHOPPER_INTENTION.value : [125, 322],
                                                                                   DatasetNames.SHOPPER_INTENTION_BALANCED.value : [207, 330]},

                      ModelNamesMetrics.ADABOOST_ESTIMATORS.value: {DatasetNames.EEG_EYE_STATE.value : [108, 307],
                                                                              DatasetNames.SEMIRANDOM.value : [117, 315],
                                                                              DatasetNames.SHOPPER_INTENTION.value : [126, 323],
                                                                              DatasetNames.SHOPPER_INTENTION_BALANCED.value : [208, 331]}}

PARAMETER = "parameter"

# aggregate methods

class AggregateNames(Enum):
    MAX = "Max"
    AVG = "Avg"
