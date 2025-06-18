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
    MLP_LAYERS = "MLP_hidden_layer"
    NEAREST_CENTROID_THRESHOLD = "Nearest_Centroid_shrink_threshold"


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

# readable form of dataset names for plot titles
READABLE_DATASETS = {
    DatasetNames.EEG_EYE_STATE.value : "EEG Eye State",
    DatasetNames.SEMIRANDOM.value : "Random",
    DatasetNames.SHOPPER_INTENTION.value : "Shopper's Intention",
    DatasetNames.SHOPPER_INTENTION_BALANCED.value : "Shopper's Intention (Balanced)",
    DatasetNames.ABALONE.value : "Abalone",
    DatasetNames.BANK_MARKETING.value : "Bank Marketing",
    DatasetNames.CAR_EVALUATION.value : "Car Evaluation",
    DatasetNames.LETTER_RECOGNITION.value : "Letter Recognition",
    DatasetNames.OBESITY.value : "Obesity",
    DatasetNames.SPAM.value : "Spam",
    DatasetNames.WINE_QUALITY.value : "Wine Quality"
}

READABLE_MODELS = {
    ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value : "Decision Tree",
    ModelNamesMetrics.KNN_NEIGHBORS.value : "K-Nearest Neighbors",
    ModelNamesMetrics.RANDOM_FOREST_DEPTH.value : "Random Forest",
    ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value : "Random Forest",
    ModelNamesMetrics.ADABOOST_ESTIMATORS.value : "Adaboost",
    ModelNamesMetrics.GAUSSIAN_PROCESS_MAX_ITER.value : "Gaussian Process",
    ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value : "Linear SVC",
    ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value : "C-Support SVC",
    ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value : "Logistic Regression",
    ModelNamesMetrics.MLP_LAYERS.value : "MLP",
    ModelNamesMetrics.NEAREST_CENTROID_THRESHOLD.value : "Nearest Centroid"
}



MODEL_TO_X_PARAM = {
    ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value : "Maximum Depth",
    ModelNamesMetrics.KNN_NEIGHBORS.value : "Neighbors",
    ModelNamesMetrics.RANDOM_FOREST_DEPTH.value : "Maximum Depth",
    ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value : "Estimators",
    ModelNamesMetrics.ADABOOST_ESTIMATORS.value : "Estimators",
    ModelNamesMetrics.GAUSSIAN_PROCESS_MAX_ITER.value : "Iterations",
    ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value : "Iterations",
    ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value : "Iterations",
    ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value : "Iterations",
    ModelNamesMetrics.MLP_LAYERS.value : "Hidden Layers",
    ModelNamesMetrics.NEAREST_CENTROID_THRESHOLD.value : "Shrink Threshold"

}

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

MODEL_TO_TRIAL_NUMS = {
    ModelNamesMetrics.KNN_NEIGHBORS.value: {
        DatasetNames.EEG_EYE_STATE.value: [100, 300],
        DatasetNames.SEMIRANDOM.value: [109, 308],
        DatasetNames.SHOPPER_INTENTION.value: [118, 316],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [200, 324],
        DatasetNames.BANK_MARKETING.value: [400],
        DatasetNames.ABALONE.value: [409],
        DatasetNames.CAR_EVALUATION.value: [418],
        DatasetNames.LETTER_RECOGNITION.value: [427],
        DatasetNames.OBESITY.value: [436],
        DatasetNames.SPAM.value: [445],
        DatasetNames.WINE_QUALITY.value: [454]
    },

    ModelNamesMetrics.GAUSSIAN_PROCESS_MAX_ITER.value: {
        DatasetNames.EEG_EYE_STATE.value: [101],
        DatasetNames.SEMIRANDOM.value: [110],
        DatasetNames.SHOPPER_INTENTION.value: [119],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [201],
        DatasetNames.BANK_MARKETING.value: [401],
        DatasetNames.ABALONE.value: [410],
        DatasetNames.CAR_EVALUATION.value: [419],
        DatasetNames.LETTER_RECOGNITION.value: [428],
        DatasetNames.OBESITY.value: [437],
        DatasetNames.SPAM.value: [446],
        DatasetNames.WINE_QUALITY.value: [455]
    },

    ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value: {
        DatasetNames.EEG_EYE_STATE.value: [102,512, 301],
        DatasetNames.SEMIRANDOM.value: [111,513, 309],
        DatasetNames.SHOPPER_INTENTION.value: [120,514, 317],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [515, 202, 325],
        DatasetNames.BANK_MARKETING.value: [402, 515, 402],
        DatasetNames.ABALONE.value: [411, 516,411],
        DatasetNames.CAR_EVALUATION.value: [420,517,420],
        DatasetNames.LETTER_RECOGNITION.value: [429,518,429],
        DatasetNames.OBESITY.value: [438,519,438],
        DatasetNames.SPAM.value: [447,520,447],
        DatasetNames.WINE_QUALITY.value: [456,521,500,456]
    },

    ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value: {
        DatasetNames.EEG_EYE_STATE.value: [103, 302],
        DatasetNames.SEMIRANDOM.value: [112, 310],
        DatasetNames.SHOPPER_INTENTION.value: [121, 318],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [203, 326],
        DatasetNames.BANK_MARKETING.value: [403],
        DatasetNames.ABALONE.value: [412],
        DatasetNames.CAR_EVALUATION.value: [421],
        DatasetNames.LETTER_RECOGNITION.value: [430],
        DatasetNames.OBESITY.value: [439],
        DatasetNames.SPAM.value: [448],
        DatasetNames.WINE_QUALITY.value: [457]
    },

    ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value: {
        DatasetNames.EEG_EYE_STATE.value: [104, 303],
        DatasetNames.SEMIRANDOM.value: [113, 311],
        DatasetNames.SHOPPER_INTENTION.value: [122, 319],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [204, 327],
        DatasetNames.BANK_MARKETING.value: [404],
        DatasetNames.ABALONE.value: [413],
        DatasetNames.CAR_EVALUATION.value: [422],
        DatasetNames.LETTER_RECOGNITION.value: [431],
        DatasetNames.OBESITY.value: [440],
        DatasetNames.SPAM.value: [449],
        DatasetNames.WINE_QUALITY.value: [458]
    },

    ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value: {
        DatasetNames.EEG_EYE_STATE.value: [105, 304],
        DatasetNames.SEMIRANDOM.value: [114, 312],
        DatasetNames.SHOPPER_INTENTION.value: [123, 320],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [205, 328],
        DatasetNames.BANK_MARKETING.value: [405],  # Corrected typo from BANK_MARKTING to BANK_MARKETING
        DatasetNames.ABALONE.value: [414],
        DatasetNames.CAR_EVALUATION.value: [423],
        DatasetNames.LETTER_RECOGNITION.value: [432],
        DatasetNames.OBESITY.value: [441],
        DatasetNames.SPAM.value: [450],
        DatasetNames.WINE_QUALITY.value: [459]
    },

    ModelNamesMetrics.RANDOM_FOREST_DEPTH.value: {
        DatasetNames.EEG_EYE_STATE.value: [106, 305],
        DatasetNames.SEMIRANDOM.value: [115, 313],
        DatasetNames.SHOPPER_INTENTION.value: [124, 321],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [206, 329],
        DatasetNames.BANK_MARKETING.value: [406],
        DatasetNames.ABALONE.value: [415],
        DatasetNames.CAR_EVALUATION.value: [424],
        DatasetNames.LETTER_RECOGNITION.value: [433],
        DatasetNames.OBESITY.value: [442],
        DatasetNames.SPAM.value: [451],
        DatasetNames.WINE_QUALITY.value: [460]
    },

    ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value: {
        DatasetNames.EEG_EYE_STATE.value: [107, 306],
        DatasetNames.SEMIRANDOM.value: [116, 314],
        DatasetNames.SHOPPER_INTENTION.value: [125, 322],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [207, 330],
        DatasetNames.BANK_MARKETING.value: [407],
        DatasetNames.ABALONE.value: [416],
        DatasetNames.CAR_EVALUATION.value: [425],
        DatasetNames.LETTER_RECOGNITION.value: [434],
        DatasetNames.OBESITY.value: [443],
        DatasetNames.SPAM.value: [452],
        DatasetNames.WINE_QUALITY.value: [461],
    },

    ModelNamesMetrics.ADABOOST_ESTIMATORS.value: {
        DatasetNames.EEG_EYE_STATE.value: [108, 307],
        DatasetNames.SEMIRANDOM.value: [117, 315],
        DatasetNames.SHOPPER_INTENTION.value: [126, 323],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [208, 331],
        DatasetNames.BANK_MARKETING.value: [408],
        DatasetNames.ABALONE.value: [417],
        DatasetNames.CAR_EVALUATION.value: [426],
        DatasetNames.LETTER_RECOGNITION.value: [435],
        DatasetNames.OBESITY.value: [444],
        DatasetNames.SPAM.value: [453],
        DatasetNames.WINE_QUALITY.value: [462]
    },

    ModelNamesMetrics.MLP_LAYERS.value : {
        DatasetNames.EEG_EYE_STATE.value: [570],
        DatasetNames.SEMIRANDOM.value: [571],
        DatasetNames.SHOPPER_INTENTION.value: [572],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [580],
        DatasetNames.BANK_MARKETING.value: [573],
        DatasetNames.ABALONE.value: [574],
        DatasetNames.CAR_EVALUATION.value: [575],
        DatasetNames.LETTER_RECOGNITION.value: [576],
        DatasetNames.OBESITY.value: [577],
        DatasetNames.SPAM.value: [578],
        DatasetNames.WINE_QUALITY.value: [579]    
        },

    ModelNamesMetrics.NEAREST_CENTROID_THRESHOLD.value : {
        DatasetNames.EEG_EYE_STATE.value: [641],
        DatasetNames.SEMIRANDOM.value: [642],
        DatasetNames.SHOPPER_INTENTION.value: [643],
        DatasetNames.SHOPPER_INTENTION_BALANCED.value: [644],
        DatasetNames.BANK_MARKETING.value: [645],
        DatasetNames.ABALONE.value: [646],
        DatasetNames.CAR_EVALUATION.value: [647],
        DatasetNames.LETTER_RECOGNITION.value: [648],
        DatasetNames.OBESITY.value: [649],
        DatasetNames.SPAM.value: [650],
        DatasetNames.WINE_QUALITY.value: [651]    
        }
}

PARAMETER = "parameter"

# aggregate methods

class AggregateNames(Enum):
    MAX = "Max"
    AVG = "Avg"

class MetricNames(Enum):
    ALG_BIAS_SEM = "alg_sem"
    EXP_CAP_SEM = "exp_cap_sem"
