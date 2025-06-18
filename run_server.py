import subprocess
from constants import *
from itertools import product

EEG_EYE_STATE = "EEG_Eye_State"
SEMIRANDOM = "SemiRandom"
SHOPPER_INTENTION = "Shopper_Intention"
SHOPPER_INTENTION_BALANCED = "Shopper_Intention_Balanced"
# datasets = [EEG_EYE_STATE]#["SemiRandom_smaller"]#DatasetNames.LETTER_RECOGNITION.value]#"EEG_Eye_State_smaller"]
datasets = [
          EEG_EYE_STATE, 
          SEMIRANDOM, 
          SHOPPER_INTENTION,
          SHOPPER_INTENTION_BALANCED,
          DatasetNames.BANK_MARKETING.value, 
          DatasetNames.ABALONE.value, 
          DatasetNames.CAR_EVALUATION.value, 
          DatasetNames.LETTER_RECOGNITION.value, 
          DatasetNames.OBESITY.value,
          DatasetNames.SPAM.value,
          DatasetNames.WINE_QUALITY.value,
          ]
# "Shopper_Intention" EEG_Eye_State
num_holdouts = [100]
holdout_sizes = [5]
starting_experiment = 641 # trial 427 (letter recognition) analysis never ran for some reason
print("yolo")
# exp_params = [{"model": ModelNamesMetrics.KNN_NEIGHBORS.value, "lower_range": 1, "upper_range":180, "step": 5, "x-axis": "neighbors", "y-axis": "metric"}]#[{"model": ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value, "lower_range": 1, "upper_range":20, "step": 1, "x-axis": "max depth", "y-axis": "metric"},  ]
exp_params = [ {"model": ModelNamesMetrics.NEAREST_CENTROID_THRESHOLD.value, "lower_range":0, "upper_range":116, "step": 5, "x-axis": "shrink threshold", "y-axis": "metric"}
  
  #{"model": ModelNamesMetrics.MLP_LAYERS.value, "lower_range": 1, "upper_range": 25, "step": 1, "x-axis" : "hidden layers", "y-axis":"metric"}
  
            #{"model": ModelNamesMetrics.KNN_NEIGHBORS.value, "lower_range": 1, "upper_range":200, "step": 5, "x-axis": "neighbors", "y-axis": "metric"},
#               {"model": ModelNamesMetrics.GAUSSIAN_PROCESS_MAX_ITER.value, "lower_range": 1, "upper_range":200, "step": 10, "x-axis": "max iterations", "y-axis": "metric"},
               #{"model": ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value, "lower_range": 1, "upper_range":4, "step": 3, "x-axis": "max depth", "y-axis": "metric"},  
#               {"model": ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value, "lower_range": 1, "upper_range":1000, "step": 50, "x-axis": "max iterations", "y-axis": "metric"},               
#               {"model": ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value, "lower_range": 1, "upper_range":1000, "step": 50, "x-axis": "max iterations", "y-axis": "metric"},
#               {"model": ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value, "lower_range": 1, "upper_range":200, "step": 10, "x-axis": "max iterations", "y-axis": "metric"},
#               {"model": ModelNamesMetrics.RANDOM_FOREST_DEPTH.value, "lower_range": 1, "upper_range":70, "step": 5, "x-axis": "depth", "y-axis": "metric"},
#               {"model": ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value, "lower_range": 1, "upper_range":200, "step": 5, "x-axis": "estimators", "y-axis": "metric"},
#               {"model": ModelNamesMetrics.ADABOOST_ESTIMATORS.value, "lower_range": 1, "upper_range":100, "step": 5, "x-axis": "estimators", "y-axis": "metric"}
             ]

for i, (dataset, num_holdout, holdout_size, exp_param) in enumerate(product(datasets, num_holdouts, holdout_sizes, exp_params)):
    current_experiment = i+starting_experiment
    print(starting_experiment+i)
    # if current_experiment != 427: continue

    training_args = ["python3", "Trial_Setup_Utils.py", "--dataset", dataset, "--num_holdout", str(num_holdout), "--size_holdout", 
    str(holdout_size), "--model_name", exp_param["model"], "--mode", "training", "--lower", str(exp_param["lower_range"]),
    "--upper", str(exp_param["upper_range"]), "--step", str(exp_param["step"]), "--model_num", str(current_experiment)]
    subprocess.run(training_args)
    
    inference_args = ["python3", "Trial_Setup_Utils.py", "--dataset", dataset, "--num_holdout", str(num_holdout), \
                      "--size_holdout", str(holdout_size), "--model_name", exp_param["model"], "--mode", "inference",\
                      "--lower", str(exp_param["lower_range"]), "--upper", str(exp_param["upper_range"]), "--model_num",\
                        str(current_experiment), "--step", str(exp_param["step"])]
    subprocess.run(inference_args)
    
    analysis_args = ["python3", "run_analysis.py", "--trial_num", str(current_experiment), "--model", str(current_experiment), "--holdout_size", str(holdout_size)]
    subprocess.run(analysis_args)
    print("\n\nANALYSIS DONE\n\n")
    graphing_args = ["python3", "run_graphing.py",  dataset, str(current_experiment), exp_param["model"], exp_param["model"], exp_param["x-axis"], exp_param["y-axis"]]
    subprocess.run(graphing_args)
    print(dataset)
    print("\nGRAPHING DONE\n\n")

    accuracies_args = ["python3", "graph_accuracies.py",  "--dataset", dataset, "--model_num", str(current_experiment)]
    subprocess.run(accuracies_args)



# RUNS BOTH TRAINING AND INFERENCE
# python3 Trial_Setup_Utils.py --dataset $DATASET --num_holdout $NUMBER_HOLDOUT --size_holdout $HOLDOUT_SIZE --model_name $MODEL --mode training --lower $LOWER_RANGE --upper $UPPER_RANGE
# python3 Trial_Setup_Utils.py --dataset $DATASET --num_holdout $NUMBER_HOLDOUT --size_holdout $HOLDOUT_SIZE --model_name $MODEL --mode inference --lower $LOWER_RANGE --upper $UPPER_RANGE --model_num $MODEL_NUMBER
