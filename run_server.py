import subprocess
from constants import *
from itertools import product

EEG_EYE_STATE = "EEG_Eye_State"  
datasets = [EEG_EYE_STATE] # "Shopper_Intention"
num_holdouts = [10]
holdout_sizes = [5]
model_nums = [3]
starting_experiment = 8

# exp_params = [{"model": "KNN", "lower_range": 1, "upper_range":4}]
exp_params = [#{"model": ModelNamesMetrics.KNN_NEIGHBORS.value, "lower_range": 1, "upper_range":3, "step": 1, "x-axis": "neighbors", "y-axis": "metric"}] #,
           #    {"model": ModelNamesMetrics.GAUSSIAN_PROCESS_MAX_ITER.value, "lower_range": 1, "upper_range":20, "step": 10, "x-axis": "max iterations", "y-axis": "metric"}]
               {"model": ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value, "lower_range": 1, "upper_range":20, "step": 10, "x-axis": "max iterations", "y-axis": "metric"}]   
            #   {"model": ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value, "lower_range": 1, "upper_range":20, "step": 10, "x-axis": "max iterations", "y-axis": "metric"},               
            #   {"model": ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value, "lower_range": 1, "upper_range":20, "step": 10, "x-axis": "max iterations", "y-axis": "metric"},
            #   {"model": ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value, "lower_range": 1, "upper_range":20, "step": 10, "x-axis": "depth", "y-axis": "metric"},
            #   {"model": ModelNamesMetrics.RANDOM_FOREST_DEPTH.value, "lower_range": 1, "upper_range":3, "step": 1, "x-axis": "depth", "y-axis": "metric"},
            #   {"model": ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value, "lower_range": 1, "upper_range":3, "step": 1, "x-axis": "depth", "y-axis": "metric"},
            #   {"model": ModelNamesMetrics.ADABOOST_ESTIMATORS.value, "lower_range": 1, "upper_range":3, "step": 1, "x-axis": "depth", "y-axis": "metric"}]


for i, (dataset, num_holdout, holdout_size, model_num, exp_param) in enumerate(product(datasets, num_holdouts, holdout_sizes, 
        model_nums, exp_params)):
    current_experiment = i+starting_experiment
    
    training_args = ["python3", "Trial_Setup_Utils.py", "--dataset", dataset, "--num_holdout", str(num_holdout), "--size_holdout", 
    str(holdout_size), "--model_name", exp_param["model"], "--mode", "training", "--lower", str(exp_param["lower_range"]),
    "--upper", str(exp_param["upper_range"]), "--step", str(exp_param["step"]), "--model_num", str(starting_experiment)]
    subprocess.run(training_args)
    
    inference_args = ["python3", "Trial_Setup_Utils.py", "--dataset", dataset, "--num_holdout", str(num_holdout), "--size_holdout", str(holdout_size), "--model_name", exp_param["model"], "--mode", "inference", "--lower", str(exp_param["lower_range"]), "--upper", str(exp_param["upper_range"]), "--model_num", str(current_experiment), "--step", str(exp_param["step"])]
    subprocess.run(inference_args)
    
    analysis_args = ["python3", "run_analysis.py", "--trial_num", str(current_experiment), "--model", str(current_experiment), "--holdout_size", str(holdout_size)]
    subprocess.run(analysis_args)

    graphing_args = ["python3", "run_graphing.py",  dataset, str(current_experiment), exp_param["model"], exp_param["model"], exp_param["x-axis"], exp_param["y-axis"]]
    subprocess.run(graphing_args)

# RUNS BOTH TRAINING AND INFERENCE
# python3 Trial_Setup_Utils.py --dataset $DATASET --num_holdout $NUMBER_HOLDOUT --size_holdout $HOLDOUT_SIZE --model_name $MODEL --mode training --lower $LOWER_RANGE --upper $UPPER_RANGE
# python3 Trial_Setup_Utils.py --dataset $DATASET --num_holdout $NUMBER_HOLDOUT --size_holdout $HOLDOUT_SIZE --model_name $MODEL --mode inference --lower $LOWER_RANGE --upper $UPPER_RANGE --model_num $MODEL_NUMBER
