#!/bin/sh

#"Usage error: 1. require dataset name, \n\
#                                2. number of holdout sets, \n\
#                                3. size of holdout set, \n\
#                                4. model to test, \n\
#                                5. whether to train/inference \n\
#                                6. input model number, \n"



# for analysis
readonly TRIAL_NUM=3
readonly MODEL_NUM=3
readonly SIZE_HOLDOUT=10
readonly SIZE_TARGET=4
readonly MODEL_NAME=Decision_Tree_max_depth 
readonly DATASET_NAME=EEG_Eye_State
# readonly TRIAL_NUM=1 already defined in analysis
readonly PLOT_TITLE=Decision_tree
readonly X_AXIS=Estimators
readonly Y_AXIS=Metric


#python3 Trial_Setup_Utils.py EEG 10 5 DECISION_TREE inference 1
#python3 Trial_Setup_Utils.py EEG 5 KNN 
# python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE $MODEL $MODE $MODEL_NUMBER 

python3 run_analysis.py $TRIAL_NUM $MODEL_NUM $SIZE_HOLDOUT $SIZE_TARGET
python3 run_graphing.py $DATASET_NAME $TRIAL_NUM $MODEL_NAME $PLOT_TITLE $X_AXIS $Y_AXIS 
