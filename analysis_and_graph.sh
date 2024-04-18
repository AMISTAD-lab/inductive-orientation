#!/bin/sh

#"Usage error: 1. require dataset name, \n\
#                                2. number of holdout sets, \n\
#                                3. size of holdout set, \n\
#                                4. model to test, \n\
#                                5. whether to train/inference \n\
#                                6. input model number, \n"



# for analysis
readonly TRIAL_NUM=4
readonly MODEL_NUM=4
readonly SIZE_HOLDOUT=5 # size of each holdout (not number of holdout sets)
readonly SIZE_TARGET=4
readonly MODEL_NAME=Decision
readonly DATASET_NAME=EEG_Eye_State
# readonly TRIAL_NUM=1 already defined in analysis
readonly PLOT_TITLE=Decision_Tree_depth
readonly X_AXIS=Depth
readonly Y_AXIS=Metric # discontinued


#python3 Trial_Setup_Utils.py EEG 10 5 DECISION_TREE inference 1
#python3 Trial_Setup_Utils.py EEG 5 KNN 
# python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE $MODEL $MODE $MODEL_NUMBER 

python3 run_analysis.py --trial_num $TRIAL_NUM --model_num $MODEL_NUM --holdout_size $SIZE_HOLDOUT
python3 run_graphing.py $DATASET_NAME $TRIAL_NUM $MODEL_NAME $PLOT_TITLE $X_AXIS $Y_AXIS 
