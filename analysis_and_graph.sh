#!/bin/sh

#"Usage error: 1. require dataset name, \n\
#                                2. number of holdout sets, \n\
#                                3. size of holdout set, \n\
#                                4. model to test, \n\
#                                5. whether to train/inference \n\
#                                6. input model number, \n"



readonly DATASET=EEG
readonly NUMBER_HOLDOUT=10
readonly HOLDOUT_SIZE=5
readonly MODEL=DECISION_TREE
readonly MODE=inference
readonly MODEL_NUMBER=1

# for analysis
readonly TRIAL_NUM=4
readonly MODEL_NUM=2
readonly SIZE_HOLDOUT=5
readonly SIZE_TARGET=4
readonly MODEL_NAME=Adaboost

# for graphing
# for analysis:
# if len(sys.argv) != 9 and len(sys.argv) != 8:
#     sys.exit("Usage error: 1. require dataset name, \n\
#                            2. trial number, \n\
#                            3. plot title(string), \n\
#                            4. x-axis label \n\
#                            5. input model number, \n")
# argv_dataset = sys.argv[1]
# trial_num = int(sys.argv[2])
# argv_model = sys.argv[3] # string model to test
# x_label = sys.argv[4]
# y_label = sys.argv[5]
readonly DATASET_NAME=EEG_Eye_State
# readonly TRIAL_NUM=1 already defined in analysis
readonly PLOT_TITLE=Adaboost
readonly X_AXIS=Estimators
readonly Y_AXIS=Metric


#python3 Trial_Setup_Utils.py EEG 10 5 DECISION_TREE inference 1
#python3 Trial_Setup_Utils.py EEG 5 KNN 
# python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE $MODEL training
# python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE $MODEL $MODE $MODEL_NUMBER 

#python3 run_analysis.py $TRIAL_NUM $MODEL_NUM $SIZE_HOLDOUT $SIZE_TARGET
python3 run_graphing.py $DATASET_NAME $TRIAL_NUM $MODEL_NAME $PLOT_TITLE $X_AXIS $Y_AXIS 
