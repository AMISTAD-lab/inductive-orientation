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
readonly MODEL_NUMBER=2



#python3 Trial_Setup_Utils.py EEG 10 5 DECISION_TREE inference 1
#python3 Trial_Setup_Utils.py EEG 5 KNN 
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE $MODEL $MODE $MODEL_NUMBER

