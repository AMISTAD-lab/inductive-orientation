#!/bin/sh

# "Usage error: 1. require dataset name, \n\
#                                 2. number of holdout sets, \n\
#                                 3. size of holdout set, \n\
#                                 4. model to test, \n\
#                                 5. whether to train/inference \n\
#                                 6. lower range of metrics to test \n\
#                                 7. upper range of metrics to test \n\
#                                 8. input model number (only for inference), \n"
# readonly DATASET=EEG
# readonly NUMBER_HOLDOUT=10
# readonly HOLDOUT_SIZE=5
# readonly MODEL=DECISION_TREE
# readonly MODE=inference
# readonly MODEL_NUMBER=1
# readonly LOWER_RANGE=1
# readonly UPPER_RANGE=50

readonly DATASET=EEG
readonly NUMBER_HOLDOUT=10
readonly HOLDOUT_SIZE=5
readonly MODEL=Decision_Tree
readonly MODEL_NUMBER=3
readonly LOWER_RANGE=1
readonly UPPER_RANGE=3


# RUNS BOTH TRAINING AND INFERENCE
#python3 Trial_Setup_Utils.py EEG 10 5 DECISION_TREE inference 1
#python3 Trial_Setup_Utils.py EEG 5 KNN 
python3 Trial_Setup_Utils.py --dataset $DATASET --num_holdout $NUMBER_HOLDOUT --size_holdout $HOLDOUT_SIZE --model_name $MODEL --mode training --lower $LOWER_RANGE --upper $UPPER_RANGE
python3 Trial_Setup_Utils.py --dataset $DATASET --num_holdout $NUMBER_HOLDOUT --size_holdout $HOLDOUT_SIZE --model_name $MODEL --mode inference --lower $LOWER_RANGE --upper $UPPER_RANGE --model_num $MODEL_NUMBER

# python3 Trial_Setup_Utils.py --dataset $DATASET --num_holdout $NUMBER_HOLDOUT --size_holdout $HOLDOUT_SIZE --model_name $MODEL --mode training --lower $LOWER_RANGE --upper $UPPER_RANGE
