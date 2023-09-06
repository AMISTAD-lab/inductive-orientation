cd ..

# training/inference
readonly DATASET=EEG
readonly NUMBER_HOLDOUT=10
readonly HOLDOUT_SIZE=5
readonly MODEL=Decision_Tree
readonly MODEL_NUMBER=3
readonly LOWER_RANGE=1
readonly UPPER_RANGE=70


# RUNS BOTH TRAINING AND INFERENCE
#python3 Trial_Setup_Utils.py EEG 10 5 DECISION_TREE inference 1
#python3 Trial_Setup_Utils.py EEG 5 KNN 
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Decision_tree training $LOWER_RANGE $UPPER_RANGE
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Random_Forest_depth training $LOWER_RANGE $UPPER_RANGE
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Random_Forest_estimators training $LOWER_RANGE $UPPER_RANGE
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Adaboost_estimators training $LOWER_RANGE $UPPER_RANGE



#python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE $MODEL inference $LOWER_RANGE $UPPER_RANGE $MODEL_NUMBER

python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Decision_tree inference $LOWER_RANGE $UPPER_RANGE 4
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Random_Forest_depth inference $LOWER_RANGE $UPPER_RANGE 5
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Random_Forest_estimators inference $LOWER_RANGE $UPPER_RANGE 6
python3 Trial_Setup_Utils.py $DATASET $NUMBER_HOLDOUT $HOLDOUT_SIZE Adaboost_estimators inference $LOWER_RANGE $UPPER_RANGE 7