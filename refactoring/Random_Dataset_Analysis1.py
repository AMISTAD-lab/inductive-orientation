from tkinter import E
import Algorithmic_Analysis
import Inductive_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
from Fully_Synthetic import generate_fully_synethic
import json
import os


# Getting Data
X, y = generate_fully_synethic(4, 2000, 100, 2)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5)

target4 = Algorithmic_Analysis.getTarget(X_test, y_test, 4, [0,1])
target3 = Algorithmic_Analysis.getTarget(X_test, y_test, 3, [0,1])


def singleAnalysis(file, target = None):
    print("Current file: ", file)
    with open(file) as logs:
        saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
        if type(target) != type(None):
            bias = Algorithmic_Analysis.computeAlgorithmicBias(target, saved_state["PD"])
            print(bias)
        expressivity = Algorithmic_Analysis.computeEntropy(saved_state["LDM"])
        print(expressivity)
        capacity = Algorithmic_Analysis.computeAlgorithmicCapacity(saved_state["LDM"], saved_state["PD"])
        print(capacity)

def runAnalysis(file, target=None):
    if file[-4:] == "json":
        singleAnalysis(file, target)
    else:
        logs = os.listdir(file)
        logs = [os.path.join(file, log) for log in logs]
        for log_file in logs:
            singleAnalysis(log_file, target)
    



# with open("./logs/trial2_Adaboost50.json") as logs:
#     saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
#     Adaboost50_Bias = Algorithmic_Analysis.computeAlgorithmicBias(target4, saved_state["PD"])
#     print(Adaboost50_Bias)
#     Adaboost50_Entropy = Algorithmic_Analysis.computeEntropy(saved_state["LDM"])
#     print(Adaboost50_Entropy)
#     Adaboost50_Alg_Cap = Algorithmic_Analysis.computeAlgorithmicCapacity(saved_state["LDM"], saved_state["PD"])
#     print(Adaboost50_Alg_Cap)

# print("NaiveBayes")
# with open("./logs/trial2_NaiveBayesClassifier.json") as logs:
#     saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
#     Adaboost50_Bias = Algorithmic_Analysis.computeAlgorithmicBias(target4, saved_state["PD"])
#     print(Adaboost50_Bias)
#     Adaboost50_Entropy = Algorithmic_Analysis.computeEntropy(saved_state["LDM"])
#     print(Adaboost50_Entropy)
#     Adaboost50_Alg_Cap = Algorithmic_Analysis.computeAlgorithmicCapacity(saved_state["LDM"], saved_state["PD"])
#     print(Adaboost50_Alg_Cap)

# print("RandomForest")
# with open("./logs/trial2_RandomForest18.json") as logs:
#     saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
#     Adaboost50_Bias = Algorithmic_Analysis.computeAlgorithmicBias(target4, saved_state["PD"])
#     print(Adaboost50_Bias)
#     Adaboost50_Entropy = Algorithmic_Analysis.computeEntropy(saved_state["LDM"])
#     print(Adaboost50_Entropy)
#     Adaboost50_Alg_Cap = Algorithmic_Analysis.computeAlgorithmicCapacity(saved_state["LDM"], saved_state["PD"])
#     print(Adaboost50_Alg_Cap)

# with open("./logs/trial2_Adaboost50.json") as logs:
#     saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
#     Adaboost50_Entropy = Algorithmic_Analysis.computeEntropyLDM(saved_state["LDM"])
#     print(Adaboost50_Entropy)
