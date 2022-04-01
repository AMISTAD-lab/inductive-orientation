import Algorithmic_Analysis
import Inductive_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
from Fully_Synthetic import generate_fully_synethic
import json


# Getting Data
X, y = generate_fully_synethic(4, 2000, 100, 2)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5)

target4 = Algorithmic_Analysis.getTarget(X_test, y_test, 4, [0,1])
target3 = Algorithmic_Analysis.getTarget(X_test, y_test, 3, [0,1])

with open("./logs/trial2_Adaboost50.json") as logs:
    saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
    Adaboost50_Bias = Algorithmic_Analysis.computeAlgorithmicBias(target4, saved_state["PD"])
    Adaboost50_Entropy = Algorithmic_Analysis.entropy(saved_state["LDM"])
    Adaboost50_Alg_Cap = Algorithmic_Analysis.calculateAlgorithmicCapacity(saved_state["LDM"], saved_state["PD"])
