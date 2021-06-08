from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import new_ldm_inductive as ldm_inductive
import matplotlib.pyplot as plt
import pandas as pd

# Initialize model specific variables
#dataset = datasets.load_iris()
dataset = pd.read_csv("BankNote_Authentication.csv")
X = dataset[dataset.columns[0:4]]
y = dataset[dataset.columns[4]]
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
adaboostClassifier = AdaBoostClassifier()
# holdout_set_percentage = 0.003 #i think this is useless now rt
# num_datasets = 5
# proportion_of_dataset = 0.3
# classes = [0,1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.003) #gives us about 5 things in the holdout set
#X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.003, random_state=42)
classes=[0,1]
num_datasets=200
proportion_of_dataset=0.3
N = 10

def trial(clf):
    list_of_LDM = ldm_inductive.computeNLDM(N, clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset)
    sparse_Pd_l = [ldm_inductive.computePD(LDM) for LDM in list_of_LDM]
    predict_proba_Pd_l = ldm_inductive.computeNPD(N, clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset, sparse=False)
    SGT_Pd_l = [ldm_inductive.simpleGoodTuring(LDM) for LDM in list_of_LDM]
    #sparse_LDM = ldm_inductive.getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset)
    #sparse_Pd = ldm_inductive.computePD(sparse_LDM)
    sparse_Pd = sparse_Pd_l[0]
    print("sparse PD:", sparse_Pd)
    #predict_proba_LDM = ldm_inductive.getLDM(clf, X_train, X_test, y_train, classes=classes, num_datasets=num_datasets, proportion_of_dataset=proportion_of_dataset,sparse=False)
    #predict_proba_Pd = ldm_inductive.computePD(predict_proba_LDM)
    predict_proba_Pd = predict_proba_Pd_l[0]
    print("predict_proba PD:", predict_proba_Pd)
    #SGT_Pd = ldm_inductive.simpleGoodTuring(sparse_LDM)
    SGT_Pd = SGT_Pd_l[0]
    print("SGT PD:", SGT_Pd)
    print("angle sparse, SGT: ", ldm_inductive.computeAngle(sparse_Pd, SGT_Pd))
    print("angle sparse, predict_proba: ", ldm_inductive.computeAngle(sparse_Pd, predict_proba_Pd))
    print("angle predict_proba, SGT: ", ldm_inductive.computeAngle(predict_proba_Pd, SGT_Pd))
    #-----------------------------------------------
    print("difference sparse, SGT: ", (sparse_Pd - SGT_Pd))
    print("difference sparse, predict_proba: ", (sparse_Pd - predict_proba_Pd))
    print("difference predict_proba, SGT: ", (predict_proba_Pd - SGT_Pd))

trial(model)
