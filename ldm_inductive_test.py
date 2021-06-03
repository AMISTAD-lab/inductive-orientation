# AMISTAD Lab - Overfitting/Underfitting Team
# Code for Labelling Distribution Matrix Tests

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import ldm_inductive
import matplotlib.pyplot as plt

# Initialize model specific variables
dataset = datasets.load_iris()
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
adaboostClassifier = AdaBoostClassifier()
holdout_set_percentage = 0.03
num_datasets = 5

#print("LDM: ", ldm_inductive.computeLdm(model, dataset, holdout_set_percentage, num_datasets))

#LDM = ldm_inductive.computeLdm(model, dataset, holdout_set_percentage, num_datasets)
#print("LDM: ", LDM)
#print("numer of columns :", len(LDM[0]))

# print()
# PD = ldm_inductive.computePD(model, dataset, holdout_set_percentage, num_datasets)
# print ("PD: ", PD)
# print("length PD: ", len(PD))


# PD3 = ldm_inductive.computePD(model3, dataset, holdout_set_percentage, num_datasets)
# PD10 = ldm_inductive.computePD(model10, dataset, holdout_set_percentage, num_datasets)
# print("Angle of alignment 1,3: ", ldm_inductive.computeAngle(PD, PD3))
# print("Angle of alignment 1,10: ", ldm_inductive.computeAngle(PD, PD10))
# print("Angle of alignment 3, 10: ", ldm_inductive.computeAngle(PD3, PD10))

# print("Entropy List: ", ldm_inductive.computeEntropy(model, dataset, holdout_set_percentage, num_datasets))

#

# list_of_KNN10_PD = ldm_inductive.computeNPD(1,model10,dataset,holdout_set_percentage,num_datasets)
# for i in range(2,25):
#     list_of_KNN10_PD.append(ldm_inductive.computePD(model10, dataset, holdout_set_percentage, num_datasets))
#     print("Variance of KNN10 after ", i," runs: ", ldm_inductive.computeVariance(list_of_KNN10_PD))
    

# run_number, variance = varianceUpToN(30, "Adaboost", adaboostClassifier, dataset, holdout_set_percentage, num_datasets)
# print(variance)

# fig = plt.figure()
# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# axes.plot(run_number, variance)
# fig.show()


# AdaboostLDM = ldm_inductive.computeSparseLdm(adaboostClassifier, dataset,holdout_set_percentage, 2, 0.3)
# ldm_inductive.plotHeatMap(AdaboostLDM)

print(ldm_inductive.computeSparseLdm(model, dataset, holdout_set_percentage, num_datasets, .3))


#Variance of KNN10 = [9.137961117571029e-06, 8.349985786171377e-06, 8.379253161284915e-06, 8.058209553702347e-06, 8.468089304752554e-06, 1.0544526062336985e-05, 1.2928675731668204e-05, 1.2495125962812751e-05, 1.2630258349416205e-05, 1.2438188323105969e-05, 1.3318601513500767e-05, 1.4063566520757905e-05, 1.3542885706658916e-05, 1.3341192530215229e-05, 1.3945064660067002e-05, 1.3743442507122094e-05, 1.3749165136106825e-05, 1.3466481600764552e-05, 1.3449799906269434e-05, 1.3904721177772961e-05, 1.3580599084391104e-05, 1.3284661520868537e-05, 1.3288266922693952e-05, 1.3027697650721075e-05, 1.2931151128001173e-05, 1.2841756199556818e-05, 1.2625051875407363e-05, 
#1.2481672682690747e-05]

#Variance of Adaboost = [3.981340951510321e-06, 4.205354679336255e-06, 4.336405761962836e-06, 4.234202880182328e-06, 4.280252382371346e-06, 4.247324157029663e-06, 4.2576758856193196e-06, 4.20964427089054e-06, 4.242533480344661e-06, 4.204611468773356e-06, 4.20962764120255e-06, 4.2040632495685345e-06, 4.225951581264813e-06, 4.244921468734921e-06, 4.191430755507383e-06, 4.187928173843736e-06, 4.184814767920495e-06, 4.201969214485349e-06, 4.139471910814603e-06, 4.159133912293592e-06, 4.1439637959648075e-06, 4.127426337742776e-06, 4.126921418023228e-06, 4.127336183300713e-06, 4.1157229205991585e-06, 4.138968186399508e-06, 4.1732880371685655e-06, 4.140856174995722e-06]

# Variance of KNN1 after 3 runs:  6.774035123372077e-06
# Variance of KNN1 after 5 runs:  6.7740351233720765e-06
# Variance of KNN1 after 3 runs, KNN3, KNN10:  7.27654940851929e-06