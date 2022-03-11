from functools import reduce 
import Data_Generator as Data_Generator
import numpy as np
class Inductive_Generator:
  def __init__(self, mode, clf, classes, X_train, y_train):
    if mode not in ["sparse", "predict proba", "simple good turing"]:
      raise Exception("Mode is not one of sparse, predict proba, or simple good turing.")
    self.mode = mode
    self.clf = clf
    self.classes = classes
    self.times_trained = 0
    self.X_train = X_train
    self.y_train = y_train
    self.data_generator = Data_Generator(self.X_train, self.y_train, 42)

  def train(self, subset_X, subset_y):
    self.clf.fit(subset_X, subset_y)
    self.times_trained += 1
  
  def b2d(binary_list):
    decimal_result = reduce(lambda a,b: 2*a+b, binary_list)
    return decimal_result

  def get_simplex(self, X_test):
    if self.mode == "sparse":
      predicted_labels = self.clf.predict(X_test)
      predicted_labels = predicted_labels.astype(int)
      return self.b2d(predicted_labels)

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")
  
  def get_LDM(self, X_test, num_datasets, num_repeat, proportion_of_dataset, data_generation, do_replace=False):
    if self.mode == "sparse":
      self.PD_length = len(self.classes)**len(X_test)
      num_entries = int(proportion_of_dataset) * len(self.X_train)

      def generateLDMHelper():
        subset_X, subset_y = self.data_generator.data_generation(num_entries, do_replace=do_replace)
        def generatePf():
          self.train(subset_X, subset_y)
          Pf = self.get_simplex(X_test)
          return Pf
        all_Pf = list(map(lambda x: generatePf(), range(num_repeat)))
        return all_Pf
      
      LDM = (list(map(lambda x: generateLDMHelper(), range(num_datasets))))

      self.LDM = LDM
      return self.LDM

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")

  def compute_PD(self):
    if self.mode == "sparse":
      values, counts = np.unique(self.LDM, return_counts = True)
      counts = counts / np.sum(counts)
      PD = np.zeros(self.PD_length)
      for i, index in enumerate(values):
        PD[index] = counts[i]
      self.PD = PD
      return PD

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")
