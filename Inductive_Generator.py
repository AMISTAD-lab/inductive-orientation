# Important
from functools import reduce 
import Data_Generator as Data_Generator
import numpy as np
import json
from tqdm import tqdm
import copy
import pickle
import os

class Inductive_Generator:
  def __init__(self, mode, clf, classes, save_path, X_train, y_train, X_fixed = None, y_fixed=None):
    """
    mode (string): "sparse", "predict proba", "simple good turing"
    clf (model from skitlearn): decision tree, etc
    classes [int]: [0,1]
    save_path (str): 
    X_train, y_train, X_fixed, y_fixed from dataset
    """
    if mode not in ["sparse", "predict proba", "simple good turing"]:
      raise Exception("Mode is not one of sparse, predict proba, or simple good turing.")
    self.mode = mode
    self.clf = clf
    self.classes = classes
    self.times_trained = 0
    self.save_path = save_path # global folder for saving models and things
    self.X_train = X_train
    self.y_train = y_train
    self.X_fixed = X_fixed
    self.y_fixed = y_fixed
    self.data_generator = Data_Generator.Data_Generator(self.X_train, self.y_train, 42, self.X_fixed, self.y_fixed)


  def train(self, subset_X, subset_y):
    clf_copy = copy.deepcopy(self.clf)
    clf_copy.fit(subset_X, subset_y)
    self.times_trained += 1
    return clf_copy

  def convertBinary2Decimal(self,binary_list):
    """'Compresses' binary list to equivalent decimal"""
    decimal_result = reduce(lambda a,b: 2*a+b, binary_list)
    return decimal_result

  def get_simplex(self, trained_model, X_test):
    """
    trained_model: the classifier trained on the dataset.
    X_test: this is the holdoutset obtained from the dataset.
    """
    if self.mode == "sparse":
      predicted_labels = trained_model.predict(X_test)
      predicted_labels = predicted_labels.astype(int)
      return self.convertBinary2Decimal(predicted_labels)

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")
  
  def get_LDM(self, X_test, num_datasets, num_repeat, proportion_of_dataset, data_generation_method, do_replace=False):
    """
    X_test numpy array: from dataset
    num_datasets (int): number of trials
    num_repeat: 
    proportion_of_dataset (between 0 and 1): percentage of the dataset to sample for each F aka training subset.
    data_generation_method (string): "generate_subset(_plus_fixed)" how to subsample.
    """
    if data_generation_method == "generate_subset":
      dg_method = self.data_generator.generate_subset
    elif data_generation_method == "generate_subset_plus_fixed":
      dg_method = self.data_generator.generate_subset_plus_fixed
    else:
      raise Exception("Only generate_subset and generate_subset_plus_fixed have been implemented")

    if self.mode == "sparse":
      self.PD_length = len(self.classes)**len(X_test)
      num_entries = int(proportion_of_dataset * len(self.X_train)) # size of training subset

      def generateLDMHelper(dataset_idx):
        subset_X, subset_y = dg_method(num_entries, do_replace=do_replace) # getting training subset
        def generatePf(repeat_idx):
          clf_copy = self.train(subset_X, subset_y) #training on a subset to get a model
          Pf = self.get_simplex(clf_copy, X_test) # note that Pf is just a number
          path = os.path.join(self.save_path, f"dataset_{dataset_idx}_repeat_{repeat_idx}.pkl")
          self.save_model(path, clf_copy)
          return Pf
        all_Pf = list(map(lambda x: generatePf(x), range(num_repeat))) 
                  # to reduce stochasticity by doing number runs on the same data subset
        
        return all_Pf
      print("generating LDM")
      LDM = (list(map(lambda x: generateLDMHelper(x), tqdm(range(num_datasets)))))

      self.LDM = LDM
      return self.LDM

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")

  def compute_PD(self):
    if self.mode == "sparse":
      values, counts = np.unique(self.LDM, return_counts = True)
      counts = counts / np.sum(counts) # normalizes counts vector
      PD = np.zeros(self.PD_length)
      for i, index in enumerate(values):
        PD[index] = counts[i]
      self.PD = PD
      return PD

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")
  
  def save_model(self, file, trained_model):
    pickle.dump(trained_model, open(file, 'wb'))


  def save_state(self, file, model_name, dataset_name):
    """
    Saves the model, dataset, LDM, and PD vectors of the current Inductive_Generator Object.
    """
    self.model_name = model_name
    self.dataset_name = dataset_name
    State = {"model": self.model_name, "dataset": self.dataset_name, "LDM":self.LDM, "PD": self.PD}
    with open(file, "a") as logs:
      json.dump(State, fp=logs, cls=Inductive_Generator_Encoder)

class Inductive_Generator_Encoder(json.JSONEncoder):
  """
  Neccessary for serializing numpy arrays.
  """
  def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
      return obj.__dict__

class Inductive_Generator_Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
      obj["LDM"] = np.asarray(obj["LDM"])
      obj["PD"] = np.asarray(obj["PD"])
      return obj