# Important
from functools import reduce 
import Data_Generator as Data_Generator
import numpy as np
import json
from tqdm import tqdm
import copy
import pickle
import os
import pdb

class Inductive_Generator:
  def __init__(self, mode, clf, classes, save_path, dataset_info, holdout_size, num_holdouts):
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
    self.X_train = dataset_info["X_train"]
    self.y_train = dataset_info["y_train"]
    self.X_fixed = dataset_info["X_test"] # fixed discontinued  (it was the subset that stayed the same within set of subsets)
    self.y_fixed = dataset_info["y_test"]
    self.holdout_size = holdout_size # size of each holdout set
    self.num_holdouts = num_holdouts # number of holdout sets randomly gathered from X_test
    self.data_generator = Data_Generator.Data_Generator(self.X_train, self.y_train, 42, self.X_fixed, self.y_fixed)

    self.LDMs = []
    self.PDs = []

  def train(self, subset_X, subset_y):
    """Trains classifier (again)"""
    clf_copy = copy.deepcopy(self.clf)
    clf_copy.fit(subset_X, subset_y)
    self.times_trained += 1
    return clf_copy

  def convertBinary2Decimal(self,binary_list):
    """'Compresses' binary list to equivalent decimal. Helper to get_simplex"""
    decimal_result = reduce(lambda a,b: 2*a+b, binary_list)
    return decimal_result

  def get_simplex(self, trained_model, X_test):
    """
    Returns P_f, column of LDM matrix (Labeled distribution matrix). This binary column is converted to a decimal number
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
    

  # -----------------NEW STUFF--------------------
  

  def getN_LDM_Pf(self, X_test, y_test, num_datasets, num_repeat, proportion_of_dataset, from_download = False, do_replace=True):
    """Gets self.num_holdouts for each holdout set --> gets LDM and Pf (HASN'T BEEN TESTED)"""
    holdout_sets_x, holdout_sets_y = Data_Generator.generateN_holdout_sets(self.num_holdouts, self.holdout_size,
                                                                            X_test, y_test, do_replace = True)
    #FIXME: do we need holdout_sets_y or nah
    #TODO: download holdout_sets in json

    # create list of LDMs associated with each holdout set
    for holdout_set in holdout_sets_x:
      #pdb.set_trace()
      self.LDMs.append(self.get_LDM(holdout_set, num_datasets, num_repeat, proportion_of_dataset, 
                                    "generate_subset", from_download=from_download, do_replace=do_replace))
  
    # compute associated PDs
    for LDM in self.LDMs:
      self.PDs.append(self.compute_PD_new(LDM))
    

      
  def get_LDM(self, X_test, num_datasets, num_repeat, proportion_of_dataset, data_generation_method, do_replace=True, from_download=False):
    """
    Returns LDM matrix of 
    X_test numpy array: from dataset (holdout set???)
    num_datasets (int): number of trials
    num_repeat: 
    proportion_of_dataset (between 0 and 1): percentage of the dataset to sample for each F aka training subset.
    data_generation_method (string): "generate_subset(_plus_fixed)" how to subsample.
    """

    # determine which data generation method for LDM
    if data_generation_method == "generate_subset":
      dg_method = self.data_generator.generate_subset
    elif data_generation_method == "generate_subset_plus_fixed": # discontinued - used to add fixed subset to all training subsets
      dg_method = self.data_generator.generate_subset_plus_fixed
    else:
      raise Exception("Only generate_subset and generate_subset_plus_fixed have been implemented")

    if self.mode == "sparse":
      if from_download:
        self.PD_length = len(self.classes)**len(X_test) # length of inductive orientation vector
      num_entries = int(proportion_of_dataset * len(self.X_train)) # size of training subset

      def generateLDMHelper(dataset_idx):
        """Generates LDM matrix"""
        subset_X, subset_y = dg_method(num_entries, do_replace=do_replace) # getting training subset
        
        def generatePf(repeat_idx):
          """Generates each vector (a binary vector represented by an integer) of LDM matrix"""
          path = os.path.join(self.save_path, f"dataset_{dataset_idx}_repeat_{repeat_idx}.pkl")
          if from_download:
            try:
              clf_copy = pickle.load(open(path, "rb"))
            except:
              raise Exception(f"No saved model exists in {path}")
            Pf = self.get_simplex(clf_copy, X_test) # note that Pf is just a number

          else:
            clf_copy = self.train(subset_X, subset_y) # training on a subset to get a model
            self.save_model(path, clf_copy)
            Pf = None
          
          return Pf
        all_Pf = list(map(lambda x: generatePf(x), range(num_repeat))) # makes num_repeats Pfs in 
                  # to reduce stochasticity by doing number runs on the same data subset
        
        return all_Pf
      
      print("generating LDM")
      LDM = (list(map(lambda x: generateLDMHelper(x), tqdm(range(num_datasets)))))

      #self.LDM = LDM # no longer use self.LDM (use self.LDMs instead)
      return LDM

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")

  def compute_PD(self):
    """OLD VERSION w/ only one LDM, Computes inductive orientation vector"""
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
  
  #----- NEW --------
  def compute_PD_new(self, LDM):
    """Temp new version of compute_PD based on multiple LDMs"""
    if self.mode == "sparse":
      values, counts = np.unique(LDM, return_counts = True)
      counts = counts / np.sum(counts) # normalizes counts vector
      PD = np.zeros(self.PD_length)
      for i, index in enumerate(values):
        PD[index] = counts[i]
      return PD

    elif self.mode == "predict proba":
      raise Exception("Sorry, only sparse mode is implemented")
    
    elif self.mode == "simple good turing":
      raise Exception("Sorry, only sparse mode is implemented")
  
  def save_model(self, file, trained_model):
    pickle.dump(trained_model, open(file, 'wb'))


#------- EDITED----------
  def save_state(self, file, model_name, dataset_name):
    """
    Saves the model, dataset, LDM, and PD vectors of the current Inductive_Generator Object.
    """
    self.model_name = model_name
    self.dataset_name = dataset_name
    #State = {"model": self.model_name, "dataset": self.dataset_name, "LDM":self.LDM, "PD": self.PD}
    
    State = {"model": self.model_name, "dataset": self.dataset_name, "LDMs":self.LDMs, "PDs": self.PDs} # new state, TODO: add place to save holdout sets and X_test
    with open(file, "a") as logs:
      json.dump(State, fp=logs, cls=Inductive_Generator_Encoder)

class Inductive_Generator_Encoder(json.JSONEncoder):
  """
  Creates json representation of LDM and P_d. Neccessary for serializing numpy arrays.
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