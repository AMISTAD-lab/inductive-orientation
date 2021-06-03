# inductive-orientation
Summer 2021 project analyzing bias in algorithms using inductive orientation vectors

All functions are located in the ldm_inductive.py file

For examples of how to access our functions, check the ldm_inductive_test.py

To generate an inductive orientation vector (the Pd vector) use 
  PD = ldm_inductive.computePD(model, dataset, holdout_set_percentage, num_datasets)
To generate an Labeling Distribution Matrix (LDM) use
  LDM = ldm_inductive.computeLdm(model, dataset, holdout_set_percentage, num_datasets)
 
Some functions are taken from work done by past AMISTAD Lab students as part of the ["The Labeling Distribution Matrix: A Tool for Estmiating Machine Learning Algorithm Capacity"](https://arxiv.org/abs/1912.10597#:~:text=version%2C%20v2)
Currently functions are
 - ldm_inductive.getSimplex
 - ldm_inductive.getLdm
 - ldm_inductive.computeLdm
 - ldm_inductive.computeEntropy
 - ldm_inductive.plotHeatMap
