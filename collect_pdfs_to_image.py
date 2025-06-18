import constants
from pdf2image import convert_from_path
import numpy as np
import cv2
import pdb
import os

def collect_metrics(model_name, metric_name):
    trials = constants.MODEL_TO_TRIAL_NUMS[model_name]
    all_images_of_pdf = []
    for dataset in trials:
        trial_num = trials[dataset][0] # only get the first trial on each dataset
        trial_metric_path = os.path.join(constants.RESULTS_FOLDER, "analysis", f"trial{trial_num}", f"{dataset}_{metric_name}.pdf")
        
        page_one = np.array(convert_from_path(trial_metric_path)[0])  # Convert PDF to List of PIL Images
        all_images_pdf.append(page_one)
    pdb.set_trace()
    #image_of_pdf = np.concatenate(tuple(convert_from_path('/path/to/pdf/source.pdf')), axis=0)
# get the trials all for one algorithm, check in constants file



# open them, create an image, append to a list

# go through the list and put them all into a big image
if __name__ == "__main__":
    model_name = constants.ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value
    metric_name = constants.MetricNames.EXP_CAP_SEM.value
    collect_metrics(model_name, metric_name)
