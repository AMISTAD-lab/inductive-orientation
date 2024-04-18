import subprocess
from constants import *
from tqdm import tqdm

for model_name in tqdm(list(MODEL_TO_TRIAL_NUMS.keys())):
    training_args = ["python3", "benchmark.py", "--model_name", model_name]
    print(f"Generating bds for {model_name}")
    try:
        subprocess.run(training_args)
        print(f"Completed bds for {model_name}")
    except Exception as error:
        print(f"An error occurred while {model_name}:\n {error}")