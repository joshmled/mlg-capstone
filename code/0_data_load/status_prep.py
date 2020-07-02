# Load packages and dependencies
import pandas as pd
import os

os.chdir(os.path.expanduser('~/Documents/mlg_capstone'))

# Set variables
status_dir = "data/0_statuses/"
status_files = pd.Series(os.listdir(status_dir))
status_files_full = pd.Series(status_dir + f for f in status_files)
sources = status_files.str.split("_").apply(lambda x: x[0])

