# Load packages and dependencies
import pandas as pd
import os
import pickle

os.chdir(os.path.expanduser('~/Documents/mlg_capstone'))

# Set variables
status_dir = "data/0_statuses/"
status_files = pd.Series(os.listdir(status_dir))
status_files_full = pd.Series(status_dir + f for f in status_files)
sources = status_files.str.split("_").apply(lambda x: x[0])
feat_dir = "data/1_features/"

# Function to turn reaction counts into percentages
## Pairwise correlations of reaction percentages
def get_reax(data):
    reax_cols = data[['num_loves', 'num_wows', 'num_hahas', 'num_sads', 
                           'num_angrys', 'num_reactions']]
    for col in reax_cols.columns[:-1]:
        reax_cols['perc' + col[3:]] = reax_cols[col] / reax_cols.num_reactions
    return reax_cols.iloc[:,5:]

# Pickle and output
for i in status_files.index:
    print('Getting reax for ' + sources[i])
    statuses = pd.read_csv(status_files_full[i])
    reax = get_reax(statuses)
    file_reax = open(feat_dir + sources[i] + '.reax', 'wb')
    pickle.dump(reax, file_reax)
    file_reax.close()