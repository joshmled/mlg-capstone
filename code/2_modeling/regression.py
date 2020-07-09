import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, KFold


# Loop through files
for file in :
    # Load in pickle files for entities, sentiment, reax, ... anything else?
    # Join entities and sentiment
    # Loop through reax columns for y vals
    for reax_col in reax.columns:  
        y = reax.reax_col
        # Train regression, with ents/sent as X, reax.reax_col as y
        # Fit regressions 
        # Save regressions        

