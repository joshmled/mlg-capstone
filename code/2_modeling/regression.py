import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, KFold

import os
import pickle

os.chdir(os.path.expanduser('~/Documents/mlg_capstone'))

# Set variables
status_dir = "data/0_statuses/"
status_files = pd.Series(os.listdir(status_dir))
status_files_full = pd.Series(status_dir + f for f in status_files)
sources = status_files.str.split("_").apply(lambda x: x[0])

feat_dir = "data/1_features/"
mod_dir = "data/2_models/"
pred_dir = "data/3_predictions/"

# Pick a regression algorithm function
reg = LinearRegression

# Loop through files
for source in sources:
    print("Performing regression for", source)
    file_ents = open(feat_dir + source + '.ents', 'rb')
    file_sent = open(feat_dir + source + '.sentiment', 'rb')
    file_reax = open(feat_dir + source + '.reax', 'rb')
    ents = pickle.load(file_ents)
    sent = pickle.load(file_sent)
    reax = pickle.load(file_reax)
    file_ents.close()
    file_sent.close()
    file_reax.close()
    assert(len(ents) == len(sent) == len(reax))
    # Join entities and sentiment for X
    X = ents.drop(['ner_raw', 'ner_list'], axis = 1).merge(
        sent.rename('sentiment'), left_index = True, right_index = True)
    y = reax
    # Remove rows with 0 reactions for prediction purposes
    y_zeros = y[y.num_reactions == 0].index.tolist()
    X, y = X.drop(y_zeros), y.drop(y_zeros)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
    # Keep only the entities that show up at least twice in the training set
    X_train_freq = X_train.drop('sentiment', axis = 1).sum()
    keeper_cols = X_train_freq[X_train_freq > 1].index.tolist() + ['sentiment']
    X_train, X_test = X_train[keeper_cols], X_test[keeper_cols]
    # Train regression, with ents/sent as X, reax columns as y
    model = reg(n_jobs = 3)
    model.fit(X_train, y_train)
    # Get R^2 metric, coefficients, and predictions for the test set
    train_r2 = model.score(X_train, y_train)
    coef = model.coef_
    sent_coef = coef[:,-1]
    pred_y = model.predict(X_test)
    coefs = {}
    coefs['coef'] = coef
    coefs['names'] = keeper_cols
    # Save outputs
    file_coefs = open(mod_dir + 'regression/' + source + '.coefs', 'wb')
    file_mod = open(mod_dir + 'regression/' + source + '.regress', 'wb')
    file_preds = open(pred_dir + source + '.preds', 'wb')
    pickle.dump(coefs, file_coefs)
    pickle.dump(model, file_mod)
    pickle.dump(pred_y, file_preds)
    file_coefs.close()
    file_mod.close()
    file_preds.close()
