# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:02:52 2020

@author: jlederman
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
#nlp = en_core_web_sm.load()

## Getting data and prepping the statuses
filepath = '~/Documents/MLG Capstone/Data/Statuses/nytimes_facebook_statuses.csv'
data = pd.read_csv(filepath, dtype={'status_message':str})

statuses_raw = data.status_message + '   ' + data.link_name


## Custom training of Named Entity Recognizer
nlp = spacy.blank('en')
#ner = nlp.create_pipe('ner')
#nlp.add_pipe(ner)

## Pulling entities for all statuses in a data frame
def apply_ner(status, nlp = nlp):
    if type(status) != str:
        status = ''
    doc = nlp(status)
    return [(X.text, X.lemma_, X.label_) for X in doc.ents if X.label_ in (
        'PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT')]

data['entities'] = statuses_raw.apply(apply_ner)
data['entities_list'] = [[ent[0] + '_' + ent[1] for ent in ents]
                         for ents in data.entities]

## One-hot entity encoding
ent_vocab = list(set([item for sublist in data.entities_list for 
                    item in sublist]))
for ent in ent_vocab:
    data[ent] = [int(ent in ents) for ents in data.entities_list]


## Pairwise correlations of reaction percentages
reax_percs = data[['num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'num_reactions']]

for col in reax_percs.columns[:-1]:
    reax_percs['perc' + col[3:]] = reax_percs[col] / reax_percs.num_reactions

reax_percs[['perc_loves', 'perc_wows', 'perc_hahas', 'perc_sads', 'perc_angrys']].corr()
# Suggests maybe loves, hahas, and sads? (Only ones with negative pairwise correlations)

def reax_rank(row, rank = 1):
    name = reax_percs.iloc[row, 6:11].sort_values(ascending = False).index[(rank - 1)]
    if reax_percs.iloc[row,][name] == 0:
        return None
    else:
        return name

reax_percs['top'] = [reax_rank(i) for i in reax_percs.index]
reax_percs['second'] = [reax_rank(i, 2) for i in reax_percs.index]

pd.crosstab(reax_percs.top, reax_percs.second)