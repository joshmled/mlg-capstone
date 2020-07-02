# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:24:39 2020

@author: jlederman
"""
import json
import os
import pandas as pd

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from spacy.pipeline import EntityRuler

nlp = en_core_web_sm.load()

data_path = "C:/Users/jlederman/Documents/MLG Capstone/Data/"

type_map = {'Organisation': 'ORG', 'Person': 'PERSON', 
            'Nationality': 'NORP'}

# BBC and Wiki from re3d dataset
with open(os.path.join(data_path, "NER_Training/bbc_entities.json"), "r") as f:
    bbc_tr_raw = [json.loads(line) for line in f.readlines()]
    bbc_tr = [{'label':type_map[d['type']], 'pattern':d['value']} 
              for d in bbc_tr_raw if (d['confidence'] > .95) & 
                (d['type'] in ('Organisation', 'Person', 'Nationality'))]
    
with open(os.path.join(data_path, "NER_Training/wiki_entities.json"), "r") as f:
    wiki_tr_raw = [json.loads(line) for line in f.readlines()]
    wiki_tr = [{'label': type_map[d['type']], 
                'pattern': d['value']} for d in wiki_tr_raw if (
                    d['confidence'] > .95) & 
                (d['type'] in ('Organisation', 'Person', 'Nationality'))]


def consolidate_row(df, row_num):
    df.token[row_num] = df.token[row_num] + ' ' + df.token[row_num + 1]
    return df.drop(row_num + 1).reset_index(drop = True)

def consolidate(df, last_b = 2):
    full_df = df[0:1]
    for index, row in df.iterrows():
        print('index', index, 'DF shape', df.shape)
        if row.prefix == 'B':
            df2 = df
            if (index < (len(df) - last_b)):
                while df2.prefix[index + 1] == 'I':
                    df2 = consolidate_row(df2, index)
            else:
                for i in range(last_b - 1):
                    df2 = consolidate_row(df2, index)
            full_df = full_df.append(df2[index:index + 1])
    return full_df[1:]
                
# WNUT Emerging Dataset
with open(os.path.join(
        data_path, "NER_Training/wnut_emerging/emerging.test.annotated"), 
        "r", errors = 'replace') as f:
    emerging_test_tr_raw = f.readlines()
    emerging_test_tr = pd.DataFrame(
        [s.replace('\n', '').split('\t') for s in emerging_test_tr_raw], 
        columns = ['token', 'label'])
    emerging_test_tr = emerging_test_tr[emerging_test_tr.label != 'O']
    emerging_test_tr = emerging_test_tr[emerging_test_tr.label.notnull()].reset_index(drop = True)
    emerging_test_tr['prefix'] = emerging_test_tr.label.str.slice(0, 1)
    emerging_test_tr.label = emerging_test_tr.label.str.slice(2)
    emerging_test_tr = consolidate(emerging_test_tr)

with open(os.path.join(
        data_path, "NER_Training/wnut_emerging/emerging.dev.conll"), 
        "r", errors = 'replace') as f:
    emerging_dev_tr_raw = f.readlines()
    emerging_dev_tr = pd.DataFrame(
        [s.replace('\n', '').split('\t') for s in emerging_dev_tr_raw], 
        columns = ['token', 'label'])
    emerging_dev_tr = emerging_dev_tr[emerging_dev_tr.label != 'O']
    emerging_dev_tr = emerging_dev_tr[emerging_dev_tr.label.notnull()].reset_index(drop = True)
    emerging_dev_tr['prefix'] = emerging_dev_tr.label.str.slice(0, 1)
    emerging_dev_tr.label = emerging_dev_tr.label.str.slice(2)
    emerging_dev_tr = consolidate(emerging_dev_tr)

with open(os.path.join(
        data_path, "NER_Training/wnut_emerging/wnut17train.conll"), 
        "r", errors = 'replace') as f:
    wnut_tr_raw = f.readlines()
    wnut_tr = pd.DataFrame(
        [s.replace('\n', '').split('\t') for s in wnut_tr_raw], 
        columns = ['token', 'label'])
    wnut_tr = wnut_tr[wnut_tr.label != 'O']
    wnut_tr = wnut_tr[wnut_tr.label.notnull()].reset_index(drop = True)
    wnut_tr['prefix'] = wnut_tr.label.str.slice(0, 1)
    wnut_tr.label = wnut_tr.label.str.slice(2)
    wnut_tr = consolidate(wnut_tr, last_b = 1)


ruler = EntityRuler(nlp, overwrite_ents = True)
ruler.add_patterns(bbc_tr + wiki_tr)
nlp.add_pipe(ruler, after = "parser")
