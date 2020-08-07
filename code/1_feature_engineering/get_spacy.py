# Load packages and dependencies
import pandas as pd
import os
import pickle

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm


os.chdir(os.path.expanduser('~/Documents/mlg_capstone'))

# Set variables
status_dir = "data/0_statuses/"
status_files = pd.Series(os.listdir(status_dir))
status_files_full = pd.Series(status_dir + f for f in status_files)
sources = status_files.str.split("_").apply(lambda x: x[0])

feat_dir = "data/1_features/"

# Set SpaCy pipeline
#my_nlp = 

def spacy_status(status, nlp):
    if type(status) != str:
        status = ''
    return nlp(status)
    
def build_spacy_docs(data, source, nlp = spacy.load('en_core_web_sm')):
    print('Building SpaCy pipeline docs for statuses:', source)
    statuses_raw = data.status_message + '   ' + data.link_name
    return statuses_raw.apply(spacy_status, nlp = nlp)

def get_ner(spacy_doc):
    return [(X.text, X.lemma_, X.label_) for X in spacy_doc.ents if X.label_ in (
        'PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT')]
    
def get_ents(source, spacy_overwrite = False, 
             nlp = spacy.load('en_core_web_sm'), data = None):
    if spacy_overwrite:
        spc_docs = build_spacy_docs(data, source, nlp)
    else:
        file_spc_docs = open(feat_dir + source + '.spacy', 'rb')
        spc_docs = pickle.load(file_spc_docs)
        file_spc_docs.close()
    print('Getting entities for statuses:', source)
    ents = pd.DataFrame()
    ents['ner_raw'] = spc_docs.apply(get_ner)
    ents['ner_list'] = [[ent[0] + '_' + ent[1] for ent in ents]
                         for ents in ents.ner_raw]
    ## One-hot entity encoding
    ent_vocab = list(set([item for sublist in ents.ner_list for 
                    item in sublist]))
    for ent in ent_vocab:
        ents[ent] = [int(ent in ent_list) for ent_list in ents.ner_list]
    return ents
        
for i in status_files.index:
    status_data = pd.read_csv(status_files_full[i])
    spc_docs = build_spacy_docs(status_data, sources[i])
    file_spc_docs = open(feat_dir + sources[i] + '.spacy', 'wb')
    pickle.dump(spc_docs, file_spc_docs)
    file_spc_docs.close()    
    ents = get_ents(sources[i], data = status_data)
    file_ents = open(feat_dir + sources[i] + '.ents', 'wb')
    pickle.dump(ents, file_ents)
    file_ents.close()
