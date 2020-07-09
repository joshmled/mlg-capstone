# Load packages and dependencies
import pandas as pd
import os
import pickle
import statistics

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import spacy
from spacy.matcher import PhraseMatcher

os.chdir(os.path.expanduser('~/Documents/mlg_capstone'))

feat_dir = "data/1_features/"

# Set variables
status_dir = "data/0_statuses/"
status_files = pd.Series(os.listdir(status_dir))
status_files_full = pd.Series(status_dir + f for f in status_files)
sources = status_files.str.split("_").apply(lambda x: x[0])

def remove_ents(spacy_doc, nlp):
    # Build SpaCy PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)
    # Add entities to matcher
    id = 1
    for ent in spacy_doc.ents:
        matcher.add(str(id), None, nlp(ent.string))
        id += 1
    # Run matcher and collect tokens to remove
    matches = matcher(spacy_doc)
    tokens_for_removal = []
    for match in matches:
        tokens_for_removal += list(range(match[1], match[2]))
    # Return doc with entities removed
    spacy_doc = nlp(' '.join(
        spacy_doc[i].text for i in range(len(spacy_doc)) if 
                       i not in tokens_for_removal))
    return(spacy_doc)
        
    
def get_sentiment(spacy_doc, nlp, rmv_ents = True):
    if rmv_ents:
        spacy_doc = remove_ents(spacy_doc, nlp)
    # Get sentence strings
    sentences = [str(sent) for sent in spacy_doc.sents]
    if len(sentences) == 0:
        return 0
    # Use VADER analyzer to get sentence-wise sentiments
    analyzer = SentimentIntensityAnalyzer()
    sent_sents = [analyzer.polarity_scores(s)['compound'] for s in sentences]
    # Get mean of sentiments for one aggregate score
    return statistics.mean(sent_sents)

def get_sent_feats(source, nlp = spacy.load('en_core_web_sm')):
    print("Getting sentiments for", source)
    file_spc = open(feat_dir + source + '.spc', 'rb')
    spacy_docs = pickle.load(file_spc)
    file_spc.close()
    return(spacy_docs.apply(get_sentiment, nlp = nlp))


for i in status_files.index:
    sent_feats = get_sent_feats(sources[i])
    file_sents = open(feat_dir + sources[i] + '.sentiment', 'wb')
    pickle.dump(sent_feats, file_sents)
    file_sents.close()
