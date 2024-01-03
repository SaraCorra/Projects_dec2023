# -*- coding: utf-8 -*-
"""
A module for reading data from famous datasets.

@author: Andrea Belli <abelli@expert.ai>
"""

import os
import re
import urllib
import pandas as pd
from math import nan


CONLL_URL_ROOT = "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/"


def open_read_from_url(url):
    """
    Take in input an url to a .txt file and return the list of its raws
    """
    print(f"Read file from {url}")
    file = urllib.request.urlopen(url)
    lines = []
    for line in file:
        lines.append(line.decode("utf-8"))

    return lines


def read_raw_conll(url_root, dir_path, filename):
    """Read a file which contains a conll03 dataset"""
    lines = []
    path = os.path.join(dir_path, filename)
    full_url = url_root + filename
    if os.path.isfile(path):
        # read from file
        print(f'Reading file {path}')
        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        lines = open_read_from_url(full_url)
    return lines[2:]


def is_real_sentence(only_token, sentence):
    """Chek if a sentence is a real sentence or a document separator"""
    first_word = ""
    if only_token:
        first_word = sentence[0]
    else:
        first_word = sentence[0][0]

    if '---------------------' in first_word or first_word == '-DOCSTART-':
        return False
    else:
        return True
        
        
def load_conll_data(filename, url_root=CONLL_URL_ROOT, dir_path='', 
                    only_tokens=False):
    """
    Take an url to the raw .txt files that you can find the repo linked above,
    load data and save it into a list of tuples data structure.
    
    Those files structure data with a word in each line with word, POS, 
    syntactic tag and entity tag separated by a whitespace. Sentences are 
    separated by an empty line.
    """
    lines = read_raw_conll(url_root, dir_path, filename)
    # TODO: find a better data structure for saving data
    # TODO: do this in a more efficient way
    X = []
    Y = []
    sentence = []
    labels = []
    output_labels=set()
    for line in lines:
        if line == "\n":
            if(len(sentence) != len(labels)):
                print(f"Error: we have {len(sentence)} words but {len(labels)} labels")
            if sentence and is_real_sentence(only_tokens, sentence):
                X.append(sentence)
                Y.append(labels)
            sentence = []
            labels = []
        else:
            features = line.split()
            tag = features.pop()
            labels.append(tag)
            output_labels.add(tag)
            if only_tokens:
                sentence.append(features.pop(0))
            else:
                sentence.append(tuple(features))
    
    print(f"Read {len(X)} sentences")
    if(len(X) != len(Y)):
        print("ERROR in reading data.")
    return X, Y, output_labels


# =========================================================================== #


def _df_to_xy(df):
    """Transform anerd dataframe in X, y sets.
    
    Given the anerd dataframe, we want to obtain a X list of lists of 
    dictionaries, which contain the features of the tokens, and a Y list of 
    lists of the tag of the tokens.
    Params:
        df: dataframe;
    Returns:
        X: list of lists of dictionaries, which contain token features;
        y: list of lists of strings, which are token tags.
    """
    y = df[['sentence_idx', 'tag']].copy()
    y = y.groupby('sentence_idx').apply(lambda d: d['tag'].values.tolist()).values
    
    df.drop(columns='tag', inplace=True)
    X = df.groupby('sentence_idx').apply(lambda d: d.to_dict('records')).values
    if len(X) != len(y):
        print('ERROR: length mismatch')
    else:
        print(f'Dataset dimension: {len(y)} sentences')
    return X, y


def load_anerd_data(path, filter_level=''):
    """Load anerd data from path.
    
    Params:
        path: path of anerd csv file
        filter_level: parameter which indicate what data extract from anerd:
            default         extract features of each token and the neighbours
            sentence_only   extract only the list of tokens
            all_data        extract features of each token, the neighbours and 
                            neighbours of neighbours
    Return:
        X: list of lists of dictionaries, which contain token features
        y: list of lists of token labels
        tags: all the possible classes
    """
    dframe = pd.read_csv(path, encoding = "ISO-8859-1", error_bad_lines=False)
    
    # Create label set
    tags = set()
    for tag in set(dframe["tag"].values):
        if tag is nan or isinstance(tag, float):
            tags.add('unk')
        else:
            tags.add(tag)

    if filter_level == 'sentence_only':
        # Return only the list of tokens for each sentence
        print('Filter level:', filter_level)
        dframe = dframe[['word', 'sentence_idx', 'tag']]
        X, y = _df_to_xy(dframe)
        newX = []
        for sent in X:
            sentence = []
            for d in sent:
                sentence.append(d['word'])
            newX.append(sentence)
        X = newX
    
    elif filter_level == 'all_data':
        # Return features of token and the tokens inside the 1 degree of
        # separation
        print('Filter level:', filter_level)
        dframe.drop(columns=['Unnamed: 0', 'prev-iob', 'prev-prev-iob'], 
                    inplace=True)
        print('Features:', dframe.columns)
        X, y = _df_to_xy(dframe)
    
    else:
        print('Filter level: default')
        dframe.drop(columns= ['Unnamed: 0', 'prev-iob', 'next-next-lemma', 
                              'next-next-pos', 'next-next-shape', 
                              'next-next-word', 'prev-prev-iob', 
                              'prev-prev-lemma','prev-prev-pos', 
                              'prev-prev-shape', 'prev-prev-word'], inplace=True)
        print('Features:', dframe.columns)
        X, y = _df_to_xy(dframe)
    
    print('Data read successfully!')
    return X, y, tags


# =========================================================================== #


def load_wikiner(path, token_only=False):
    """Load WikiNER dataset.
    
    Params:
        path: path to txt file if WikiNER dataset;
        token_only: if True return only the list of token, if false return
                    also pos tag for each token.
    Return:
        sentences: list of sentences, each sentences is a list of token
        tags: list of list of token tags
        output_labels: set of all the labels in the dataset
    """
    raw_sents = []
    with open(path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            if line != '\n':
                raw_sents.append(line)
    
    # Split tokens
    for sent_idx in range(len(raw_sents)):
        raw_sents[sent_idx] = raw_sents[sent_idx].split()
    
    # Extract features and separate them from tags
    sentences = []
    tags = []
    output_labels = set()
    for raw_sent in raw_sents:
        sent = []
        tag = []
        for word in raw_sent:
            features = word.split('|')
            ent = features.pop()
            tag.append(ent)
            output_labels.add(ent)
            if token_only:
                sent.append(features.pop(0))
            else:
                sent.append(tuple(features))
        sentences.append(sent)
        tags.append(tag)
    print(f'Read {len(sentences)} sentences.')
    return sentences, tags, output_labels


def _get_digits(text):
    """Preprocess numbers in tokens accordingly to itWac word embedding."""
    try:
        val = int(text)
    except:
        text = re.sub('\d', '@Dg', text)
        return text
    
    if val >= 0 and val < 2100:
        return str(val)
    else:
        return "DIGLEN_" + str(len(str(val)))


def _normalize_text(word):
    """Preprocess word in order to match with the itWac embedding vocabulary"""
    if "http" in word or ("." in word and "/" in word):
        word = str("___URL___")
        return word
    if len(word) > 26:
        return "__LONG-LONG__"
    new_word = _get_digits(word)
    if new_word != word:
        word = new_word
    if word[0].isupper():
        word = word.capitalize()
    else:
        word = word.lower()
    return word


def itwac_preprocess_data(sentences):
    """Preprocess text in order to match with the itWac embedding vocabulary"""
    new_sentences = []
    for sentence in sentences:
        new_sent = list()
        for word in sentence:
            new_sent.append(_normalize_text(word))
        new_sentences.append(new_sent)
    return new_sentences