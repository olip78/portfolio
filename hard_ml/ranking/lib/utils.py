from langdetect import detect

import string
from collections import Counter
from typing import Dict, List, Union

import nltk
import torch
import numpy as np
import math
import pandas as pd

from itertools import chain

class Vectorizer():
    """GLOVE vectorizer
    string preprocessing & vectorization
    """
    def __init__(self, glove_vectors_path, min_token_occurancies, agg_method):
        self.glove_dict = self._read_glove_embeddings(glove_vectors_path)
        self.min_token_occurancies = min_token_occurancies
        self.agg_method = agg_method

    # ---- string preprocessing & tocenization ----    
    def handle_punctuation(self, inp_str: str) -> str:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        inp_str = inp_str.translate(translator)
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.handle_punctuation(inp_str).lower().strip()
        return nltk.word_tokenize(inp_str)
    
    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        with open(file_path) as f:
            glove_dict = {}
            for l in f:
                l = l.split(' ')
                glove_dict[l[0]] = np.array(list(map(float, l[1:])))
        return glove_dict     

    def _tokenize(self, string_: str) -> np.array:
        tokens = self.simple_preproc(string_)     
        return tokens
    
    def _vectorize(self, tokens: List[str]) -> np.array:
        """glove vectorisation with mean or idf weighting
        """
        if self.agg_method  == 'mean':
            v = np.zeros(self.glove_dict['the'].shape)
            n = 0
            for t in tokens:
                if t in self.glove_dict.keys():
                    v += self.glove_dict[t]
                    n += 1
            v = v/n     
        elif self.agg_method  == 'idf':
            vv = []
            tt = []
            for t in tokens:
                if t in self.glove_dict.keys():
                    vv.append(self.glove_dict[t])
                    tt.append(t)
            idf = self.idf[tt].values
            idf = idf/idf.sum()
            vv = np.array(vv)  
            v = np.dot(idf, vv)    
        return v.astype(np.float32)
    
    def vectorize(self, string_: str) -> np.array:
        tokens = self._tokenize(string_)  
        v = self._vectorize(tokens)
        return v
    
    def get_all_documents(self, all_documents: Dict[str, str]):
        # tokenisation of documents
        self.all_documents = []
        self.all_documents_tokens = []
        self.mapper = {}
        counter = 0
        for k, text in all_documents.items():
            tokens = self._tokenize(text)
            if len(tokens) > 1:
                self.all_documents_tokens.append(tokens)
                self.all_documents.append(text) 
                self.mapper[counter] = k
                counter += 1
                
        # idf 
        freq = Counter(list(chain(*self.all_documents_tokens)))
        freq = {k: [v] for k, v in freq.items()}
        df = pd.DataFrame.from_dict(freq, orient='columns')
        self.idf = df.T.loc[:, 0]
        n = self.idf.sum()
        self.idf = self.idf.map(lambda x: math.log(n/x))
        # all documents weighted vectorization
        self.all_documents_vec = []
        for tokens in self.all_documents_tokens:
            if len(tokens) > 1:
                self.all_documents_vec.append(self._vectorize(tokens))
        self.all_documents_vec = np.array(self.all_documents_vec).astype(np.float32)  


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    """assembles a batches of several training examples
    https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem 
        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels

def languagedetection(query):
    """language detection. olny english can be processed
    """
    language = detect(query)
    if language == 'en':
        return True
    else:
        return False
