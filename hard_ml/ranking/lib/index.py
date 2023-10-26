import string
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import faiss

from .utils import Vectorizer

        
class Selection(torch.nn.Module):
    """FAISS kNN
    """
    def __init__(self, 
                 glove_path,
                 min_token_occurancies=1, 
                 agg_method='idf',
                 n_output=100, 
                 metric='cos', 
                 d_vect=50,
                 nlist = 25
                ):

        self.metric = metric 
        self.d_vect = d_vect
        self.nlist = nlist
        self.n_output = n_output
        
        self.vectorizer = Vectorizer(glove_path,
                                     min_token_occurancies=min_token_occurancies,
                                     agg_method=agg_method)
        
    def init_index(self, all_documents):
        """index initialization
        """
        self.vectorizer.get_all_documents(all_documents)
        training_documents = self.vectorizer.all_documents_vec
        if self.metric == 'cos':
            quantizer = faiss.IndexFlatIP(self.d_vect)
            self.index = faiss.IndexIVFFlat(quantizer,  
                                            self.d_vect,
                                            self.nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(training_documents)
        elif self.metric == 'l2':
            self.index = faiss.IndexFlatL2(self.d_vect)
            self.index.add(training_documents)
            
    def search(self, question):  
        """kNN
        """   
        xq = self.vectorizer.vectorize(question)
        xq = np.reshape(xq, (-1, 1)).T
        _, I = self.index.search(xq, self.n_output)
        texts = []
        I_out = []
        for i in I[0]:
            texts.append(self.vectorizer.all_documents[i])
            I_out.append(self.vectorizer.mapper[i])
        return I_out, texts
