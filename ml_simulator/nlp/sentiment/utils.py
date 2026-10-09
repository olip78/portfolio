import math
from dataclasses import dataclass
from typing import List, Generator, Tuple
from transformers import PreTrainedTokenizer
from sklearn.model_selection import cross_val_score, KFold

import torch
import numpy as np

@dataclass
class DataLoader:
    """data loader for bert 
    """
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        with open(self.path) as file:
            row_count = len(file.readlines())
        return math.ceil((row_count - 1) / self.batch_size)

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        res = [self.tokenizer.encode(x,
                                     add_special_tokens=True,
                                     max_length=self.max_length)
               for x in batch]
        return res[:self.max_length]

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        def to_lables(sent: str) -> int:
            """converts str santiment to in lables
            """
            if sent=='neutral':
                res = 0
            elif sent=='positive':
                res = 1
            else:
                res = -1
            return res
        reviews = []
        lables = []
        with open(self.path) as file:
            for line in file.readlines()[1 + i * self.batch_size:1 + (i + 1) * self.batch_size]:
                row = line.split(",", 4)
                lables.append(row[-2])
                reviews.append(row[-1][:-1])
        lables = list(map(to_lables, lables))
        return reviews, lables

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        if self.padding is not None:
            if self.padding == 'max_lenght':
                #max_length = max([len(t) for t in tokens])
                max_length = max(len(t) for t in tokens)
            else:
                max_length = self.max_length
            tokens_arr = np.zeros((self.batch_size, max_length), dtype='int')
            for i, t in enumerate(tokens):
                tokens_arr[i, :len(t)] = t
            tokens = tokens_arr.tolist()
        return tokens, labels


def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    """attention mask
    """
    mask = []
    for tokens in padded:
        mask.append([1 if t != 0 else 0 for t in tokens])
    return mask


def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    mask = attention_mask(tokens)

    # Calculate embeddings
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)

    with torch.no_grad():
        last_hidden_states = model(tokens, attention_mask=mask)

    # Embeddings for [CLS]-tokens
    features = last_hidden_states[0][:,0,:].tolist()
    return features


def evaluate(model, embeddings, labels, cv=5) -> List[float]:
    """model cross validation
    """
    scores = - cross_val_score(model, embeddings, y=labels, scoring='neg_log_loss', cv=KFold(n_splits=cv))
    return scores
