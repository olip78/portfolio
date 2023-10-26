from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from transformers import DistilBertModel, DistilBertTokenizer
from utils import DataLoader, attention_mask, review_embedding
from utils import evaluate
import torch

MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
bert = DistilBertModel.from_pretrained(MODEL_NAME)

loader = DataLoader('New_Query_2023_09_30.csv', tokenizer, max_length=128, padding='batch')

X, y = [], []
for tokens, labels in loader:
    features = review_embedding(tokens, bert)
    X += features
    y += labels
    
model = LogisticRegression()
evaluate(model, X[:len(y)], y, cv=5)
