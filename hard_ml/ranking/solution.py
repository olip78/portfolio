from flask import Flask, request, jsonify, abort

import json
import os

import torch

from lib.ranking import Model
from lib.utils import languagedetection
from lib.index import Selection

def get_simmilar_questions(query, model):
    """model inference
    """
    ids, texts = selection.search(query)
    model_input = model.prediction_data(query, texts, vocab)
    res = model.knrm.predict(model_input)
    _, I = torch.sort(res.flatten(), descending=True)
    I = I[:10] 
    texts_out = [texts[i] for i in I]
    ids_out = [ids[i] for i in I]
    return texts_out, ids_out

app = Flask(__name__)

EMB_PATH_KNRM = os.environ["EMB_PATH_KNRM"]
MLP_PATH = os.environ["MLP_PATH"]
EMB_PATH_GLOVE = os.environ["EMB_PATH_GLOVE"]
VOCAB_PATH = os.environ["VOCAB_PATH"]

exchange = {'status': 'model is loading', 'index_size': 0, 'index': 'no index yet!'}
results = {'lang_check': [], 'suggestions': []}

# model loading
model = Model(EMB_PATH_GLOVE,
              min_token_occurancies = 1,
              knrm_kernel_num = 21,
              knrm_out_mlp = [10, 5],
              train_lr = 0.01,
              )

state_dict = torch.load(EMB_PATH_KNRM) 
emb_matrix = state_dict['weight']
mlp_state_dict = torch.load(MLP_PATH)
model.build_model_with_pretrained_weights(mlp_state_dict, emb_matrix)

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

exchange['status'] = 'ok'


@app.route('/badrequest400')
def bad_request():
    return abort(400)

@app.route('/ping', methods=['GET'])
def ping():
    global exchange
    return jsonify(exchange)

@app.route('/update_index', methods=['GET', 'POST'])
def update_index():
    """updates the FAISS index on get documents
    """
    global selection, exchange

    if request.method == 'POST': 
        exchange = {'status': 'index is being updated', 'index_size': 0}
        content = json.loads(request.json)
        n = len(content['documents'])
        selection = Selection(EMB_PATH_GLOVE, 
                              agg_method='idf', 
                              metric='l2',   
                              n_output=250, 
                              nlist = 25)
        selection.init_index(content['documents'])
        exchange = {'status': 'ok', 'index_size': n, 'index': 'ready'}
    return jsonify(exchange)

@app.route('/query', methods=['GET', 'POST'])
def query_batch():
    """handles batch of queries. implements three step selection logic
    """
    global results, exchange

    if request.method == 'POST': 
        if  exchange['index'] == 'ready':
            results = {'lang_check': [], 'suggestions': []}
            content = json.loads(request.json)
            queries = content['queries']
    
            for query in queries:
                language = languagedetection(query)
                results['lang_check'].append(language)
                if language:
                    texts_out, ids_out = get_simmilar_questions(query, model)
                    results['suggestions'].append(list(zip(ids_out, texts_out)))
                else:    
                    results['suggestions'].append(None)
            exchange['status'] = 'ok'
        else:
            exchange['status'] = 'FAISS is not initialized!'
            return jsonify(exchange)
    
    return jsonify(results)
