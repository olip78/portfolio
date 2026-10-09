import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk

import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F

from .KNRM import KNRM
from .utils import collate_fn


class Model:
    """the ranking model
    """
    def __init__(self, glove_vectors_path: str,
                 glue_qqp_dir: str, 
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [10, 5],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.015, 
                 change_train_loader_ep: int = 25,
                 n_training: int = 4500
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.min_token_occurancies = min_token_occurancies
        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep
        self.n_training = n_training
        

    def initialize_training(self):
        """training initialization
        """
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)
        
        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)
        
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        
        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
                                           self.idx_to_text_mapping_dev,
                                           vocab=self.vocab, 
                                           oov_val=self.vocab['OOV'],
                                           preproc_func=self.simple_preproc)
        
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 
                                                          batch_size=self.dataloader_bs, 
                                                          num_workers=0,
                                                          collate_fn=collate_fn,
                                                          shuffle=False)
        
    def prediction_data(self, query, documents, vocab, 
                        max_len: int = 30):
        """prepares data for prediction
        """
        oov_val = vocab['OOV']
        pad_val = vocab['PAD']
        query_idx = []
        counter = 0
        for t in self.simple_preproc(query):
            if t in vocab.keys():
                query_idx.append(vocab[t])
                counter += 1
            if counter == max_len:
                break
        
        documents_idx = []
        doc_max_length = -1
        for d in documents:
            counter = 0
            document_idx = []
            for t in self.simple_preproc(d):
                if t in vocab.keys():
                    document_idx.append(vocab[t])
                    counter += 1
                if counter == max_len:
                    break
            
            if len(document_idx) > doc_max_length:
                doc_max_length = len(document_idx)
            documents_idx.append(document_idx)
        
        documents_idx_equal_lenght = []
        for document_idx in documents_idx:
            document_idx += [pad_val]*(doc_max_length - len(document_idx))
            documents_idx_equal_lenght.append(document_idx)
            
        res = {'query': torch.LongTensor([query_idx]), 
               'document': torch.LongTensor(documents_idx_equal_lenght)}
        return res

        
    def build_model_with_pretrained_weights(self, mlp_state_dict, emb_matrix):
        self.knrm = KNRM(emb_matrix, freeze_embeddings=True,
                         out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        self.knrm.mlp.load_state_dict(mlp_state_dict)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        """prepares dataset
        """
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    
    # ---- string preprocessing & tocenization ----
    
    def handle_punctuation(self, inp_str: str) -> str:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        inp_str = inp_str.translate(translator)
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.handle_punctuation(inp_str).lower().strip()
        return nltk.word_tokenize(inp_str)
    
    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        return {k: v for k, v in vocab.items() if v >= min_occurancies}

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        texts = pd.DataFrame([])
        for df in list_of_df:
            df = df.loc[:, ['text_left', 'text_right']].stack()
            df.drop_duplicates(inplace=True)
            texts = pd.concat([texts, df])
        texts.drop_duplicates(inplace=True)
        texts = list(texts.loc[:, [0]].values)
        tokens = []
        while len(texts) > 0:
            tokens += self.simple_preproc(texts.pop(0)[0])   
        res = self._filter_rare_words(Counter(tokens), min_occurancies) 
        return list(res.keys())

    # --- glove_embedding ----

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        with open(file_path) as f:
            glove_dict = {}
            for l in f:
                l = l.split(' ')
                glove_dict[l[0]] = np.array(list(map(float, l[1:])))
        return glove_dict       

    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:

        def apply_glove_dict(token):
            n = len(glove_dict['the'])
            if token in glove_dict.keys():
                return glove_dict[token]
            elif token == 'PAD':
                return np.zeros(n)
            else:    
                return 2*rand_uni_bound*np.random.rand(n) - rand_uni_bound
            
        np.random.seed(random_seed)    
        glove_dict = self._read_glove_embeddings(file_path)
        inner_keys = ['PAD', 'OOV'] + inner_keys
        unk_words = list(set(inner_keys) - set(glove_dict.keys()))
        vocab = {v: k for k, v in enumerate(inner_keys)}
        emb_matrix = [np.array(apply_glove_dict(token)) for token in inner_keys] 
        emb_matrix = np.vstack(emb_matrix)
        
        return emb_matrix, vocab, unk_words

    # ----- KNRM model -----
    
    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    
    # ----- data -----
    
    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int
                                   ) -> List[List[Union[str, float]]]:
        """selection of triplet documents for training
        """
        np.random.seed(seed)
        
        # subsample selection
        n = inp_df.shape[0] // self.n_training
        query_group_begin = inp_df.id_left.unique()
        k = int(len(query_group_begin)/n)
        query_selection = np.random.choice(query_group_begin, k)
        
        # pairs
        inp_df_cuted = inp_df[inp_df['id_left'].isin(query_selection)] 
        pairs = np.array(self.create_val_pairs(inp_df_cuted, seed=seed))
        query_array = pairs[:, 0]

        # triplets
        res = []     
        for k, p in enumerate(pairs):
            i, j1, t1 = p
            ll = list(np.where(query_array==i)[0])
            ll.remove(k)
            l = np.random.choice(ll, 1)[0]

            _, j2, t2 = pairs[l]

            if int(t1) == 2 and int(t1) > int(t2):
                target = 1
            else:
                target = 0
            res.append((i, j1, j2, target))
        return res

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        """builds validation pairs
        """
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs
    
    #. ------ df to dict------
    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true_: np.array, ys_pred_: np.array, ndcg_top_k: int = 10) -> float:
        """ndcg metric
        """
        def dcg(ys_true, ys_pred):
            _, argsort = torch.sort(ys_pred, descending=True, dim=0)
            argsort = argsort[:ndcg_top_k]
            ys_true_sorted = ys_true[argsort]
            ret = 0
            for i, l in enumerate(ys_true_sorted, 1):
                ret += (2 ** l - 1) / math.log2(1 + i)
            return ret

        ys_true = torch.from_numpy(ys_true_)
        ys_pred = torch.from_numpy(ys_pred_)

        ideal_dcg = dcg(ys_true, ys_true)
        pred_dcg = dcg(ys_true, ys_pred)
        return (pred_dcg / ideal_dcg).item()
    
    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        """validation. calculates ndcg metrics
        """
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)
    
    def train(self, n_epochs: int):
        """training loop implementation
        """
        def eval_() -> float:
            with torch.no_grad():
                self.model.eval()
                ndcg = self.valid(self.model, self.val_dataloader)
                return ndcg

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        self.losses = []

        for epoch_i in range(n_epochs):
            if epoch_i % self.change_train_loader_ep == 0:
                self.training_triplets_for_ndcg = self.sample_data_for_train_iter(self.glue_train_df, 
                                                                             seed=self.random_seed)
                self.training_dataset = TrainTripletsDataset(self.training_triplets_for_ndcg,
                                                             self.idx_to_text_mapping_train,
                                                             vocab=self.vocab, 
                                                             oov_val=self.vocab['OOV'],
                                                             preproc_func=self.simple_preproc)

                self.training_dataloader = torch.utils.data.DataLoader(self.training_dataset, 
                                                                       batch_size=self.dataloader_bs, 
                                                                       num_workers=0,
                                                                       collate_fn=collate_fn, 
                                                                       shuffle=False)
            
            for batch in self.training_dataloader:
                self.model.train()
                batch_X1, batch_X2, batch_lables = batch
                
                optimizer.zero_grad()
                batch_pred = self.model(batch_X1, batch_X2)
                batch_loss = criterion(batch_pred, batch_lables)
                # Backpropagation
                batch_loss.backward(retain_graph=True) 
                optimizer.step()
                
                self.losses.append(batch_loss.item())
            if epoch_i % 5 == 0:
                ndcg = eval_()
                print(epoch_i, ndcg)
                
        ndcg = eval_()
        print(epoch_i, ndcg)


class RankingDataset(torch.utils.data.Dataset):
    """base dataset class for training triplets / val. pairs
    """
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        res = []
        for token in tokenized_text:
            if token in self.vocab:
                res.append(self.vocab[token])
            else:
                res.append(self.oov_val)
        return res

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:      
        text = self.idx_to_text_mapping[idx]
        tokens = self.preproc_func(text)
        tokens = self._tokenized_text_to_index(tokens)
        return tokens[:self.max_len]

    def __getitem__(self, idx: int):
        pass

class TrainTripletsDataset(RankingDataset):
    """triplets for training
    """
    def __getitem__(self, idx):
        triplet = self.index_pairs_or_triplets[idx] # (id_left, id_right_1, id_right_2, target)
        output_dict_1 = {}
        output_dict_1['query'] = self._convert_text_idx_to_token_idxs(triplet[0])
        output_dict_1['document'] = self._convert_text_idx_to_token_idxs(triplet[1])
        output_dict_2 = {}
        output_dict_2['query'] = self._convert_text_idx_to_token_idxs(triplet[0])
        output_dict_2['document'] = self._convert_text_idx_to_token_idxs(triplet[2])
        
        return output_dict_1, output_dict_2, triplet[3]
        

class ValPairsDataset(RankingDataset):
    """pairs for validaton
    """
    def __getitem__(self, idx):
        pair = self.index_pairs_or_triplets[idx] # (id_left, id_right, target)
        output_dict = {}
        output_dict['query'] = self._convert_text_idx_to_token_idxs(pair[0])
        output_dict['document'] = self._convert_text_idx_to_token_idxs(pair[1])
        return output_dict, pair[2]

