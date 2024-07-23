# CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_port 5001 MJ3_Finetune_extract_append_predictor_Sample_10fold-lr-huber-DDP.py \
# --device_ids 0,1,2,3 --label_type rl --epochs 1 --huber_loss --train_file 4.1_train_data_GSM3130435_egfp_unmod_1.csv \
# --prefix ESM2SISS_FS4.1.ep93.1e-2.dr5 --lr 1e-2 --dropout3 0.5 \
# --modelfile /scratch/users/yanyichu/UTR-LM/Model/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl --finetune --bos_emb --test1fold

# {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
import os
import argparse
from argparse import Namespace
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import esm
from esm.data import *
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


import numpy as np
import pandas as pd
import random
import math
from sklearn.metrics import r2_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from copy import deepcopy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

class EmbeddingLayer(nn.Module):
    def __init__(self, layers = 6, heads = 16, embed_dim = 128, alphabet = None, inp_len = 50,
                 modelfile = None, device = 'cpu'):
        
        super(EmbeddingLayer, self).__init__()
        
        self.layers = layers
        self.heads = heads
        self.embedding_size = embed_dim
        self.alphabet = alphabet
        self.inp_len = inp_len
        
        if 'SISS' in args.modelfile:
            self.esm2 = ESM2_SISS(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        elif 'SS' in args.modelfile:
            self.esm2 = ESM2_SS(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        else:
            self.esm2 = ESM2(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        if modelfile:
            self.esm2.load_state_dict(torch.load(modelfile, map_location=device), strict=False)
        else:
            raise ValueError('Please provide a pretrained model file.')

        print('Loaded model from ', modelfile)
    
    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation = True):
        
        x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)

        #print('representation: ', x["representations"][layers].shape) # (batch_size, seq_len, embed_dim)

        if args.avg_emb:
            x = x["representations"][layers][:, 1 : self.inp_len+1].mean(1)
            x_o = x.unsqueeze(2) # (batch_size, embed_dim, 1)
        elif args.bos_emb:
            x = x["representations"][layers][:, 0]
            x_o = x.unsqueeze(2) # (batch_size, embed_dim, 1)
        else:
            x_o = x["representations"][layers][:, 1 : self.inp_len+1]
            x_o = x_o.permute(0, 2, 1) # (batch_size, embed_dim, seq_len)

        return x_o


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--modelfile', type=str, default='../../Model/Pretrained/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl')
    parser.add_argument('--avg_emb', action = 'store_true') # avg pooling, if --finetune: False
    parser.add_argument('--bos_emb', action = 'store_true') #[CLS] embedding, if --finetune: False
    
    args = parser.parse_args()
    print(args)

    # Seed.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Seq Data.
    alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
    print(alphabet.tok_to_idx)
    assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
    
    data = [
        ("utr1", "AAGCTG", "AGCTA<mask>", "000001"),
        ("utr2", "AGCTAG", "AGCTA<mask>", "000001"),
        ("utr3", "AGCTAG", "AGCTA<mask>", "000001"),
    ]
    tokenized_data = [alphabet.tokenize(item[1]) for item in data]
    inp_len = max([len(item) for item in tokenized_data])
    print('inp_len:', inp_len)

    # Model. 
    layers, heads, embed_dim, batch_toks = 0, 0, 0, 0
    model_info = args.modelfile.split('/')[-1].split('_')
    for item in model_info:
        if 'layers' in item: 
            layers = int(item[0])
        elif 'heads' in item:
            heads = int(item[:-5])
        elif 'embedsize' in item:
            embed_dim = int(item[:-9])
        elif 'batchToks' in item:
            batch_toks = 4096
    print("[pretrained] layers: {}, heads: {}, embed_dim: {}, batch_toks: {}".format(layers, heads, embed_dim, batch_toks))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if torch.backends.mps.is_available():
    #     device = "mps"

    embedding_model = EmbeddingLayer(layers = layers, heads = heads, embed_dim = embed_dim, 
                                    alphabet = alphabet, inp_len = inp_len,
                                    modelfile = args.modelfile, device = device)
    embedding_model = embedding_model.to(device)
    
    # Infer.
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_masked_strs, batch_tokens, batch_masked_tokens, batch_masked_indices = batch_converter(data)
    print('tokens:', batch_tokens)
    print('batch_labels:', batch_labels)
    print('tokens:', batch_tokens.shape)
    embed = embedding_model(batch_tokens)

    print(embed.shape)