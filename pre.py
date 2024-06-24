# -*- coding: utf-8 -*-
# @Time         : 2024/6/24 12:00
# @Author       : Jue Wang and Yufan Liu
# @Description  : Generate protein sequence embeddings by ESM-2 or ProtBert

from transformers import BertModel, BertTokenizer
import re
import pickle
import torch
import esm
from tqdm import tqdm

## protbert
#tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir='./cache_model/')  # specify the cache model path
#pretrain_model = BertModel.from_pretrained("Rostlab/prot_bert" , cache_dir='./cache_model/')

# utilization

#def get_protein_features(seq):
#    sequence_Example = ' '.join(seq)
#    sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
#    encoded_input = tokenizer(sequence_Example, return_tensors='pt')
#    last_hidden = pretrain_model(**encoded_input).last_hidden_state.squeeze(0)[1:-1,:]
#    return last_hidden.detach()


#def data_pkl_generator(root, saver):
#    data = open(root, 'r').readlines()
#    data_dict = {}
#    for i in tqdm(range(len(data))):
#        if data[i].startswith('>'):
#            seq = data[i+1].strip()
#            label = data[i+2].strip()
#            data_dict[data[i].strip()[1:]] = (get_protein_features(seq), label)
#    pickle.dump(data_dict, open(saver, 'wb'))

Name = 'example'
#files = ['./Raw_data/' + Name + '.txt']
#savers = ['./Dataset/SM/' + Name + '.pkl']
#for file, saver in zip(files, savers):
#    data_pkl_generator(file, saver)


# esm-v2
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

batch_converter = alphabet.get_batch_converter()
model.eval()

#raws = pickle.load(open('./Dataset/SM/' + Name + '.pkl', 'rb'))  # generated protbert file

# data training data
data_dict = {}
data = open("./Raw_data/" + Name + ".txt", 'r').readlines()
for i in tqdm(range(len(data))):
    if data[i].startswith('>'):
        pid = data[i].strip()[1:]
        seq = [(pid, data[i+1].strip())]
        batch_labels, batch_strs, batch_tokens = batch_converter(seq)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        data_dict[pid] = (token_representations.squeeze(0)[1:-1, :], data[i+2].strip())

pickle.dump(data_dict, open("./Dataset/SM/esm_" + Name + ".pkl", 'wb'))
