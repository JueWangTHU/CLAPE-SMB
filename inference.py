# -*- coding: utf-8 -*-
# @Time         : 2024/3/23 10:36
# @Author       : Jue Wang and Yufan Liu
# @Description  : predicting process

import torch
import argparse
import esm
from triplet import TripletClassificationModel
from tqdm import tqdm

# load instructions
parse = argparse.ArgumentParser()
parse.add_argument('--threshold', '-t', type=float, help='Threshold of classification score', default=0.5)
parse.add_argument('--input', '-i', help='Input protein sequences in txt format', required=True)
parse.add_argument('--output', '-o', help='Output file path, default CLAPE_SMB_result.txt',
                   default='CLAPE_SMB_result.txt')

args = parse.parse_args()

# parameter judge
if args.threshold > 1 or args.threshold < 0:
    raise ValueError("Threshold is out of range.")

# input sequences
input_file = open(args.input, 'r').readlines()
seq_dict = {}
for line in input_file:
    if line.startswith(">"):
        name = line.strip()[1:]
        seq_dict[name] = ''
    else:
        seq_dict[name] += line.strip()
seq_ids = list(seq_dict.keys())
seqs = list(seq_dict.values())

# feature generation
print("=====Loading pre-trained protein language model=====")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
print("Done!")


def get_protein_features(seq):
    batch_labels, batch_strs, batch_tokens = batch_converter(seq)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    return token_representations.squeeze(0)[1:-1, :]


# generate sequence feature
features = []
print("=====Generating protein sequence feature=====")
for k, v in tqdm(seq_dict.items()):
    features.append(get_protein_features([(k, v)]).unsqueeze(0))
print("Done!")

# load backbone MLP model
print("=====Loading classification model=====")
predictor = TripletClassificationModel.load_from_checkpoint("./Models/coach420/epoch=17-step=12708.ckpt").full_model
predictor.eval()
print("Done!")

# prediction process
results = []
print(f"=====Predicting Small molecules-binding sites=====")
for f in tqdm(features):
    out = predictor(f)[0].squeeze(0).detach().numpy()[:, 1]
    score = ''.join([str(1) if x > args.threshold else str(0) for x in out])
    results.append(score)
print("Done!")

print(f"=====Writing result files into {args.output}=====")
with open(args.output, 'w') as f:
    for i in range(len(seq_ids)):
        f.write(seq_ids[i] + '\n')
        f.write(seqs[i] + '\n')
        f.write(results[i] + '\n')
print(f"Congrats! All process done! Your result file is saved as {args.output}")
