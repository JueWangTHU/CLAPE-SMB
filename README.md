# CLAPE-SMB: Protein-small molecules binding sites prediction

This repo holds the code of CLAPE (Contrastive Learning And Pre-trained Encoder) for protein-small molecules binding (SMB) sites prediction. 

CLAPE-SMB is a member of [CLAPE framework](https://github.com/YAndrewL/clape).

CLAPE-SMB is primarily dependent on a large-scale pre-trained protein language model [ESM-2](https://github.com/facebookresearch/esm)  implemented using [HuggingFace's Transformers](https://huggingface.co/) and [PyTorch](https://pytorch.org/). Please install the dependencies in advance. 

## Files and folders description
### 1. Raw_data
This folder contains raw data. 

The first line is the protein id (which might not be PDB ID); the second line is the protein sequence, and the third line is the data label indicating binding sites or non-binding sites.

Note: if you wanna find the original data in PDB format, please kindly refer to the following 3 papers: [PUResNet](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00547-7), [P2Rank](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0285-8), and [GraphBind](https://academic.oup.com/nar/article/49/9/e51/6134185?login=true). 

In this project, we both used standard datasets and our own datasets. 

#### Standard datasets
sc-PDB as the training set, JOINED as the validation set and COACH420 as the testing set. 

#### Our dataset: UniProtSMB
We create a large, high-quality dataset named UniprotSMB for better train and evaluate CLAPE-SMB. We collect all reviewed proteins with 3D structures from the UniProt database, mark their small molecules-binding sites, utilize UCLUST to group proteins exhibiting sequence similarity exceeding 50% into the same cluster, and use MAFFT to merge their small molecules-binding sites into a single protein sequence. Next, we removed all sequences exceeding a length of 1,024, finally resulting in UniProtSMB, which consisted of 8,013 proteins. 

### 2. Models
This fold contains trained weights files. The original ckpt file was too large, so we split it and you can use follow commands to merge them: 

```
cd ./Models/COACH420
cat COACH420_* > coach420.ckpt
```

## Codes description
The training and implementation of this project are based on PyTorch and PyTorch-lightning 1.9.5, the higher version might not be compatible. 

### 1. data.py
Implementation of datasets class.

### 2. model.py
Implementation of backbone models such as MLP, CNN, RNN. 

### 3. losses.py
Implementation of several customized loss functions such as class-balanced focal loss, TCL, and CrossEntropy, and the reference was described in the code comments. 

### 4. triplet.py
Implementation of the total training setting, please kindly refer to the document of PyTorch-lightning for details. (1.9.5)

### 5. pre.py
This python file generates protein sequence embeddings, ESM-2 and ProtBert were provided.

### 6. count.py
This python file counts the number of positive and negative samples for training steps. 

### 7. inference.py
#### Usage:

We provide the Python script for predicting small-molecules binding sites of given protein sequences in txt format. Here we provide a sample file, please use following commands:

```
python inference.py --input example.txt --output CLAPE_SMB_result.txt
```

Some parameters are described as follows:

| Parameters  | Descriptions                                                 |
| ----------- | ------------------------------------------------------------ |
| --threshold | Specify the threshold for identifying the binding site, the value needs to be between 0 and 1, default: 0.5. |
| --input     | The path of the input file in txt format.                  |
| --output    | The path of the output file, the first and the second line are the same as the input file, and the third line is the prediction result. |

Please contact wangjue21@mails.tsinghua.edu.cn for questions. 
