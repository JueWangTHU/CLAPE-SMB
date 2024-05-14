# CLAPE-SMB: Protein-small molecule binding site prediction

This repo holds the code of CLAPE (Contrastive Learning And Pre-trained Encoder) for protein-small molecule binding (SMB) site prediction. 

CLAPE-SMB is a member of [CLAPE framework](https://github.com/YAndrewL/clape).

CLAPE-SMB is primarily dependent on a large-scale pre-trained protein language model [ESM-2](https://github.com/facebookresearch/esm)  implemented using [HuggingFace's Transformers](https://huggingface.co/) and [PyTorch](https://pytorch.org/). Please install the dependencies in advance. 

## Files and folders description
### 1. Raw_data
This folder contains raw data. 

The first line is the protein id (which might not be PDB ID); the second line is the protein sequence, and the third line is the data label indicating binding sites or non-binding sites.

Note: if you wanna find the original data in PDB format, please kindly refer to the following 3 papers: [PUResNet](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00547-7), [P2Rank](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0285-8), and [GraphBind](https://academic.oup.com/nar/article/49/9/e51/6134185?login=true). 

In this project, we both used standard datasets and our own datasets. 

#### Standard datasets
sc-PDB, chen11, JOINED, and COACH420 are popular protein-small molecule binding datasets. 

#### Integrated datase: SJC
sc-PDB, JOINED, and COACH420 are integrated into a new dataset named SJC, which is taken from the first letter of each dataset name. To eliminate the impact of redundant sequences and enhance model robustness, all protein sequences from three datasets are processed using UCLUST, with a sequence similarity cutoff of 50%. Then these non-redundant sequences are further divided into training (80%), validation (10%), and test (10%) sets. 

#### Our dataset: UniProtSMB
We create a large, high-quality dataset named UniprotSMB for better train and evaluate CLAPE-SMB. First, we collected all 14,064 reviewed proteins with 3D structures and small molecule-binding sites among 248,805,733 proteins in the UniProtKB database as of April 17, 2024. Second, we conducted a length cutoff and obtained 12,804 proteins with a length not exceeding 1024. Third, we identified their binding sites, including drugs, cofactors, ATP, and other small molecules, but not including metal ions. Next, we clustered proteins with a sequence similarity cutoff of 50% using UCLUST, obtaining 4,964 clusters. Subsequently, we conducted MAFFT on all proteins within each cluster and merged all binding sites to the longest sequence, resulting in the UniProtSMB dataset containing 4,964 proteins. At last, we divided UniProtSMB to training set (3,972 proteins), validation set (496 proteins) and test set (496 proteins). 

### 2. Models
This fold contains trained weights files. The original ckpt file was too large, so we split it and you can use follow commands to merge them: 

```
cd ./Models/SJC
cat SJC_* > SJC.ckpt
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

We provide the Python script for predicting small-molecule binding site of given protein sequences in txt format. Here we provide a sample file, please use following commands:

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
