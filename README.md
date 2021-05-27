# Lexicon Enhanced Chinese Sequence Labeling Using BERT Adapter

Code and checkpoints for the ACL2021 paper "Lexicon Enhanced Chinese Sequence Labelling Using BERT Adapter"


Checkpoints will be released soon. Please be patient.

Arxiv link of the paper: https://arxiv.org/abs/2105.07148

# Requirement

* Python 3.7.0
* Transformer 3.4.0
* Numpy 1.18.5
* Packaging 17.1
* skicit-learn 0.23.2
* torch 1.16.0+cu92
* tqdm 4.50.2
* multiprocess 0.70.10
* tensorflow 2.3.1
* tensorboardX 2.1

# Input Format
CoNLL format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line.

```cpp
美   B-LOC  
国   E-LOC  
的   O  
华   B-PER  
莱   I-PER  
士   E-PER  

我   O  
跟   O  
他   O  
谈   O  
笑   O  
风   O  
生   O   
```

# Pretrained BERT and Embedding

* Chinese BERT: 
* Chinese word embedding: 

# Run
* 1. Convert .char.bmes file to .json file
`python3 to_json.py`

* 2.run the shell
`sh run_ner.sh`


If you want to load my checkpoints, we should make some revise to your transformers.


# Cite


