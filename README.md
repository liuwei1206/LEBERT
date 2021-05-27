# Lexicon Enhanced Chinese Sequence Labeling Using BERT Adapter

Code and checkpoints for the ACL2021 paper "Lexicon Enhanced Chinese Sequence Labelling Using BERT Adapter"

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
* seqeval 1.2.1

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

# Chinese BERT，Chinese Word Embedding, and Checkpoints
### Chinese BERT

Chinese BERT: https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin

### Chinese word embedding: 

Word Embedding: https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz

### Checkpoints and Shells

* [Weibo NER](https://drive.google.com/file/d/1HP-Fc06dMN1jqxoRivLwtAJvQm3MG64Y/view?usp=sharing)
* [Ontonote4 NER](https://drive.google.com/file/d/1Tr_G-aK32cCfeJXd8f3mAU9reo-KHRKu/view?usp=sharing)
* [MSRA NER](https://drive.google.com/file/d/1QsTiTPovvrhQ-xxSbRh9DV45-svCcWNH/view?usp=sharing)
* [Resume NER](https://drive.google.com/file/d/1ES8uMSAq3pE8MRpiOBKYNWr0qXq9j93r/view?usp=sharing)
* [CTB5 POS](https://drive.google.com/file/d/1ES8uMSAq3pE8MRpiOBKYNWr0qXq9j93r/view?usp=sharing)
* [CTB6 POS](https://drive.google.com/file/d/1J16IbWxW1Rbx5ycDPw7JWWxzjpeFDJbN/view?usp=sharing)
* [UD1 POS](https://drive.google.com/file/d/1ic1OTCdskn7P8QDPSfyxy_o5ouPXQ06o/view?usp=sharing)
* [UD2 POS](https://drive.google.com/file/d/1F0EL5YV7tGkYqDATXXhVDUFVoj9z-oVK/view?usp=sharing)
* [CTB6 CWS](https://drive.google.com/file/d/1FahANYMK27uVwinBvY6SXubSzVAdaYqC/view?usp=sharing)
* [MSR CWS](https://drive.google.com/file/d/1EtHv3bv9bYVLbXg-YrnsGV-BiVLVVBs-/view?usp=sharing)
* [PKU CWS](https://drive.google.com/file/d/117Rb-JvQiLpSlbrWTebZW9Y4dDf-I0sR/view?usp=sharing)

# Directory Structure of data

* berts
    * bert
        * config.json
        * vocab.txt
        * pytorch_model.bin 
* dataset
* vocab
* embedding
* result
* log

# Run

* 1.Convert .char.bmes file to .json file, `python3 to_json.py`

* 2.run the shell, `sh run_ner.sh`



#### If you want to load my checkpoints, you need to make some revisions to your transformers.

My model is trained in distribution mode so it can not be directly loaded by single-GPU mode. You can follow the below steps to revise the transformers before load my checkpoints.

* Enter the source code director of Transformer, `cd source/transformers-master`
* Find the modeling_util.py, and positioned to about 995 lines
* change the code as follows:
![image](https://user-images.githubusercontent.com/34615810/119770324-9bc7f980-beee-11eb-9547-9e0e9b1c3180.png)

* Compile the revised source code and install. `python3 setup.py install`


# Cite
```
@misc{liu2021lexicon,
      title={Lexicon Enhanced Chinese Sequence Labeling Using BERT Adapter}, 
      author={Wei Liu and Xiyan Fu and Yue Zhang and Wenming Xiao},
      year={2021},
      eprint={2105.07148},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
