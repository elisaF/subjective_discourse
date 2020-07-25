The following baselines are mostly a clone of the [Hedwig repo](https://github.com/castorini/hedwig) from the Data Systems Group at the University of Waterloo.

## Models

+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for document classification [(Adhikari et al., NAACL 2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Adhikari_etal_NAACL2019.pdf)
+ [XML-CNN](models/xml_cnn/): CNNs for extreme multi-label text classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [HAN](models/han/): Hierarchical Attention Networks [(Zichao et al., NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [Char-CNN](models/char_cnn/): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
+ [Kim CNN](models/kim_cnn/): CNNs for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)

Each model directory has a `README.md` with further details.

## Additional setup

We assume you've already performed the main setup steps as documented [here](../../../README.md). The following additional steps are required to run the baselines:

Activate your previously created environment:
```bash
conda activate subjective
```

Install gensim:

```bash
pip install gensim
```

Download the NLTK stopwords:

```bash
python -m nltk.downloader all
```

Download and unpack the word vectors:

```bash
cd subjective_discourse/
git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
cd hedwig-data/embeddings/word2vec
tar -xvzf GoogleNews-vectors-negative300.tgz
```
