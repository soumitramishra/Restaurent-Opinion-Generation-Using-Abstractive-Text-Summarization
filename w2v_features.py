# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:53:22 2019

@author: sumedh
"""
import numpy as np
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

def create_corpus(data):
    ''' Create Corpus of words '''
    corp = [sent_tokenize(j) for i in data for j in data[i]]
    sents = [j for i in corp for j in i]
    words = [word_tokenize(s) for s in sents]
    return words

def word2vec_model(corpus='', emb_size=300, window=5, workers=5, 
                   batch_words=50, epochs=30, min_count=1, loc='', google = False):
    ''' train word2vec model for embeddings and save model '''
    if google:
        print('Loading Google stuff....')
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        #vocab = model.index2word
        print('Google New Vectors Loaded')
    else:
        model= Word2Vec(corpus, size=emb_size, sg=0, compute_loss=True, window=window,
                        min_count=min_count, workers=workers, batch_words=batch_words)
        model.train(corpus,total_examples=len(corpus),epochs=epochs)
        if loc:
            model.save("%sWord2Vec"%loc)
        print('Custom Embedding Trained...')
    
    return model


def one_hot_encode(corpus, vocab, max_vocab = None):
    ''' one hot encode the corpus '''
    
    annotated = {}
    all_words = [j for i in corpus for j in i]
    word_count = Counter(all_words)
    if not max_vocab:
        max_vocab = int(len(set(all_words)))
    
    word_count = word_count.most_common(max_vocab)
    all_words = [w[0] for w in word_count]
    eos_tag = ['<','eos','>']
    all_words = [w for w in all_words if w not in eos_tag]
    all_words.append(''.join(eos_tag))
    all_words = [w for w in all_words if w in vocab]
    #all_words.extend(vocab)
    lb_enc = LabelEncoder()
    int_encoded = lb_enc.fit_transform(list(set(all_words)))
    
    del all_words
    
    ohe = OneHotEncoder(sparse = False)
    int_encoded = int_encoded.reshape(len(int_encoded),1)
    ohe_encoded = ohe.fit_transform(int_encoded)
    
    for k in range(len(ohe_encoded)):
        inverted = lb_enc.inverse_transform([np.argmax(ohe_encoded[k,:])])[0]
        annotated[inverted] = ohe_encoded[k]
    
   
    
    return lb_enc, ohe_encoded, annotated


def w2v_matrix(model, data, google = False):
    ''' create a word2vec matrix '''    
    dat ={'review':[], 'summaries':[]}
    
    if google:
        vocab = model.vocab
    else:
        vocab = model.wv.vocab
        
    for i in tqdm(range(len(data['review']))):
        r = [model[w] for w in word_tokenize(data['review'][i]) if w in vocab]
        dat['review'].append(r)
        
        s = [model[w] for w in word_tokenize(data['summaries'][i]) if w in vocab]
        dat['summaries'].append(s)
    
    print('Word Vector created...')
    return dat

def cut_seq(data, review_len, summary_len):
    ''' limit sequence length of the summaries and reviews '''
    dat = {'review':[],'summaries':[]}
    for k in range(len(data['review'])):
        if len(data['review'][k]) < review_len or len(data['summaries'][k]) < summary_len:
             pass
        else:
            dat["review"].append(data["review"][k][:review_len])
            dat["summaries"].append(data["summaries"][k][:summary_len])
    return dat

def addones(seq):
    ''' add padding '''
    return np.insert(seq, [0], [[0],], axis = 0)
