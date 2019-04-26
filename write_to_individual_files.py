# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:48:50 2019

@author: sumedh
"""
import os
import pandas as pd
from concurrent.futures.thread import ThreadPoolExecutor
from nltk import sent_tokenize

path = os.getcwd()

combine_data = pd.read_csv('data/Combined_data.txt')
bis_id = list(set(combine_data['business_id']))

def write_to_file(i):
    dt = combine_data.loc[combine_data['business_id'] == i]
    dt.drop(dt.columns[0], axis = 1, inplace= True)
    text = dt['text'].values
    max_sent = []
    for t in text:
        sents = sent_tokenize(t)
        sents = [s+' <eos>' for s in sents]
        max_sent.extend(sents)
        
    text = ' '.join(max_sent)
    res_name = ''.join(set(dt['name']))+'_'+i
    
    with open (path+'\\data\\reviews\\'+res_name+'.txt','w',encoding = 'utf-8') as f:
            f.write(text)
    
    return('')


for ids in bis_id:
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(write_to_file, ids)