# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:26:38 2019

@author: sumedh
"""

import os
import re
from tqdm import tqdm

def load_data(dire,category):
    """dataname refers to either training, test or validation"""
    for dirs,subdr, files in os.walk(dire+category):
        filenames=files
    return filenames

def parsetext(dire,category,filename):
    with open("%s\\%s"%(dire+category,filename),'r',encoding="Latin-1") as readin:
       text=readin.read()
    return text.lower()


def cleantext(text):
    text = re.sub('\d','#',text)
    text = ''.join(text.splitlines())
    text = ' '.join(text.split())
    return text.split('summary:')


def make_data_dict(filenames, x,y):
    data={'review':[],'summaries':[]}
    for f in tqdm(filenames):
        dat = cleantext(parsetext(x['reviews'],y[0], '%s' %f))
        data['review'].append(dat[0])
        data['summaries'].append(dat[1])
    print('All files read successfully...')
    return data 

    
