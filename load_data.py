# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:26:38 2019

@author: sumed
"""

import os
import re


path = os.getcwd()
reviews = path+'\\summaries handwritten\\'

datasets={'reviews':reviews}
data_categories=["training","validation","test"]
data={'review':[],'summaries':[]}


def load_data(dire,category):
    """dataname refers to either training, test or validation"""
    for dirs,subdr, files in os.walk(dire+category):
        filenames=files
    return filenames

def parsetext(dire,category,filename):
    with open("%s\\%s"%(dire+category,filename),'r',encoding="Latin-1") as readin:
        print("file read successfully")
        text=readin.read()
    return text.lower()

def cleantext(text):
    text=re.sub('\d','#',text)    
    return text.split('summary: ')

filenames=load_data(datasets["summaries"],data_categories[0])

for f in filenames:
    dat = cleantext(parsetext(f))
    data['review'].append(dat[0])
    data['summaries'].append(dat[1])
    
