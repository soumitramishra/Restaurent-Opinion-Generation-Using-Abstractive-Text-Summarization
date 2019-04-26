# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:04:48 2019

@author: sumedh
"""

import os
import shutil
import random
from tqdm import tqdm

def get_empty_files(rootdir):
    files = os.listdir(rootdir)
    new_files = [f for f in tqdm(files) if f[-4:] == '.txt']
    zfiles = [f for f in tqdm(files) if f not in new_files]
    
    print(str(len(zfiles))+' Empty files founds...')
    print('Returing Non-Empty files...')
    return new_files
    
def make_datasets(**kwargs):
    split = int(len(kwargs['files']) * kwargs['split'])
    idx = random.sample(range(len(kwargs['files'])),split)
    
    train = [n for i,n in tqdm(enumerate(kwargs['files'])) if i in idx]
    test = [n for i,n in tqdm(enumerate(kwargs['files'])) if i not in idx]
    
    for f in tqdm(train):
        shutil.move(kwargs['data_path']+'/'+f, kwargs['train_path']+'/'+f)
    print(str(len(train))+' Train set files relocated...')
    for f in tqdm(test):
        shutil.move(kwargs['data_path']+'/'+f, kwargs['test_path']+'/'+f)
    print(str(len(test))+' Test set files relocated...')
    return print('Train and Test Partitions Created')

path = os.getcwd()
data_path = path+'/data/reviews/'
train_path = path+'/data/training'
test_path = path+'/data/test'

files = get_empty_files(data_path)
make_datasets(files = files,
              train_path = train_path,
              test_path = test_path,
              data_path = data_path,
              split = 0.8)

