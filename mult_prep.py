# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:38:56 2019

@author: sumedh
"""

import json
import pandas as pd
from tqdm import tqdm
import csv
from time import sleep
from concurrent.futures.thread import ThreadPoolExecutor

######## Data Munging #########
def prep_json(**kwargs):
    '''
    Preprocesses the yelp json file using multiple workers    
    '''
    if 'bis_id' in kwargs.keys():        
        with ThreadPoolExecutor(max_workers=12) as executor:
            f =  [executor.submit(kwargs['func'], 
                              line = line,
                              bis_id = kwargs['bis_id'], 
                              file = kwargs['file']) for line in tqdm(open(kwargs['file'], encoding = 'utf-8'))]
    else:
        futures_list = [kwargs['func'](line = line) for line in tqdm(open(kwargs['file'], encoding = 'utf-8'))]
        with open(kwargs['file'].split('.')[0]+'.txt','w', newline = '', encoding = 'utf-8') as f:
            for futures in tqdm(futures_list):
                wr = csv.writer(f, quoting = csv.QUOTE_ALL)
                wr.writerow(futures)
    return(print('Done'))


def preprocess_business(**kwargs):
    '''
    This function preprocesses the input jsons and extracts the required attributes
    '''
    ob = json.loads(kwargs['line'])
    df_l = []
    for k, v in ob.items():
        if k in ['business_id', 'name', 'review_count','categories','stars']:
            df_l.append(v)
    return(df_l)


def preprocess_review(**kwargs):
    ob = json.loads(kwargs['line'])
    df_l = []
    if ob['business_id'] in kwargs['bis_id']:
        for k, v in ob.items():
            if k in ['business_id', 'stars', 'text','useful','funny','cool']:
                    df_l.append(v)
    sleep(0.01)
    with open(kwargs['file'].split('.')[0]+'.txt','a', newline = '', encoding = 'utf-8') as f:
            wr = csv.writer(f, quoting = csv.QUOTE_ALL)
            wr.writerow(df_l)
#########################################

prep_json(file = 'data/yelp_academic_dataset_business.json', func = preprocess_business)
data = pd.read_csv('data/yelp_academic_dataset_business.txt', names = ['business_id', 'name', 'review_count','categories','stars'], sep = ',')

####### Get the list business id's that needs to be extracted ############
new_rest = data[data['categories'].str.contains("Restaurants", na = False)]
bis_id = list(set(new_rest['business_id']))
prep_json(file = 'data/yelp_academic_dataset_review.json', func = preprocess_review, bis_id = bis_id)

####### Get some summary statsitics of the review ###########
new_data = pd.read_csv('data/yelp_academic_dataset_review.txt',names =['business_id', 'stars','useful','funny','cool', 'text'],sep=',')
#stats = new_data.groupby('business_id').agg('count')
#stats.to_csv('data/summary_stats.csv', encoding = 'utf-8')
####### Combine the reviews and the business ids and save ###########
result = pd.merge(new_data,data[['business_id', 'name', 'categories']],on='business_id')
result.to_csv('data/Combined_data.txt',encoding = 'utf-8')
