# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:38:56 2019

@author: sumedh
"""

import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures.thread import ThreadPoolExecutor
import csv

######## Data Munging #########
def prep_json(file, t = 0):
    '''
    Preprocesses the yelp json file using multiple workers    
    the arguement t = 0 processes the business json and t = 1 processes the review
    '''
    with ThreadPoolExecutor(max_workers=12) as executor:
             futures_list = [executor.submit(preprocess, line, t) for line in tqdm(open(file, encoding = 'utf-8'))]
             if t == 0:
                 with open('processed_business.txt','w', newline = '', encoding = 'utf-8') as f:
                     for futures in tqdm(futures_list):
                         res = futures.result()
                         wr = csv.writer(f, quoting = csv.QUOTE_ALL)
                         wr.writerow(res)
             else:
                 with open('processed_review.txt','w', newline = '', encoding = 'utf-8') as f:
                     for futures in tqdm(futures_list):
                         res = futures.result()
                         if len(res) == 3:
                             wr = csv.writer(f, quoting = csv.QUOTE_ALL)
                             wr.writerow(res)        
    return(print('Done'))


def preprocess(line, t = 0):
    '''
    This function preprocesses the input jsons and extracts the required attributes
    t = 0 proccesses the business json and t = 1 processes the reviews json
    '''
    ob = json.loads(line)
    df_l = []
        
    if t == 0:
        for k, v in ob.items():
            if k in ['business_id', 'name', 'review_count','categories']:
                df_l.append(v)
    else:
        if ob['business_id'] in bis_id:
            for k, v in ob.items():
                if k in ['business_id', 'stars', 'text']:
                    df_l.append(v)
    return(df_l)


prep_json('yelp_academic_dataset_business.json')
data = pd.read_csv('processed_business.txt', names = ['business_id', 'name', 'review_count','categories'], sep = ',')

####### Get the list business id's that needs to be extracted ############
new_rest = data[data['categories'].str.contains("Restaurants", na = False)]
bis_id = list(set(new_rest['business_id']))

prep_json('yelp_academic_dataset_review.json',1)

####### Get some summary statsitics of the review ###########
new_data = pd.read_csv('processed_review.txt',names =['business_id', 'stars', 'text'],sep=',')
stats = new_data.groupby('business_id').agg('count')
stats.to_csv('summary_stats.csv', encoding = 'utf-8')

###### Combine the reviews and the business ids and save ###########
result = pd.merge(new_data,data[['business_id', 'name', 'categories']],on='business_id')
result.to_csv('Combined_data.txt',encoding = 'utf-8')