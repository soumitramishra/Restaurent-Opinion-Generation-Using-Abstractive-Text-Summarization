import re
from math import log, floor
from collections import Counter
from nltk.corpus import stopwords
from statistics import median, mean
import readability
import string
from tqdm import tqdm

def get_data(filename):
    '''Get data from file'''
    with open(filename,encoding="utf-8",mode="r") as f:
        s =str(f.read()).lower().replace("<eos>", "").translate(str.maketrans('', '', string.punctuation))
        return s

def generate_ngrams(s, n):
    '''
    Generates the ngrams for a given text
    input: s is the sentence, n of n-gram
    output: n-grams list
    '''
    # http://www.albertauyeung.com/post/generating-ngrams-python/
    # Convert to lowercases
    s = s.lower()
    s = s.strip()
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngrams = [" ".join(ngram) for ngram in ngrams]
    ngrams = Counter(ngrams)
    if n==1:                
        ngrams = {k:v for k, v in ngrams.items() if v > 1 }
        med = median(ngrams.values())
        ngrams = {k:v for k, v in ngrams.items() if v > med }
        
    return ngrams


def get_ngram_containing(unigram,bigram):
    ''' 
    '''
    new_dict = {}
    uni = list(unigram.keys())
    for k,v in bigram.items():
        ks = k.split(' ')
        if ks[0] in uni or ks[1] in uni:
            new_dict[k] = v
    return new_dict

def ngram_tuple(ngram):
    return [(k.split(' ')[0],k.split(' ')[1]) for k,v in ngram.items()]


def generate_scored_ngrams(seed):
    '''
    '''
    new_dict = {}
    for s in tqdm(seed):
        for s2 in seed:
            if s != s2:
                if s[1] == s2[0]:
                    val = tuple(set(s+s2))
                    new_dict[val] = readability.getmeasures(val, lang='en')['readability grades']['FleschReadingEase']
    
    return new_dict

def prob(ngram):
    '''
    '''
    n = len(ngram)
    new_dict = {k:v/n for k,v in ngram.items()}
    return new_dict

def pmi(x,y):
    val = 0
    if x+' '+y in list(bigram_prob.keys()):
        val  = log((bigram_prob[x+' '+y]*bigrams[x+' '+y])/(unigram_prob[x]*unigram_prob[y]))
    return val 


def representativeness(ngram):
    
    window = floor(len(ngram)/2)
    s_rep = []
    for n,gram in enumerate(ngram):
        val = []
        for i in range((n-window), min(n+window+1,len(ngram))):
            if i >= 0 and i != n:
                val.append(pmi(gram,ngram[i]))
        s_rep.append(sum(val)/(2*window))
    
    return mean(s_rep)

data = get_data('Four Peaks Brewing_JzOp695tclcNCNMuBl7oxA.txt')
unig = generate_ngrams(data, 1)
bigrams = generate_ngrams(data,2)

uni = list(unig.keys())
seed_dict = {}
for bi,v in bigrams.items():
    b = bi.split(' ')
    if b[0] in uni and b[1] in uni:
        seed_dict[bi] = v

d = get_ngram_containing(unig,seed_dict)
bigram_tup = ngram_tuple(d)
scores = generate_scored_ngrams(bigram_tup)
unigram_prob = prob(unig)
bigram_prob = prob(seed_dict)
    
with open('scores.txt','w') as f:
    for k,v in scores.items():
        print(str(k)+' '+str(v), file = f)


rep_dict= {}
for ngram in tqdm(scores.keys()):
    rep_dict[ngram] = representativeness(ngram)
    