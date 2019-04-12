import re
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
from statistics import median
import readability
import string
def get_data(filename):
    with open(filename,encoding="utf-8",mode="r") as f:
        s =str(f.read()).lower().replace("<eos>", "").translate(str.maketrans('', '', string.punctuation))
        return s

def get_highest_freq_unigrams(data):
    words = data.split()
    stopwordset = set(stopwords.words('english'))
    words = [word for word in words if word not in stopwordset]
    unigrams = list(ngrams(words, 1))
    unigrams_dict = dict(Counter(unigrams))

    lis = []
    i = 1
    unig_set = set(unigrams_dict.values())
    unig_set.remove(1)
    med = median(unig_set)
    unigrams_list = [(k, unigrams_dict[k]) for k in sorted(unigrams_dict, key=unigrams_dict.get, reverse=True)]
    for k,v in unigrams_list:
        lis.append(k[0])
        i += 1
        if i>=med:
            break
    return lis

data = get_data('data/#1Brothers Pizza_4z-QW_f3RwCAxHB5fd58TA.txt')
unig = get_highest_freq_unigrams(data)


def get_all_ngrams(dat,n):
    words = dat.split()
    stopwordset = set(stopwords.words('english'))
    words = [word for word in words if word not in stopwordset]
    ng = list(ngrams(words, n))
    return ng


bigrams = get_all_ngrams(data,2)


def get_bigrams_start_with(dat,uni):
    bigrms = get_all_ngrams(dat,2)
    ret = []
    for bigram in bigrms:
        if bigram[0] in uni:
            ret.append(bigram)
    return ret


init_bigrams = get_bigrams_start_with(data,unig)


def generate_candidates(bgram,unigs):
    string = ' '.join(bgram)
    print(string)
    read_score = readability.getmeasures(bgram, lang='en')['readability grades']['FleschReadingEase']
    print(read_score)
candidates = []
for bigram in init_bigrams:
    generate_candidates(bigram,unig)

print(init_bigrams)


