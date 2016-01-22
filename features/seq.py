import numpy as np
import utils
from sets import Set
from sklearn import metrics

def parse_seq(filename):
    seqs=utils.read_file(filename)
    X=[seq_i.split("#")[0] for seq_i in seqs]
    X=[count(x_i) for x_i in X]
	#X=[to_ngram(x_i) for x_i in X]
    y=[seq_i.split("#")[1] for seq_i in seqs]
    y=[int(y_i) for y_i in y]
    n_cats,cat_names=find_n_cats(X)
    np_hist=create_histogram(X,n_cats,cat_names)
    indicators=[get_indicator(i,np_hist) for i in range(0,n_cats)] 
    cats=[get_category(i,y) for i in range(0,n_cats)] 
    entropy_matrix=np.zeros((len(cats),len(indicators)))
    for i,cat_i in enumerate(cats):
        for j,indic_j in enumerate(indicators):
            entropy_matrix[i][j]=metrics.mutual_info_score(cat_i,indic_j)
    print(entropy_matrix)

def create_histogram(X,n_cats,cat_names):
    np_hist=np.zeros((len(X),n_cats))
    for j,x_i in enumerate(X):
        for i,cat_name_i in enumerate(cat_names):
            np_hist[j][i]=x_i.get(cat_name_i,0)
    return np_hist

def get_indicator(i,hist):
	return hist[:,i]>0.0

def get_category(i,y):
    return np.array(y)==i

def count(seq):
    hist={}
    for c_i in seq:
        cat_count=hist.get(c_i,0)
        hist[c_i]=cat_count+1
    return hist

def find_n_cats(hist):
    all_cats=Set()
    all_keys=[hist_i.keys() for hist_i in hist]
    for keys_i in all_keys:
    	for key_i in keys_i:
    		all_cats.add(key_i)
    return len(all_cats),list(all_cats)
    
def to_ngram(seq):
    ngrams=[]
    for i in range(0,len(seq)-1):
        ngrams.append(seq[i]+seq[i+1])
    return ngrams 