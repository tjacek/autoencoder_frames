import utils
import numpy as np

ABC="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def extract_features(in_path,out_path):
    action_t_series=utils.read_dir_objects(in_path)
    features=[category_count(action) for action in action_t_series]
    utils.to_labeled_file(out_path,features)

def category_count(action):
    arrays=action.to_array()
    raw_count=[category_histogram(arr_i) for arr_i in arrays]
    return np.concatenate(raw_count),action.cat

def category_histogram(arr):
    dim=arr.shape[1]
    hist=np.zeros(dim)
    hist=hist.astype(float)
    cat_series=category_series(arr)
    print(cats_to_seq(cat_series))
    for cat_i in cat_series:    
        hist[cat_i]+=1.0
    hist/=float(arr.shape[0])
    return list(hist)

def category_series(arr):
    size=arr.shape[0]
    cat_series=np.zeros(size)
    for i in range(size):
        cat_series[i]=np.argmax(arr[i])
    return cat_series

def make_sequences(in_path,out_path):
    action_t_series=utils.read_dir_objects(in_path)
    sequences=[]
    for action in action_t_series:
        arr=action.to_array()[0]
        cat_series=category_series(arr)
        seq=cats_to_seq(cat_series)
        seq+="#"+str(action.cat)+"\n"
        sequences.append(seq)
    str_seq=utils.array_to_txt(sequences)
    utils.save_string(out_path,str_seq)

def cats_to_seq(cat_series):
    seq=""
    for cat_i in cat_series:
        cat_i=int(cat_i)
        seq+=ABC[cat_i]
    return seq