import utils
import numpy as np

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