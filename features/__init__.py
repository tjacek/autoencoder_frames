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
    print(arr.shape)
    hist=np.zeros(arr.shape[1])
    hist=hist.astype(float)
    for i in range(arr.shape[0]):
        cat=np.argmax(arr[i])
        hist[cat]+=1.0
    hist/=float(arr.shape[0])
    return list(hist)