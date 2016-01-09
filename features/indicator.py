import features as feat
import utils

def extract_indicator_features(in_path,out_path):
    action_t_series=utils.read_dir_objects(in_path)
    features=[from_instance(action) for action in action_t_series]
    utils.to_labeled_file(out_path,features)

def from_instance(action):
    arrays=action.to_array()
    dims=[arr_i.shape[1] for arr_i in arrays]
    cats=[feat.category_series(arr_i) for arr_i in arrays]
    hists=[feat.category_histogram(dim_i,cat_i) for dim_i,cat_i in zip(dims,cats)]
    features=utils.concate_lists(hists)
    features+=indicator_features(cats[0])
    return features,str(action.cat)+"#"+str(action.person)

def indicator_features(cats):
    indc_extr=[af_indicator,cc_indicator,dd_indicator,cb_indicator]
    features=[extr_i(cats) for extr_i in indc_extr]
    return features

def cb_indicator(cats):
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==2 and cats[i+1]==1):
            count=1.0
    return count    

def af_indicator(cats):
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==0 and cats[i+1]==5):
            count=1.0
    return count  

def cc_indicator(cats):
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==2 and cats[i+1]==2):
            count=1.0
    return count 

def dd_indicator(cats):
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==3 and cats[i+1]==3):
            count=1.0
    return count 