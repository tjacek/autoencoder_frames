import features as feat
import utils

def extract_indicator_features(in_path,out_path,ind_id):
    action_t_series=utils.read_dir_objects(in_path)
    features=[from_instance(action,ind_id) for action in action_t_series]
    utils.to_labeled_file(out_path,features)

def from_instance(action,ind_id):
    arrays=action.to_array()
    dims=[arr_i.shape[1] for arr_i in arrays]
    cats=[feat.category_series(arr_i) for arr_i in arrays]
    hists=[feat.category_histogram(dim_i,cat_i) for dim_i,cat_i in zip(dims,cats)]
    features=utils.concate_lists(hists)
    features+=apply_indicator_features(cats[0],ind_id)
    return features,str(action.cat)+"#"+str(action.person)

def apply_indicator_features(cats,indicator_id):
    indc_extr=indicator_features[indicator_id]
    features=[extr_i(cats) for extr_i in indc_extr]
    return features

def bc_indicator(cats):
    C_index=feat.ABC.index("C")
    B_index=feat.ABC.index("B")
    size=cats.size
    count=[0.0,0.0]
    for i in range(0,size):
        if(cats[i]==1):
            count[0]=B_index
        if(cats[i]==2):
            count[1]=C_index
    return count[0]*count[1]   

def fg_indicator(cats):
    E_index=feat.ABC.index("E")
    F_index=feat.ABC.index("F")
    G_index=feat.ABC.index("G")
    size=cats.size
    count=[1.0,1.0,1.0]
    for i in range(0,size):
        if(cats[i]==4):
            count[0]=0.0
        if(cats[i]==5):
            count[1]=0.0
        if(cats[i]==6):
            count[2]=0.0
    result=count[0]*count[1]*count[2]
    return  result*bc_indicator(cats)

def cb_indicator(cats):
    C_index=feat.ABC.index("C")
    B_index=feat.ABC.index("B")
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==C_index and cats[i+1]==B_index):
            count=1.0
    return count    

def af_indicator(cats):
    A_index=feat.ABC.index("A")
    F_index=feat.ABC.index("F")
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==A_index and cats[i+1]==F_index):
            count=1.0
    return count  

def fe_indicator(cats):
    A_index=feat.ABC.index("F")
    F_index=feat.ABC.index("E")
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==A_index and cats[i+1]==F_index):
            count=1.0
    return count  

def cc_indicator(cats):
    C_index=feat.ABC.index("C")
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==C_index and cats[i+1]==C_index):
            count=1.0
    return count 

def dd_indicator(cats):
    D_index=feat.ABC.index("D")
    size=cats.size
    count=0.0
    for i in range(0,size-1):
        if(cats[i]==D_index and cats[i+1]==D_index):
            count=1.0
    return count 

indicator_features=[[],
                    [af_indicator,fe_indicator,cc_indicator,dd_indicator,cb_indicator],
                    [bc_indicator,fg_indicator]]