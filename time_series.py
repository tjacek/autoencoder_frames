import utils
import numpy as np
import pandas as pd
import deep.composite as comp
import data
import matplotlib.pyplot as plt
import re

def to_action_frame(extr,actions):
    td=[action_time_series(extr,act) for act in actions]
    time_series=[td_i[0] for td_i in td]
    category=[td_i[1] for td_i in td]
    person=[td_i[2] for td_i in td]
    return pd.DataFrame({'td':time_series,'cat':category,'person':person})

def splited_dataset(action_frame,out_path):
    odd=action_frame[action_frame['person']==0]
    even=action_frame[action_frame['person']==1]
    action_features(odd,out_path+"_train")
    action_features(even,out_path+"_test")

def action_time_series(extr,action):
    print(action)
    feat=[extr.get_features(fr[0],fr[1],fr[2]) for fr in action.frames]
    feat=np.array(feat)
    name=str(action)
    category=extract_info(name,0)
    person=extract_info(name,1) % 2
    return (create_time_series(feat),category,person)

def action_features(action_frame,out_path):
    t_series=action_frame['td']
    cats=action_frame['cat']
    #utils.make_dir(out_path)
    features=[(get_vector(td),name) for td,name in zip(t_series,cats)]
    utils.to_labeled_file(out_path,features)

def strict_vector(t_series):
    arr=np.array(t_series)
    hist=np.zeros(arr.shape[1])
    hist=hist.astype(float)
    for i in range(arr.shape[0]):
        cat=np.argmax(arr[i])
        print(cat)
        hist[cat]+=1.0
    hist/=float(arr.shape[0])
    return list(hist)

def get_vector(t_series):    
    t_series_list=get_time_series(t_series)
    #means=[td.mean() for td in t_series_list]
    corel=cross_corel_extr(t_series)
    features=strict_vector(t_series)+corel
    return features

def extract_info(action_name,i):
    info=action_name.split("_")[i]
    info=re.sub(r"[a-zA-Z]","",info)
    return int(info)

def get_time_series(time_series):
	return  [time_series[col_i] for col_i in time_series]

def visualize_actions(action_frame,out_path):
    t_series=action_frame['td']
    names=action_frame['names']
    utils.make_dir(out_path)
    for td,name in zip(t_series,names):
    	full_path=out_path+"/"+name
    	visualize(full_path,td)

def visualize(path,df,show=False):
    path=path.replace(".img",".png")
    plt.figure()
    df.plot()
    if(show):
        plt.show()
    plt.savefig(path,format='png')   
    plt.close()

def create_time_series(features):
    columns=['c'+str(i) for i in range(features.shape[1])]
    index=range(features.shape[0])
    return pd.DataFrame(features,index=index,columns=columns)

def read_all(action_path,cls_path):
    actions=data.read_actions(action_path)
    extr=comp.read_proj_extr(cls_path)
    return actions,extr

def cross_corel_extr(time_series):
    result=time_series.corr()
    corel=result.values.flatten()
    return corel.tolist()

if __name__ == "__main__":
    action_path="../old/final_actions/"
    cls_path="../old/nn/cls/"
    plot_path="../plots"
    out_path="../result/dataset"
    actions=data.read_actions(action_path)
    extr=comp.read_proj_extr(cls_path)
    af=to_action_frame(extr,actions)
    action_features(af,out_path)
    #splited_dataset(af,out_path)
    #visualize_actions(af,out_path)