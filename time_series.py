import utils
import numpy as np
import pandas as pd
import deep.composite as comp
import data
import matplotlib.pyplot as plt

def to_action_frame(extr,actions):
    td=[action_time_series(extr,act) for act in actions]
    time_series=[td_i[0] for td_i in td]
    names=[td_i[1] for td_i in td]
    return pd.DataFrame({'td':time_series,'names':names})

def action_time_series(extr,action):
    print(action)
    feat=[extr.get_features(fr[0],fr[1],fr[2]) for fr in action.frames]
    feat=np.array(feat)
    return (create_time_series(feat),str(action))

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

if __name__ == "__main__":
    action_path="../final_actions/"
    cls_path="../nn/cls/"
    out_path="../plots"
    actions=data.read_actions(action_path)
    extr=comp.read_proj_extr(cls_path)
    af=to_action_frame(extr,actions)
    visualize_actions(af,out_path)