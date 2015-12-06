import numpy as np
import pandas as pd
import deep.composite as comp
import data

def to_time_series(extr,actions):
	return [action_time_series(extr,act) for act in actions]

def action_time_series(extr,action):
    feat=[extr.get_features(fr[0],fr[1],fr[2]) for fr in action.frames]
    feat=np.array(feat)
    return create_time_series(feat)

def create_time_series(features):
    columns=['c'+str(i) for i in range(features.shape[1])]
    index=range(features.shape[0])
    return pd.DataFrame(features,index=index,columns=columns)

if __name__ == "__main__":
    action_path="../final_actions/"
    cls_path="../nn/cls/"
    actions=data.read_actions(action_path)
    extr=comp.read_proj_extr(cls_path)
    to_time_series(extr,actions)