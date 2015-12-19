import utils
import numpy as np
import pandas as pd
import deep.composite as comp
import features
import features.indicator as ind
import data
import re
import ConfigParser

class ActionTimeSeries(object):
    def __init__(self,name,cat,person,t_series):
        self.name=name
        self.cat=cat
        self.person=person
        self.t_series=t_series
        self.dim=len(self.t_series[0])

    def to_array(self):
        arrays=[]
        for i in range(self.dim):
            dim_i=[t_j[i].flatten() for t_j in self.t_series]
            arrays.append(np.array(dim_i))
        return arrays
        
def create_time_series(action_path,cls_path,out_path,dim=0):
    actions=data.read_actions(action_path)
    extractor=comp.read_composite(cls_path)
    all_t_series=[make_action_ts(extractor,action,dim) for action in actions]
    utils.make_dir(out_path)
    for action_ts in all_t_series:
        full_path=out_path+action_ts.name
        utils.save_object(action_ts,full_path)

def make_action_ts(extractor,action,dim=0):
    t_series=[extractor.get_features(fr[dim]) for fr in action.frames]
    name=action.name
    name=name.replace(".img","")
    category=extract_info(name,0)
    person=extract_info(name,1)
    return ActionTimeSeries(name,category,person,t_series)

def extract_info(action_name,i):
    info=action_name.split("_")[i]
    info=re.sub(r"[a-zA-Z]","",info)
    return int(info)

if __name__ == "__main__":
    config_path="../cascade/config/hand.cfg"
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    conf=config.items("Extract")
    conf=dict([ list(pair_i) for pair_i in conf])
    dim=int(conf['dim']) 
    create_time_series(conf['action'],conf['cls_ts'],conf['series'],dim)
    if(bool(conf['indicator'])):    
        ind.extract_indicator_features(conf['series'],conf['dataset'])
    else:
        features.extract_features(conf['series'],conf['dataset'])
    features.make_sequences(conf['series'],conf['seq'],0)