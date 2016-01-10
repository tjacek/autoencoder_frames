import utils
import numpy as np
import pandas as pd
import deep.sda as sda
import features
import features.indicator as ind
import data
import re
import ConfigParser
import utils

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
        
def create_time_series(conf,dim=0):
    action_path=conf['action']
    cls_path=conf['cls_ts']
    cls_config=conf['cls_config']
    out_path=conf['series']
    actions=data.read_actions(action_path)
    extractor=sda.read_sda(cls_path,cls_config)
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

def read_config(config_path):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    conf=config.items("Extract")
    conf=dict([ list(pair_i) for pair_i in conf])
    return conf

if __name__ == "__main__":
    config_path="../cascade4/config/throw.cfg"
    conf=read_config(config_path)
    dim=int(conf['dim']) 
    create_time_series(conf,dim)
    indicator=int(conf['indicator'])
    if(indicator!=0):    
        ind.extract_indicator_features(conf['series'],conf['dataset'],indicator)
    else:
        features.extract_features(conf['series'],conf['dataset'])
    features.make_sequences(conf['series'],conf['seq'],0)