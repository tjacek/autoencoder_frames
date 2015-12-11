import utils
import numpy as np
import pandas as pd
import deep.composite as comp
import data
import matplotlib.pyplot as plt
import re

class ActionTimeSeries(object):
    def __init__(self,name,cat,person,t_series):
        self.name=name
        self.cat=cat
        self.person=person
        self.t_series=t_series

def create_time_series(action_path,cls_path,out_path):
    actions=data.read_actions(action_path)
    extractor=comp.read_proj_extr(cls_path)
    all_t_series=[make_action_ts(extractor,action) for action in actions]
    utils.make_dir(out_path)
    for action_ts in all_t_series:
        full_path=out_path+action_ts.name
        utils.save_object(action_ts,full_path)

def make_action_ts(extractor,action):
    t_series=[extractor.get_features(fr[0],fr[1],fr[2]) for fr in action.frames]
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
    action_path="../_final_actions/"
    cls_path="../nn/cls/"
    out_path="../_time_series"
    create_time_series(action_path,cls_path,out_path)    