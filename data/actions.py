import re,pandas as pd 
import utils
import numpy as np

class Action(object):
    def __init__(self,name,images):
	self.name=name
	self.orginal_shape=images[0].shape
	self.images=[standarize_img(img) for img in images]
	self.dim=self.images[0].shape[0]
	self.length=len(self.images)
        self.time_series=None

    def to_time_series(self,cls):
        time_series=[cls.get_features(img) for img in self.images] 
        time_series=[ts.flatten() for ts in time_series] 
        time_series=np.array(time_series)
        columns=['c'+str(i) for i in range(time_series.shape[1])]
        index=range(self.length)
        self.time_series=pd.DataFrame(time_series,index=index,columns=columns)

    def to_action_img(self):
        time_series=self.time_series
        multipler=10
        cols=time_series.columns
        length=len(time_series)
        dim=len(cols)*multipler
        action_img=np.zeros((length,dim))
        for i in range(length):
            for j in range(dim):
                chnl=cols[j/multipler]
                action_img[i][j]=time_series[chnl][i]
        return action_img

    def __str__(self):
        return self.name

def read_action_frame(dir_path):
    actions_paths=utils.get_dirs(dir_path)
    actions_paths=utils.append_path(dir_path, actions_paths)
    actions=[read_action(path) for path in actions_paths]
    return create_action_frame(actions)

def create_action_frame(actions):
    names=[action.name for action in actions]
    cats=[ extract_info(name,0) for name in names]
    persons=[ extract_info(name,1) for name in names]
    return  pd.DataFrame({ 'Action':actions,
                           'Category':cats,
                           'Person':persons})

def read_action(action_path):
    action_name=get_action_name(action_path)
    images=utils.read_img_dir(action_path)
    return Action(action_name,images)

def save_action(path,action):
    action_path=path+str(action)+"/"
    utils.make_dir(action_path)
    for i,img in enumerate(action.images):
        img_path=action_path+str(action)+"_"+str(i)
        img=np.reshape(img,(80,40))
        utils.save_img(img_path,img)

def get_action_name(action_path):
    name=action_path.split("/")[-1]
    return name.replace("_sdepth","")

def extract_info(action_name,i):
    info=action_name.split("_")[i]
    info=re.sub(r"[a-zA-Z]","",info)
    return int(info)

def standarize_img(img):
    img=img.flatten()
    #img=img/max(img)
    return np.reshape(img,(1,img.size))  

if __name__ == "__main__":
    path="../../"#"/home/user/cls/"
    action_path=path+"small/"#a01_s01_e01_sdepth"
    actions=read_action_frame(action_path)
    print(actions.head())
    print(len(actions))
