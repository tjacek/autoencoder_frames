import pandas as pd 
import utils

class Action(object):
    def __init__(self,name,images):
	self.name=name
	self.orginal_shape=images[0].shape
	self.images=[img.flatten() for img in images]
	self.dim=self.images[0].shape[0]
	self.length=len(self.images)

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

def get_action_name(action_path):
    return action_path.split("/")[-1]

def extract_info(action_name,i):
    return action_name.split("_")[i]

if __name__ == "__main__":
    path="../"#"/home/user/cls/"
    action_path=path+"small/"#a01_s01_e01_sdepth"
    actions=read_action_frame(action_path)
    print(actions.head())
    print(len(actions))
