import utils
import numpy as np
import preproc.projections as proj
import pandas as pd

class FinalAction(object):
    def __init__(self,name,frames,n_imgs=3):
    	self.name=name
        self.frames=frames
        self.n_imgs=n_imgs

    def get_dim(self,i):
    	assert i<self.n_imgs
    	return [frame.imgs[i] for frame in self.frames]

    def to_arrays(self):
        return [self.get_dim(i) for i in range(self.n_imgs)]

    def __len__(self):
        return len(self.frames)

    def diff(self):
        diff=[self.frames[i]-self.frames[i-1] for i in range(1,len(self))]
        action_diff=FinalAction(self.name,diff)
        diff=action_diff.to_arrays()
        diff=[sum(diff_i) for diff_i in diff]
        return FinalFrame(diff)

    def __str__(self):
        return self.name
        
class FinalFrame(object):
    def __init__(self,imgs):
    	self.n_imgs=len(imgs)
        self.imgs=[img.flatten() for img in imgs]

    def save(self,in_path):
        for i,img_i in enumerate(self.imgs):
            full_path=in_path+"_"+str(i)
            if(i==0):
                img_i=np.reshape(img_i,(60,60))
                utils.save_img(full_path,img_i)

    def __getitem__(self,index):
        img=self.imgs[index]
        img=np.reshape(img,(1,img.size))
        return img

    def __sub__(self,other):
        new_imgs=[img_i-img_j for img_i,img_j in zip(self.imgs,other.imgs)]
        new_imgs=[np.absolute(img_i) for img_i in new_imgs]
        return FinalFrame(new_imgs)

def get_max(in_path,out_path):
    actions=read_actions(in_path)
    named_imgs=[]
    for i,action_i in enumerate(actions):
        dim_x=action_i.get_dim(0)
        max_array=np.zeros(dim_x[0].shape)
        maxim=[np.argmax(img_i)  for img_i in dim_x]
        max_array[maxim]=i*10#1.0
        max_array=np.reshape(max_array,(60,60))
        named_imgs.append((action_i.name,max_array))
    utils.make_dir(out_path)
    utils.save_images(out_path,named_imgs)

def get_action_diff(in_path,out_path):
    actions=read_actions(in_path)
    diff_fr=[ (action_i.name,action_i.diff()) for action_i in actions]
    utils.make_dir(out_path)
    for name_i,frame_i in diff_fr:
        frame_i.save(out_path+name_i)

def read_actions(dir_path):
    actions_paths=utils.get_dirs(dir_path)
    actions_paths=utils.append_path(dir_path, actions_paths)
    actions=[read_final_action(path) for path in actions_paths]
    return actions#create_action_frame(actions)

def read_final_action(path):
    name=utils.get_name(path)
    print(name)
    proj_action=proj.read_img_action(path+"/",False)
    frames=[FinalFrame(fr.projections) for fr in proj_action.frames]
    return FinalAction(name,frames)

def get_projections(i,actions):
    projections=[]
    for action in actions:
        projections+=action.get_dim(i)
    projections=[img_i/255.0 for img_i in projections]
    return projections

def get_named_projections(i,actions):
    projections=[]
    for action in actions:
        name=action.name
        frames=action.get_dim(i)
        n_proj=[(name+str(j),fr)for j,fr in enumerate(frames)]
        projections+=n_proj
    return projections

def read_image_frame(dir_path):
    cat_paths=utils.get_paths(dir_path,dirs=True)
    images=[]
    for cat,cat_path in enumerate(cat_paths):
        print(cat_path)
        cat_imgs=utils.get_paths(cat_path,dirs=False)
        cat_imgs=[(cat,read_img(img_path)) for img_path in cat_imgs]
        images+=cat_imgs
    return create_img_frame(images)

def create_img_frame(images):
    labels=[ img[0] for img in images]
    flat_imgs=[ img[1] for img in images]
    return  pd.DataFrame({ 'Images':flat_imgs,
                           'Category':labels,})
    
def read_img(img_path):
    img=proj.read_img(img_path)
    return img.flatten()

if __name__ == "__main__":
    action_path="../_final_actions/"
    get_max(action_path,"../max/")
    #actions=read_actions(action_path)
    #projections=get_projections(0,actions)
    #print(len(actions))
    #print(len(projections))