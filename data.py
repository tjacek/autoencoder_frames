import utils
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

class FinalFrame(object):
    def __init__(self,imgs):
    	self.n_imgs=len(imgs)
        self.imgs=[img.flatten() for img in imgs]

def read_actions(dir_path):
    actions_paths=utils.get_dirs(dir_path)
    actions_paths=utils.append_path(dir_path, actions_paths)
    actions=[read_final_action(path) for path in actions_paths]
    return actions#create_action_frame(actions)

def read_final_action(path):
	name=utils.get_name(path)
	proj_action=proj.read_img_action(path+"/")
	frames=[FinalFrame(fr.projections) for fr in proj_action.frames]
	return FinalAction(name,frames)

def get_projections(i,actions):
	projections=[]
	for action in actions:
		projections+=action.get_dim(i)
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
    action_path="../final_actions/"
    actions=read_actions(action_path)
    projections=get_projections(0,actions)
    print(len(actions))
    print(len(projections))