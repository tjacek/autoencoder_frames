import utils
import numpy as np
from binary import RawAction
from point_cloud import create_point_cloud,show_points
import point_cloud as pc

class Action(object):
    def __init__(self,point_clouds):
    	self.point_clouds=point_clouds
    	self.standarized=False
    	self.dim=None
    	#self.point_clouds[0].apply(show_points)

    def standarize(self):
        if(not self.standarized):
    	    min_points=self.min()
            self.apply(lambda p:p-min_points)
            self.dim=self.max()
    	    self.standarized=True

    def apply(self,fun):
    	for cloud in self.point_clouds:
    		cloud.apply(fun)

    def max(self):
    	max_points=[cloud.find_max() for cloud in self.point_clouds]
    	max_points=np.array(max_points)
    	glob_max=np.amax(max_points,axis=0)
    	return glob_max

    def min(self):
        min_points=[cloud.find_min() for cloud in self.point_clouds]
        min_points=np.array(min_points)
        glob_min=np.amin(min_points,axis=0)
        return glob_min

    def save_imgs(self,out_path):
    	utils.make_dir(out_path)
    	imgs=[cloud.to_img(self.dim) for cloud in self.point_clouds]
        imgs=[("act"+str(i),img) for i,img in enumerate(imgs)]
        utils.save_images(out_path,imgs)

    def save_projection(self,out_path):
        utils.make_dir(out_path)
        paths=['xy/','zx/','zy/']
        paths=[out_path+path for path in paths]
        [utils.make_dir(path) for path in paths]
        imgs_xy=self.get_imgs(pc.ProjectionXY())
        utils.save_images(paths[0],imgs_xy)
        imgs_xz=self.get_imgs(pc.ProjectionXZ())
        utils.save_images(paths[1],imgs_xz)
        imgs_zy=self.get_imgs(pc.ProjectionYZ())
        utils.save_images(paths[2],imgs_zy)

    def get_imgs(self,proj):
        imgs=[cloud.to_img(self.dim,proj) for cloud in self.point_clouds]
        imgs=[("act"+str(i),img) for i,img in enumerate(imgs)]
        return imgs

def make_action(raw_action_path):
	raw_action=utils.read_object(raw_action_path)
	p_clouds=[create_point_cloud(img) for img in raw_action.frames]
	return Action(p_clouds)

if __name__ == "__main__":
	action_path="../raw_action"
	action=make_action(action_path)
	action.standarize()
	print(action.dim)
	action.save_projection("../show2/")
