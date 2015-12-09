import utils
import numpy as np
import scipy.misc as image
import point_cloud as pc
import cv2

DIRS=['xy/','zx/','zy/']

class ProjectionAction(object):
    def __init__(self,frames):
        self.frames = frames

    def smooth(self):
        for frame in self.frames:
            frame.smooth()

    def scaled_action(self):
        scaled_frames=[frame.scaled_frame() for frame in self.frames]
        return ProjectionAction(scaled_frames)

    def save(self,path):
        utils.make_dir(path)
        name=utils.get_name(path)
    	for proj_type in DIRS:
    	    utils.make_dir(path+'/'+proj_type)
        for i,frame in enumerate(self.frames):
            frame.save(path,name+str(i))

class ProjectionFrame(object):
    def __init__(self,projections):
        self.projections=projections   

    def smooth(self):
        self.projections[1]=smooth_img(self.projections[1])
        self.projections[2]=smooth_img(self.projections[2])

    def scaled_frame(self):
        projections=[scale_3D(self.projections[0])]
        projections+=scale_2D(self.projections[1:3])
        return ProjectionFrame(projections)

    def save(self,path,name):
        for proj,postfix in zip(self.projections,DIRS):
            proj_path=path+"/"+postfix
            utils.make_dir(proj_path)
            full_path=proj_path+name
            print(full_path)
            utils.save_img(full_path,proj)

def scale_3D(img3D):
    p_cloud=pc.create_point_cloud(img3D,False)
    start_dims=p_cloud.find_max()
    end_dims=(60,60,60)
    new_dims=new_dim(start_dims,end_dims)
    p_cloud.rescale(new_dims)
    return p_cloud.to_scaled_img(end_dims)

def scale_2D(imgs2D):
    start_dims=[img.shape for img in imgs2D]
    end_dims=[(60,60) for img in imgs2D]
    new_dims=[new_dim(d0,d1) for d0,d1 in zip(start_dims,end_dims)]
    p_clouds=[pc.create_point_cloud(img,True) for img in imgs2D]
    for p_cloud,dim in zip(p_clouds,new_dims):
        p_cloud.rescale(dim) 
    projections=[p_cloud.to_img(dim) for p_cloud,dim in zip(p_clouds,end_dims)]
    return projections

def new_dim(dx,dy):
    new_dim=[float(x_i)/float(y_i) for x_i,y_i in zip(dx,dy)]
    return tuple(new_dim)

def smooth_img(img):
    print(img.shape)
    kern=np.ones((5,5),np.float32) #/64.0
    #img=img*100
    img=cv2.filter2D(img,-1,kern)
    img=img.astype(np.uint8)
    img=cv2.threshold(img,0,255,cv2.THRESH_BINARY)[1]
    return img

def read_img_action(path):
    names=utils.get_files(path+"xy/")
    #names_size=len(raw_names)
    #names=["act"+str(i)+".png" for i in range(names_size)]
    names=[utils.get_name(frame_path) for frame_path in names]
    print(names)
    new_proj=[read_projection_frame(path,name) for name in names]
    return ProjectionAction(new_proj)

def read_projection_frame(frame_path,name,normal=True):
    frame_paths=[ frame_path+prefix+name  for prefix in DIRS]
    projs=[read_img(path) for path in frame_paths]
    return ProjectionFrame(projs)

def read_img(path):
    img=image.imread(path)
    img=img.astype(float)
    img=img#/255.0
    return img

if __name__ == "__main__":
    action_path="../show/"
    action=read_img_action(action_path)
    scaled_action=action.scaled_action()
    scaled_action.save("../show/")
