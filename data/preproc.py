import actions,utils
import numpy as np

def standarize_actions(in_path,out_path):
    action_frame=actions.read_action_frame(in_path)
    utils.make_dir(out_path)
    for action in action_frame['Action']:
    	print(action)
    	discretize_action(action)
    	actions.save_action(out_path,action)

def discretize_action(action):
    action.images=[discretize_img(img) for img in action.images]
    return action

def discretize_img(img):
    nonzero=img[np.nonzero(img)]
    avg=np.average(nonzero)
    sd=np.std(nonzero)
    return [disc_pixel(pix,avg,sd) for pix in img.flatten()]
 
def disc_pixel(pixel,avg,sd):
    if(pixel==0.0):
        return 0.0
    pixel-=avg
    pixel/=sd
    pixel=np.floor(pixel)
    if(pixel==-1 or pixel==1):
    	pixel=0.0
    return (pixel+3.0)*10.0   

def normalize_action(action):
    ar_max=act_max(action)
    ar_min=act_min(action)
    action.images=[translate(x_i,ar_min) for x_i in action.images ]
    det=ar_max-ar_min
    action.images=[scale(x_i,ar_min) for x_i in action.images ]
    return action

def act_max(action):
    max_array=[np.amax(x_i) for x_i in action.images]
    return float(max(max_array))

def act_min(action):
	min_array=[np.min(a[np.nonzero(a)]) for a in action.images]
	return float(min(min_array))

def translate(img,min_action):
    img[np.nonzero(img)]-=(min_action - 1.0)
    return img

def scale(img,det):
    img/=(det + 1.0)
    img*=200.0
    return img