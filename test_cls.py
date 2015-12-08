import utils
import data
import deep.composite as comp
import numpy as np

def show_category(dim,size,action_path,cls_path,out_path):
    utils.make_dir(out_path)
    for i in range(size):
        full_path=out_path+"cls"
        print(full_path)
        apply_cls(dim,i,action_path,cls_path,full_path)

def apply_cls(dim,s_cat,action_path,cls_path,out_path):
    actions=data.read_actions(action_path)
    extr=comp.read_composite(cls_path)
    imgs=data.get_named_projections(dim,actions)
    img_cats=[]
    for img_i in imgs:
        img_j=np.reshape(img_i[1],(1,img_i[1].size))
        cat=extr.get_category(img_j)
        if(s_cat==cat):
        	img_cats.append(img_i)
    out_path=out_path+str(s_cat)+"/"
    utils.make_dir(out_path)
    for cat,img_i in img_cats:
    	full_path=out_path+cat#"img"+str(i)
        img_i=np.reshape(img_i, (60,60))
        utils.save_img(full_path,img_i)

if __name__ == "__main__":
    action_path="../cats/a16/"
    cls_path="../nn/cls/xy"
    out_path="../out/"
    show_category(2,7,action_path,cls_path,out_path)