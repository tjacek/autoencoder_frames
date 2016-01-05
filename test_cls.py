import utils
import data
import deep.sda as sda
import numpy as np

def show_category(dim,size,params):
    out_path=params['out_path']
    utils.make_dir(out_path)
    for i in range(size):
        full_path=out_path+"cls"
        print(full_path)
        apply_cls(dim,i,params)

def apply_cls(dim,s_cat,params):
    actions=data.read_actions(params['action_path'])
    extr=sda.read_sda(params['cls_path'],params['conf_path'])
    imgs=data.get_named_projections(dim,actions)
    img_cats=[]
    for img_i in imgs: 
        img_j=np.reshape(img_i[1],(1,img_i[1].size))
        cat=extr.test(img_j)
        if(s_cat==cat):
        	img_cats.append(img_i)
    out_path=params['out_path']
    out_path=out_path+str(s_cat)+"/"
    utils.make_dir(out_path)
    for cat,img_i in img_cats:
    	full_path=out_path+cat
        img_i=np.reshape(img_i, (60,60))
        utils.save_img(full_path,img_i)

def get_params():
    return {'action_path':"../_final_actions/",
            'cls_path':"../cascade4/nn/cls/large",
            'conf_path':"../cascade4/cls_config/xy",
            'out_path':"../out/"}


if __name__ == "__main__":
    show_category(0,17,get_params())