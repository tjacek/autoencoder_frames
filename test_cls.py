import utils
import data
import deep.sda as sda
import numpy as np
import sys

def show_category(dim,size,params):
    out_path=params['out_path']
    utils.make_dir(out_path)
    actions=data.read_actions(params['action_path'])
    extr=sda.read_sda(params['cls_path'],params['conf_path'])
    for i in range(size):
        full_path=out_path+"cls"
        print(full_path)
        apply_cls(dim,i,params,actions,extr)

def apply_cls(dim,s_cat,params,actions,extr):
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
            'cls_path':"../cascade5/nn/cls/large",
            'conf_path':"../cascade5/cls_config/xy",
            'out_path':"../out/"}

def parse_args(args):
    if(len(args)>1):
        cats=int(args[1])
    else:
        cats=20
    return cats

if __name__ == "__main__":
    cats=parse_args(sys.argv)
    show_category(0,cats,get_params())