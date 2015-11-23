import utils,data,deep
import data.images as images
import data.actions as acts
import deep.nn as nn
import deep.composite as comp
import series

def create_cls(img_path,ae_path,out_path):
    img_frame=images.read_image_frame(img_path)
    n_cats=data.get_n_cats(img_frame)
    print(n_cats)
    cls=comp.create_extractor(n_cats,ae_path)
    #cls=nn.built_nn_cls(n_cats)
    cls=deep.learning_iter(img_frame,cls,n_epochs=1000)
    utils.save_object(cls.get_model(),out_path)
    return cls

def action_imgs(action_frame,out_path):
    utils.make_dir(out_path)
    actions=action_frame['Action']
    act_imgs=[ (action.name,action.to_action_img()) for action in actions]
    utils.save_images(out_path,act_imgs)
    print(action_frame.head())

def action_features(action_path,cls_path,out_path):
    action_frame,cls=read_all(action_path,cls_path)
    convert_actions(action_frame,cls)
    extractor=series.trivial_extr
    labeled_vectors=series.extract_features(action_frame,extractor)
    utils.to_labeled_file(out_path,labeled_vectors)
    return action_frame

def splited_af(action_path,cls_path,out_path):
    action_frame,cls=read_all(action_path,cls_path)
    convert_actions(action_frame,cls)
    extractor=series.curry_extractor(None)
    data.save_splited(out_path,action_frame,extractor)
    return action_frame

def convert_actions(action_frame,cls):
    actions= action_frame['Action']
    return [action.to_time_series(cls) for action in actions]

def apply_cls(action_path,cls_path,out_path):
    action_frame,cls=read_all(action_path,cls_path)
    actions= action_frame['Action']
    images=[]
    for act in actions:
        images+=[( cls.get_category(img) , utils.to_2D(img) ) for img in act.images]
    def give_name(cat,i):
        return 'cat_'+str(cat)+'_'+str(i)     
    imgs=[(give_name(img[0],i),img[1]) for i,img in enumerate(images)]
    utils.save_images(out_path,imgs)

def read_all(action_path,cls_path):
    action_frame=acts.read_action_frame(action_path)
    cls=comp.read_composite(cls_path)
    return action_frame,cls

if __name__ == "__main__":
    img_path="../imgs/"
    cls_path="../nn/comp4_"
    ae_path="../nn/ae3"
    action_path="../large_/"
    out_path="../action_imgs/"
    #cls=create_cls(img_path,ae_path,cls_path)
    #af=action_features(action_path,cls_path,"../result/af.lb")
    #action_imgs(af,out_path)
    splited_af(action_path,cls_path,"../result/af.lb")
    #apply_cls(action_path,cls_path,"../test2/")