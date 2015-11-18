import utils,data,deep
import data.images as images
import data.actions as acts
import deep.nn as nn
import deep.composite as comp
import series

def create_cls(img_path,out_path):
    img_frame=images.read_image_frame(img_path)
    n_cats=data.get_n_cats(img_frame)
    print(n_cats)
    cls=nn.built_nn_cls(n_cats)
    cls=deep.learning_iter(img_frame,cls,n_epochs=1000)
    utils.save_object(cls,out_path)
    return cls

def action_imgs(action_path,cls_path,out_path):
    action_frame=acts.read_action_frame(action_path)
    cls=comp.read_composite(cls_path)#utils.read_object(cls_path)
    convert_actions(action_frame,cls)
    utils.make_dir(out_path)
    actions=action_frame['Action']
    act_imgs=[ (action.name,action.to_action_img()) for action in actions]
    utils.save_images(out_path,act_imgs)
    print(action_frame.head())

def action_features(action_path,cls_path,out_path):
    action_frame=acts.read_action_frame(action_path)
    cls=comp.read_composite(cls_path)#utils.read_object(cls_path)
    convert_actions(action_frame,cls)
    extractor=series.trivial_extr
    labeled_vectors=series.extract_features(action_frame,extractor)
    utils.to_labeled_file(out_path,labeled_vectors)

def convert_actions(action_frame,cls):
    actions= action_frame['Action']
    return [action.to_time_series(cls) for action in actions]

if __name__ == "__main__":
    img_path="../imgs/"
    cls_path="../nn/comp1"
    action_path="../large_/"
    out_path="../action_imgs/"
    #cls=create_cls(img_path,cls_path)
    #utils.read_object()
    #action_imgs(action_path,cls_path,out_path)
    action_features(action_path,cls_path,"../result/af.lb")
    #deep.check_prediction(img_frame,cls)