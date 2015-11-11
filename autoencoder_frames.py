import utils
import data.images as images
import data.actions as acts
import deep,deep.nn as nn

def create_cls(img_path,out_path):
    img_frame=images.read_image_frame(img_path)
    cls=nn.built_nn_cls()
    cls=deep.learning_iter(img_frame,cls,n_epochs=100)
    utils.save_object(cls,out_path)
    return cls

def action_imgs(action_path,cls_path,out_path):
    action_frame=acts.read_action_frame(action_path)
    cls=utils.read_object(cls_path)
    convert_actions(action_frame,cls)
    utils.make_dir(out_path)
    actions=action_frame['Action']
    act_imgs=[ (action.name,action.to_action_img()) for action in actions]
    utils.save_images(out_path,act_imgs)
    print(action_frame.head())

def convert_actions(action_frame,cls):
    actions= action_frame['Action']
    return [action.to_time_series(cls) for action in actions]

if __name__ == "__main__":
    img_path="../imgs/"
    cls_path="../nn/test"
    action_path="../small/"
    out_path="../action_imgs/"
    action_imgs(action_path,cls_path,out_path)
    #cls=create_cls(img_path,out_path)
    #deep.check_prediction(img_frame,cls)