import utils
import data.images as images
import deep,deep.nn as nn

def create_cls(img_path,out_path):
    img_frame=images.read_image_frame(img_path)
    cls=nn.built_nn_cls()
    cls=deep.learning_iter(img_frame,cls,n_epochs=100)
    utils.save_object(cls,out_path)
    return cls

if __name__ == "__main__":
    img_path="../imgs/"
    out_path="../nn/test"
    cls=create_cls(img_path,out_path)
    #deep.check_prediction(img_frame,cls)