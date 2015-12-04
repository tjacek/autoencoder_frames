import data
import deep
import deep.autoencoder as ae
import deep.composite as comp
import utils

def create_cls(in_path,ae_out,out_path):
    imgs=data.read_image_frame(in_path)
    n_cats=2
    cls=comp.create_extractor(n_cats,ae_path)
    X=imgs['Images'].tolist()
    y=imgs['Category'].tolist()
    print(X[0].shape)
    deep.learning_iter_super(cls,X,y,n_epochs=500)
    utils.save_object(cls.get_model(),out_path) 

if __name__ == "__main__":
    action_path="../cls"
    ae_path="../nn/ae/xy"
    cls_path="../nn/cls"
    create_cls(action_path,ae_path,cls_path)