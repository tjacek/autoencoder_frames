import data
import deep
import deep.autoencoder as ae
import deep.composite as comp
import utils

def create_cls(in_path,ae_path):
    imgs=data.read_image_frame(in_path)
    n_cats=2
    cls=comp.create_extractor(n_cats,ae_path)
    X=imgs['Images'].tolist()
    y=imgs['Category'].tolist()
    print(X[0].shape)
    deep.learning_iter_super(cls,X,y,n_epochs=500)
    return cls
    #utils.save_object(cls.get_model(),out_path) 

def create_proj_cls(in_path,ae_path,out_path):
    dirs=['xy','zx','zy']
    img_paths=[ in_path+'/'+d for d in dirs]
    ae_paths=[ ae_path+'/'+d for d in dirs]

    clas=[create_cls(img_path,ae_path) for img_path,ae_path in zip(img_paths,ae_paths)]
    extr=comp.ProjectionExtractor(clas[0],clas[1],clas[2])
    comp.save_proj_extr(extr,out_path)

if __name__ == "__main__":
    action_path="../cls"
    ae_path="../nn/ae/"
    cls_path="../nn/cls/"
    create_proj_cls(action_path,ae_path,cls_path)