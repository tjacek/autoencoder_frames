import data
import deep
import deep.autoencoder as ae
import deep.composite as comp
import utils
import ConfigParser

def create_cls(in_path,ae_path,out_path):
    imgs=data.read_image_frame(in_path)
    n_cats=4
    X=imgs['Images'].tolist()
    y=imgs['Category'].tolist()
    n_cats=max(y)+1
    cls=comp.create_extractor(n_cats,ae_path)
    deep.learning_iter_super(cls,X,y,n_epochs=1000)
    utils.save_object(cls.get_model(),out_path) 
    return cls

def create_proj_cls(in_path,ae_path,out_path):
    dirs=['xy','zx','zy']
    img_paths=[ in_path+'/'+d for d in dirs]
    ae_paths=[ ae_path+'/'+d for d in dirs]
    print(img_paths)
    clas=[create_cls(img_path,ae_path) for img_path,ae_path in zip(img_paths,ae_paths)]
    extr=comp.ProjectionExtractor(clas[0],clas[1],clas[2])
    comp.save_proj_extr(extr,out_path)

if __name__ == "__main__":
    config_path="../cascade/config/hard.cfg"
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    dict_conf=config.items("Cls")
    dict_conf=dict([ list(pair_i) for pair_i in dict_conf]) 
    create_cls(dict_conf['data'],dict_conf['ae'],dict_conf['cls'])