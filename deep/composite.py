import utils
import numpy as np
import theano
import theano.tensor as T
import deep
import deep.autoencoder as autoencoder
import deep.nn as nn

class FeatureExtractor(object):
    def __init__(self,ae,nn):
	self.autoencoder=ae
	self.nn=nn
    
    def train(self,img,label):
        reduced_img=self.autoencoder.test(img)
        return self.nn.train(reduced_img,label)

    def get_features(self,img):
        redu_img=self.autoencoder.test(img)
        return self.nn.get_features(redu_img)

    def get_model(self):
        ae_model=self.autoencoder.model
        nn_model=self.nn.model
        return CompositeModel(ae_model,nn_model)

    def get_category(self,img):
        redu_img=self.autoencoder.test(img)
        return self.nn.test(redu_img)[0]

class CompositeModel(object):
    def __init__(self,ae_model,nn_model):
        self.ae_model=ae_model
        self.nn_model=nn_model

class ProjectionExtractor(object):
    def __init__(self,xy,zx,zy):
        self.xy=xy
        self.zx=zx
        self.zy=zy

    def get_features(self,xy,zx,zy):
        xy_feat=self.xy.get_features(xy)
        zx_feat=self.zx.get_features(zx)
        zy_feat=self.zy.get_features(zy)
        feat=xy_feat+zx_feat+zy_feat
        return feat

def read_proj_extr(in_path):
    paths=[in_path+pf for pf in ['xy','zx','zy']]
    cls=[read_composite(cls_path) for cls_path in paths]
    return ProjectionExtractor(cls[0],cls[1],cls[2])

def create_extractor(n_cats,ae_path,n=800):
    ae=autoencoder.read_autoencoder(ae_path)
    hyper_params=nn.get_hyper_params(n_cats,500)
    nn_cls=nn.built_nn_cls(n_cats,hyper_params)
    return FeatureExtractor(ae,nn_cls)

def read_composite(cls_path):
    model=utils.read_object(cls_path)
    rand=deep.RandomNum()
    ae=autoencoder.init_autoencoder(model.ae_model,rand)
    cls_nn=nn.init_nn(5,model.nn_model)
    return FeatureExtractor(ae,cls_nn)