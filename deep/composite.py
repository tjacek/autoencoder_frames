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
        reduced_img=self.autoencoder.test(img)
	return self.nn.get_features(redu_img)

    def get_model(self):
        ae_model=self.autoencoder.model
        nn_model=self.nn.model
        return CompositeModel(ae_model,nn_model)

class CompositeModel(object):
    def __init__(self,ae_model,nn_model):
        self.ae_model=ae_model
        self.nn_model=nn_model

def create_extractor(n_cats,ae_path,n=900):
    ae=autoencoder.read_autoencoder(ae_path)
    hyper_params=nn.get_hyper_params(n_cats,1600)
    nn_cls=nn.built_nn_cls(n_cats,hyper_params)
    return FeatureExtractor(ae,nn_cls)
