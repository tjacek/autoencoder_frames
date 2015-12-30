import numpy as np
import theano
import theano.tensor as T
import deep
import autoencoder
import nn

class SdaModel(object):
    def __init__(self,ae_layer,hidden,logistic):
    	self.ae_layer=ae_layer
        self.hidden=hidden
        self.logistic=logistic
        
    def get_params(self):
        params=self.ae_layer.get_params()
        params+=self.hidden.get_params()
        params+=self.logistic.get_params()  
        return params  

def built_sda_cls(n_cats,ae_path,hyper_params=None):
    if(hyper_params==None):
        hyper_params=get_hyper_params(n_cats)
        hyper_params['ae_path']=ae_path
    model= create_sda_model(hyper_params)
    free_vars=deep.LabeledImages()
    train,test,prob_dist=nn.create_nn_fun(free_vars,model,hyper_params)
    return deep.Classifier(free_vars,model,train,test,prob_dist)

def create_sda_model(hyper_params):
    n_in=hyper_params['n_in']
    n_ae=hyper_params['n_ae']
    n_hidden=hyper_params['n_hidden']
    n_out=hyper_params['n_out']
    #ae=autoencoder.read_autoencoder(ae_path)
    rand=deep.RandomNum()
    ae_layer=deep.create_layer((n_in,n_ae),rand,"_ae")
    hidden=deep.create_layer((n_ae,n_hidden),rand,"_hidden")
    logistic=deep.create_layer((n_hidden,n_out),rand,"_vis")
    return SdaModel(ae_layer,hidden,logistic)

def get_hyper_params(n_cats=2,n_in=3600,learning_rate=0.05):
    params={'learning_rate': learning_rate,'ae_path':"placeholder",
            'n_in':n_in,'n_ae':600,'n_hidden':300,'n_out':n_cats}
    return params