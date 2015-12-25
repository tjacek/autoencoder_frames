import numpy as np
import theano
import theano.tensor as T
import deep


class SdA(object):
 	def __init__(self, ae_layer,hidden,logistic):
 		self.model = SdaModel(ae_layer,hidden,logistic)

    def train(self,img,label):
        return None

    def get_features(self,img):
        return None

    def get_model(self):
        return None

    def get_category(self,img):
        return None

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

def create_nn_fun(free_vars,model,hyper_params):

    return train,test,prob_dist