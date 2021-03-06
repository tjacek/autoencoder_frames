import numpy as np
import theano
import theano.tensor as T
import deep
import autoencoder
import tools

class MlpModel(object):
    def __init__(self,hidden,logistic):
        self.hidden=hidden
        self.logistic=logistic
        
    def get_params(self):
        return self.hidden.get_params() + self.logistic.get_params()

def built_nn_cls(hyper_params):
    model= create_mlp_model(hyper_params)
    free_vars=deep.LabeledImages()
    train,test,prob_dist=create_nn_fun(free_vars,model,hyper_params)
    return deep.Classifier(free_vars,model,train,test,prob_dist)

def create_mlp_model(hyper_params):
    n_in=hyper_params['n_in']
    n_hidden=hyper_params['n_hidden']
    n_out=hyper_params['n_out']
    rand=deep.RandomNum()
    hidden_shape=(n_in,n_hidden)
    hidden=deep.create_layer(hidden_shape,rand,"_hidden")
    vis_shape=(n_hidden,n_out)
    logistic=deep.create_layer(vis_shape,rand,"_vis")
    return MlpModel(hidden,logistic)

def create_nn_fun(free_vars,model,hyper_params):
    learning_rate=hyper_params['learning_rate']
    py_x=get_px_y(free_vars,model)
    loss=tools.get_loss_function(free_vars,py_x)
    return tools.construct_functions(free_vars,model,py_x,loss,learning_rate)

def get_px_y(free_vars,model):
    hidden=model.hidden
    output_layer=model.logistic
    h = T.nnet.sigmoid(T.dot(free_vars.X, hidden.W) + hidden.b)
    pyx = T.nnet.softmax(T.dot(h, output_layer.W) + output_layer.b)
    return pyx