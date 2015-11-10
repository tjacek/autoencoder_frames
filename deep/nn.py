import numpy as np
import theano
import theano.tensor as T
import deep

class MlpModel(object):
    def __init__(self,hidden,logistic):
        self.hidden=hidden
        self.logistic=logistic
        
    def get_params(self):
        return self.hidden.get_params() + self.logistic.get_params()

def built_nn_cls(shape=(3200,2)):
    hyper_params=get_hyper_params()
    free_vars=deep.LabeledImages()
    model= create_mlp_model(hyper_params)
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
    loss=get_loss_function(free_vars,py_x)
    input_vars=free_vars.get_vars()
    params=model.get_params()
    update=deep.compute_updates(loss, params, learning_rate)
    train = theano.function(inputs=input_vars, 
                                outputs=loss, updates=update, 
                                allow_input_downcast=True)
    y_pred = T.argmax(py_x, axis=1)
    prob_dist=theano.function(inputs=[free_vars.X], outputs=py_x, 
            allow_input_downcast=True) 
    test=theano.function(inputs=[free_vars.X], outputs=y_pred, 
            allow_input_downcast=True) 
    return train,test,prob_dist

def get_px_y(free_vars,model):
    hidden=model.hidden
    output_layer=model.logistic
    h = T.nnet.sigmoid(T.dot(free_vars.X, hidden.W) + hidden.b)
    pyx = T.nnet.softmax(T.dot(h, output_layer.W) + output_layer.b)
    return pyx

def get_loss_function(free_vars,py_x):
    return T.mean(T.nnet.categorical_crossentropy(py_x,free_vars.y))

def get_hyper_params(learning_rate=0.05):
    params={'learning_rate': learning_rate,
            'n_in':3200,'n_out':2,'n_hidden':900}
    return params
