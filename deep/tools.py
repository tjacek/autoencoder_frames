import numpy as np
import theano
import theano.tensor as T
import ConfigParser
import deep

def read_hyper_params(config_path):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    conf=config.items("Float")
    conf=[ (pair_i[0],float(pair_i[1])) for pair_i in conf]
    conf_f=dict([ list(pair_i) for pair_i in conf])
    conf=config.items("String")
    conf_s=dict([ list(pair_i) for pair_i in conf])
    return dict(conf_f, **conf_s)

def construct_functions(free_vars,model,py_x,loss,learning_rate):
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

def get_loss_function(free_vars,py_x):
    return T.mean(T.nnet.categorical_crossentropy(py_x,free_vars.y))

if __name__ == "__main__":
    in_path="/home/user/af/test.conf"
    print(read_hyper_params(in_path))