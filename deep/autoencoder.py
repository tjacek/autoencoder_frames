import numpy as np
import theano
import theano.tensor as T
import deep,utils
import scipy.misc

class AutoencoderModel(object):
    def __init__(self,W,b,b_prime):
        self.W=W
        self.b=b
        self.b_prime=b_prime
        self.W_prime = W.T

    def get_params(self):
        return [self.W, self.b, self.b_prime]

class AutoEncoder(object):
    def __init__(self,free_vars,model,
    	         train,test,get_image,rand):
    	self.free_vars=free_vars
        self.model=model  
        self.train=train
        self.test=test
        self.get_image=get_image
        self.rand=rand

def built_ae_cls():
    hyper_params=get_hyper_params()
    free_vars=deep.LabeledImages()
    model,rand= create_ae_model(hyper_params)
    train,test,get_image=create_ae_fun(free_vars,model,rand,hyper_params)
    return AutoEncoder(free_vars,model,train,test,get_image,rand)

def read_autoencoder(cls_path):
    model=utils.read_object(cls_path)
    hyper_params=get_hyper_params()
    free_vars=deep.LabeledImages()
    rand=deep.RandomNum()
    train,test,get_image=create_ae_fun(free_vars,model,rand,hyper_params)
    return AutoEncoder(free_vars,model,train,test,get_image,rand)

def create_ae_model(hyper_params):
    n_hidden=hyper_params['n_hidden']
    n_visible=hyper_params['n_visible']
    rand=deep.RandomNum()
    initial_W =rand.random_matrix(n_visible,n_hidden)
    W = deep.make_var(initial_W,'W')
    init_b=rand.random_vector(n_hidden)
    bhid = deep.make_var(init_b,'b')
    init_bvis=rand.random_vector(n_visible)
    bvis = deep.make_var(init_bvis,"bvis")
    return AutoencoderModel(W,bhid,bvis),rand

def create_ae_fun(free_vars,model,rand,hyper_params):
    learning_rate=hyper_params['learning_rate']
    corruption_level=hyper_params['corruption_level']
    tilde_x = get_corrupted_input(free_vars,corruption_level,rand)
    x=free_vars.X
    y = get_hidden_values(model,tilde_x)
    z = get_reconstructed_input(model,y)
    loss=get_crossentropy_loss(x,y,z)
    input_vars=free_vars.get_vars()
    params=model.get_params()
    updates=deep.compute_updates(loss, params, learning_rate)
    train = theano.function([x],loss,updates=updates)
    test = theano.function([x],y)
    get_image = theano.function([x],z)
    return train,test,get_image

def get_crossentropy_loss(x,y,z):
    L = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)
    return T.mean(L)

def get_corrupted_input(free_vars,corruption_level,rand):
    rng=rand.theano_rng
    x=free_vars.X
    return rng.binomial(size=x.shape, n=1,p=1 - corruption_level,
                        dtype=theano.config.floatX) * x

def get_hidden_values(model, x):
    return deep.get_sigmoid(x,model.W,model.b)

def get_reconstructed_input(model,hidden):
    return deep.get_sigmoid(hidden,model.W_prime,model.b_prime)

def get_hyper_params(learning_rate=0.05):
    params={'learning_rate': learning_rate,'corruption_level':0,
            'n_visible':3200,'n_hidden':900}
    return params

def reconstruct_images(img_frame,ae,out_path):
    utils.make_dir(out_path)
    imgs=img_frame['Images']
    #cats=img_frame['Category']
    for i,img in enumerate(imgs):
        img=np.reshape(img,(1,3200))
        rec_image=ae.get_image(img)
        img2D=np.reshape(rec_image,(80,40))
        img_path=out_path+"img"+str(i)+".png"
        print(img_path)
        scipy.misc.imsave(img_path,img2D)
