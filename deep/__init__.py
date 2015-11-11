import utils
import theano
import theano.tensor as T
import numpy as np

class LayerModel(object):
    def __init__(self,W,b):
        self.W=W
        self.b=b

    def get_params(self):
        return [self.W,self.b] 

class LabeledImages(object):
    def __init__(self):
        self.X=T.matrix('x')
        self.y=T.lvector('y')

    def get_vars(self):
        return [self.X,self.y]

class Classifier(object):
    def __init__(self,free_vars,model,train,test,prob_dist):
        self.free_vars=free_vars
        self.model=model
        self.test=test
        self.prob_dist=prob_dist
        self.train=train

class RandomNum(object):
    def __init__(self):
        self.rng = np.random.RandomState(123)

    def random_matrix(self,n_x,n_y):
        bound=np.sqrt(6. / (n_x + n_y))
        raw_matrix=self.rng.uniform(-bound,bound,n_x*n_y)
        matrix=np.array(raw_matrix,dtype=theano.config.floatX)
        return np.reshape(matrix,(n_x,n_y))

    def random_vector(self,n_x):
        bound=np.sqrt(6. / n_x)
        raw_vector=self.rng.uniform(-bound,bound,n_x)
        return np.array(raw_vector,dtype=theano.config.floatX)

def create_layer(shape,rand,postfix=""):
    init_weights=rand.random_matrix(shape[0],shape[1])
    W=make_var(init_weights,"W"+postfix)
    init_bias=rand.random_vector(shape[1])
    b=make_var(init_bias,"b"+postfix)
    return LayerModel(W,b)

def make_var(value,name):
    return theano.shared(value=value,name=name,borrow=True)

def compute_updates(loss, params, learning_rate=0.05):
    gparams = [T.grad(loss, param) for param in params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]
    return updates

def learning_iter(img_frame,cls,
                  n_epochs=1000,batch_size=10):
    X_b=img_frame['Images']
    y_b=img_frame['Category']

    n_train_batches=get_number_of_batches(y_b,batch_size)#len(y_b)
    print '... training the model'
    timer = utils.Timer()
    for epoch in xrange(n_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            x_i,y_i=get_batch(batch_index,X_b,y_b,batch_size)
            #print(y_i.shape)
            c.append(cls.train(x_i,y_i))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    timer.stop()
    print("Training time %d ",timer.total_time)
    return cls

def check_prediction(img_frame,cls):
    X_b=img_frame['Images']
    y_b=img_frame['Category']
    def standard_img(i): 
        return get_batch(i,X_b,y_b,1)[0]
    x_std=[standard_img(i) for i in range(len(y_b))]
    y_pred=[cls.test(x_i) for x_i in x_std]
    prob_vecs=[cls.prob_dist(x_i) for x_i in x_std]
    compr=[ y_i==y_p for y_i,y_p in zip(y_b,y_pred)]
    compr=[int(c[0]) for c in compr]
    print(prob_vecs)
    print(compr)
    acc=np.mean(compr)
    print(acc)
    return acc

def get_number_of_batches(dataset,batch_size):
    n_batches=len(dataset)/batch_size
    if((len(dataset) % batch_size) != 0):
        n_batches+=1 
    return n_batches

def get_batch(i,x,y,batch_size):
    begin=i*batch_size
    end=(i+1)*batch_size
    x_i=x[begin:end].tolist()
    x_i=np.array(x_i)
    y_i=y[begin:end].tolist()
    y_i=np.array(y_i)
    return x_i,y_i