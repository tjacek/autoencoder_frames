import numpy as np
import sklearn.decomposition as decomp
import utils

class PcaReduction(object):
    def __init__(self,imgs):
        imgs=list(imgs)
        imgs=np.array(imgs)
        dim=imgs.shape[1]
        print(dim)
        self.pca=decomp.PCA(n_components=dim)
        self.pca.fit(imgs)

    def transform(self,x):
        return self.pca.transform(x)

    def inverse_transform(self,x):
        return self.pca.inverse_transform(x)

    def get_components(self,k):
        return self.pca.components_[0:k]

    def visualize(self,k,new_shape=(80,40)):
    	princ_comps=self.get_components(k)
    	princ_comps=utils.unflat_images(princ_comps,new_shape)
    	princ_comps=utils.named_images("pca",princ_comps)
        return princ_comps

class PcaDecorator(object):
    def __init__(self,pca,autoencoder):
        self.pca=pca
        self.autoencoder=autoencoder

    def train(self,x):
        x_proj=self.pca.transform(x)
        return self.autoencoder.train(x_proj)

    def transform(self,x):
        x_proj=self.pca.transform(x)
        return self.autoencoder.transform(x_proj)

    def get_image(self,x):
        x_proj=self.pca.transform(x)
        x_hid=self.autoencoder.get_image(x_proj)
        return self.pca.inverse_transform(x_hid)

    def get_model(self):
        ae_model=self.autoencoder.model
        return PcaModel(ae_model,self.pca)

class PcaModel(object):
    def __init__(self,ae_model,nn_model):
        self.ae_model=ae_model
        self.nn_model=nn_model
