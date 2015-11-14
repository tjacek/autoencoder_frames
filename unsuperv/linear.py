import numpy as np
import sklearn.decomposition as decomp
import utils

class PcaReduction(object):
    def __init__(self,imgs):
        self.pca=decomp.PCA()
        imgs=list(imgs)
        imgs=np.array(imgs)
        print(imgs.shape)
        self.pca.fit(imgs)

    def transform(self,x):
        return self.pca.transform(x)

    def get_components(self,k):
        return self.pca.components_[0:k]

    def visualize(self,k,new_shape=(80,40)):
    	princ_comps=self.get_components(k)
    	princ_comps=utils.unflat_images(princ_comps,new_shape)
    	princ_comps=utils.named_images("pca",princ_comps)
        return princ_comps