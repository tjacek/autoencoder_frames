import numpy as np
import sklearn.decomposition as decomp

class PcaReduction(object):
    def __init__(self,imgs):
        self.pca=decomp.PCA()
        imgs=list(imgs)
        imgs=np.array(imgs)
        print(imgs.shape)
        self.pca.fit(imgs)

    def transform(self,x):
        return self.pca.transform(x)       