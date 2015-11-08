import re,utils
import pandas as pd
import numpy as np
#import imp
#utils =imp.load_source("utils","../utils.py")

def read_image_frame(dir_path):
    paths=utils.get_paths(dir_path,dirs=True)
    images=[]
    for cat,path in enumerate(paths):
	cat_imgs=utils.read_img_dir(path)
        cat_imgs=[standard_image(cat,img) for img in cat_imgs]
        images+=cat_imgs
    return create_img_frame(images)

def create_img_frame(images):
    labels=[ img[0] for img in images]
    flat_imgs=[ img[1] for img in images]
    return  pd.DataFrame({ 'Images':flat_imgs,
                           'Category':labels,})

def standard_image(cat,img):
    img=img.flatten()
    img/=max(img)
    img=np.reshape(img,(img.size,1))
    return cat,img

if __name__ == "__main__":
    path="../../imgs"
    df=read_image_frame(path)
    print(df.head())
