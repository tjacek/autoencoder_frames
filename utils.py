import os
import os.path as io 
import timeit,pickle,re
import scipy.misc as image
import numpy as np

class Timer(object):
    def __init__(self):
        self.start_time=timeit.default_timer()

    def stop(self):
        self.end_time = timeit.default_timer()
        self.total_time = (self.end_time - self.start_time)

    def show(self):
        print("Training time %d ",self.total_time)

def get_files(path):
    all_in_dir=os.listdir(path)
    files= filter(lambda f:is_file(f,path),all_in_dir)
    files.sort()
    return files

def get_dirs(path):
    all_in_dir=os.listdir(path)
    dirs= filter(lambda f: not is_file(f,path),all_in_dir)
    dirs.sort()
    return dirs

def get_paths(dir_path,dirs=False):
    if(dirs):
        files=get_dirs(dir_path)
    else:
        files=get_files(dir_path)
    files=["/" + f for f in files]
    return append_path(dir_path,files)

def read_file(path):
    file_object = open(path,'r')
    lines=file_object.readlines()  
    file_object.close()
    return lines

def is_file(f,path):
        return io.isfile(io.join(path,f))

def make_dir(path):
    if(not os.path.isdir(path)):
	os.system("mkdir "+path)

def get_name(path):
    return path.split("/")[-1]

def array_to_txt(array):
    return reduce(lambda x,y:x+str(y),array,"")

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def save_string(path,string):
    file_str = open(path,'w')
    file_str.write(string)
    file_str.close()

def read_object(path):
    file_object = open(path,'r')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def append_path(path,files):
    return map(lambda f: path+f,files)

def replace_sufix(sufix,files):
    return map(lambda s:s.replace(sufix,""),files)

def read_images(files):
    return [image.imread(f) for f in files]

def read_img_dir(action_path):
    all_files=get_files(action_path)
    all_files=append_path(action_path+"/",all_files)
    return read_images(all_files)

def to_txt_file(path,array):
    txt=array_to_txt(array)
    save_string(path,txt)

def change_postfix(filename,old=".seq",new=".lb"):
    return filename.replace(old,new)

def save_images(path,act_imgs):
    print(path)
    make_dir(path)
    for name,img in act_imgs:
        save_img(path+name,img)

def save_img(path,img):
    full_path=path+".png"
    image.imsave(full_path,img)

def unflat_images(flat_img,new_shape):
    return [np.reshape(img,new_shape) for img in flat_img]

def named_images(name,imgs):
    return [ (name+str(i),img) for i,img in enumerate(imgs)]

def to_labeled_file(path,labeled_vectors):
    lb=""
    for instance,cat in labeled_vectors:
        line=""
        for cord_i in instance:
            line+=str(cord_i)+","
        line+="#"+str(cat)+"\n"
        lb+=line
    save_string(path,lb)

def to_2D(img,dim=(80,40)):
    return np.reshape(img,dim)