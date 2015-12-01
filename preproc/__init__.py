import utils
import binary

def transform_files(in_path,out_path,transform,dirs=False):
    utils.make_dir(out_path)
    if(dirs):
        names=utils.get_dirs(in_path)
    else:
        names=utils.get_files(in_path)
    for name in names:
        full_in_path=in_path+name
        full_out_path=out_path+name
        transform(full_in_path,full_out_path)

def binary_to_raw(in_path,out_path):
    raw_action=binary.read_binary(in_path)
    out_path=out_path.replace(".bin",".raw")
    print(out_path)
    utils.save_object(raw_action,out_path) 
