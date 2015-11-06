import pandas as pd 
import utils

class Action(object):
	def __init__(self,images):
		self.orginal_shape=images[0].shape
		self.images=[img.flatten() for img in images]
		self.dim=self.images[0].shape[0]
		self.length=len(self.images)

	def __str__(self):
		return "Action - length:" +str(self.length) + " dim:" + str(self.dim)


def read_action(action_path):
    images=utils.read_img_dir(action_path)
    return Action(images)

def get_action_name(action_path):
    return action_path.split("/")[-1]

if __name__ == "__main__":
    path="../"#"/home/user/cls/"
    action_path=path+"small/a01_s01_e01_sdepth"
    action=read_action(action_path)
    print(action)