import numpy as np

class PointCloud(object):
    def __init__(self, points):
        self.points = points
        self.point_dim=3

    def find_max(self):
    	return self.find_extremum(max)

    def find_min(self):
    	return self.find_extremum(min)

    def find_extremum(self,fun):
        indexes=range(self.point_dim)
        return [fun(self.get_dim(i)) for i in indexes]

    def get_dim(self,i):
        return [point[i] for point in self.points]

    def apply(self,fun):
    	self.points=[fun(point) for point in self.points]

    def rescale(self,dim):
        new_points=[]
        for point in self.points:
            new_point=[]
            for i,dim_i in enumerate(dim):
                new_point.append(point[i]/float(dim_i))
            new_points.append(new_point)
        self.points=new_points

    def find_extreme_point(self,k,max_pc=True):
        value=self.points[0][k]
        index=0
        for i,point in enumerate(self.points):
            if(max_pc):
                grt= bool(point[k]>value)
            else:
                grt= bool(point[k]<value)
            if(grt):
                value=point[k]
                index=i
        return self.points[index]

    def to_array(self):
        return np.array(self.points)

    def to_img(self,dim,proj=None):
    	if(proj==None):
    		proj=ProjectionXZ()
    	img=proj.get_img(dim)
    	print(img.shape)
    	for point in self.points:
    		proj.apply(point,img,True)
    	return img

class PointCloud2D(PointCloud):
    def __init__(self,points):
        PointCloud.__init__(self, points)
        self.point_dim=2

def create_point_cloud(array,pc2D=False):
    width=array.shape[0]
    height=array.shape[1]
    points=[]
    if(pc2D):
        get_point=get_point2D
    else:
        get_point=get_point3D
    for x_i in range(width):
        for y_j in range(height):
            z=array[x_i][y_j]
            if(z!=0):
		        points.append(get_point(x_i,y_j,z))
    if(pc2D):
        return PointCloud2D(points)
    return PointCloud(points)

def get_point2D(x,y,z):
    return np.array((x,y))

def get_point3D(x,y,z):
    return np.array((x,y,z))

def show_points(point):
	print(point)
	return point

class ProjectionXY(object):
    def get_img(self,dim):
        img_dim=(dim[0]+1,dim[1]+1)
        return np.zeros(img_dim)

    def apply(self,point,img,binary=True):
        x,y,z=point
        x=int(x)
        y=int(y)
        if(binary):
            img[x][y]=100
        else:
            img[x][y]=z

class ProjectionXZ(object):
    def get_img(self,dim):
        img_dim=(dim[0]+1,dim[2]+3)
        return np.zeros(img_dim)

    def apply(self,point,img,binary=True):
        x,y,z=point
        x=int(x)
        z=int(z)
        if(binary):
            img[x][z]=100
        else:
            img[x][z-2]=y
            img[x][z-1]=y
            img[x][z]=y
            img[x][z+1]=y

class ProjectionYZ(object):
    def get_img(self,dim):
        img_dim=(dim[2]+2,dim[1]+1)
        return np.zeros(img_dim)

    def apply(self,point,img,binary=True):
        x,y,z=point
        z=int(z)
        y=int(y)
        if(binary):
            img[z][y]=100
        else:
            img[z][y]=x
