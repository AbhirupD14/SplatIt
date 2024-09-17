import cv2

class GSPLAT:
    #Initialize the class, we need to take in the recording path from the user and a destination path in which the output file
    #will be placed
    def __init__(self, recording_path, destination_path):
        self.path = recording_path
        self.dest = destination_path
        self.blacklist = ["frame_extract","point_cloud_gen", "gaussians", "rast"]
    
    #Check if the method that is being used is internal or not
    #If it is internal, raise an error
    #Else allow the user to call the method
    def __getattribute__(self, name):
        if name in self.blacklist:
            raise AttributeError(f"{name} is not accessible!")
        else:
            return super(GSPLAT, self).__getattribute__(name)
    
    #Extract frames from the video
    def frame_extract(self):
        pass
    
    #Generate the point cloud
    def point_cloud_gen(self):
        pass

    #Generate gaussian splats from the point cloud
    def gaussians(self):
        pass

    #Rasterize splats
    def rast(self):
        pass

    #Generate output file
    def generate(self):
        pass

