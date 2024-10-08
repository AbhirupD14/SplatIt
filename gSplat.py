import os
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor

class GSPLAT:
    #Initialize the class, we need to take in the recording path from the user and a destination path in which the output file
    #will be placed
    def __init__(self, recording_path, destination_path):
        self.path = recording_path
        self.dest = destination_path
        self.blacklist = ["frame_extract","point_cloud_gen", "gaussians", "rast"]
        self.pc = ''
    
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
        frame_data = cv2.VideoCapture(self.path)
        extracted, image = frame_data.read()
        count = 0

        while extracted:
            cv2.imwrite("frame%d.jpg" % count, image) #will create a folder called frames to save each image to
            print('Read next frame: ', extracted)
            count += 1
            depth_map = self.frame_to_depth(image)
            self.point_cloud_gen(depth_map) #Generate a point cloud for every map
        frame_data.release()
    
    def frame_to_depth():
        all_frames = os.listdir('/frames')
        depthMaps = []
        #Check if a GPU is available, if so use it else use cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the MiDaS model and set to evaluation mode
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS").to(device)
        model.eval()
        
        # Preprocessing for the model
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
        for frame in all_frames:
            input_img = transform(frame).unsqueeze(0).to(device)
            
            # Make depth prediction
            with torch.no_grad():
                prediction = model(input_img)
            
            # Convert prediction to depth map and normalize
            depth_map = prediction.squeeze().cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize
            depthMaps.append(depth_map)
        return depthMaps
    
    #Generate the point cloud
    def point_cloud_gen(self, depth_map):
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

