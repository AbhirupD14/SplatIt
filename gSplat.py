import os
import torch
import cv2
import PIL
import open3d
import shutil
class GSPLAT:
    #Initialize the class, we need to take in the recording path from the user and a destination path in which the output file
    #will be placed
    def __init__(self, recording_path, destination_path):
        self.path = recording_path
        self.dest = destination_path
        self.blacklist = ["frame_extract","point_cloud_gen", "gaussians", "rast"]
        self.pc = ''
        self.depthMaps = []
    
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
    
def frame_to_depth(self):
    # List all frame files in the /frames directory
    all_frames = os.listdir('/frames')

    # Check if a GPU is available, if so use it else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the MiDaS model and set it to evaluation mode
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS").to(device)
    model.eval()

    # Load MiDaS transformation for preprocessing
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

    for frame_file in all_frames:
        # Load each frame as an image
        frame_path = os.path.join('/frames', frame_file)
        input_img = PIL.Image.open(frame_path)

        # Apply transformation and prepare input tensor
        input_img = transform(input_img).unsqueeze(0).to(device)

        # Make depth prediction
        with torch.no_grad():
            prediction = model(input_img)

        # Convert prediction to depth map and normalize
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize

        self.depthMaps.append(depth_map)

        #Delete the frames directory as we won't need them anymore
        if os.path.exists('/frames'):
            shutil.rmtree('/frames')
    
    #Generate the point cloud
    def point_cloud_gen(self):
        self.frame_extract()
        self.frame_to_depth()
        
    #Generate gaussian splats from the point cloud
    def gaussians(self):
        pass

    #Rasterize splats
    def rast(self):
        pass

    #Generate output file
    def generate(self):
        pass

