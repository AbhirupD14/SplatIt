import os
import torch
import cv2
import PIL
import torchvision.transforms as transforms 
# import open3d as o3d
class GSPLAT:
    #Initialize the class, we need to take in the recording path from the user and a destination path in which the output file
    #will be placed
    def __init__(self, recording_path, destination_path = None):
        self.path = recording_path
        if destination_path == None:
            self.dest = os.getcwd()
        else:
            self.dest = destination_path
        self.blacklist = ["frame_extract","point_cloud_gen", "gaussians", "rast"]
        self.pc = ''
        self.depthMaps = []
    
    #Check if the method that is being used is internal or not
    #If it is internal, raise an error
    #Else allow the user to call the method
    # def __getattribute__(self, name):
    #     if name in self.blacklist:
    #         raise AttributeError(f"{name} is not accessible!")
    #     else:
    #         return super(GSPLAT, self).__getattribute__(name)

    #Extract frames from the video
    def frame_extract(self):
        if os.path.isdir('frames'):
            return
        count = 0
        frame_data = cv2.VideoCapture(self.path)
        frames = frame_data.get(cv2.CAP_PROP_FRAME_COUNT)  
        os.mkdir('frames') #make frames directory
        os.chdir('frames') #change to the new director
        print('ATTEMPTING TO READ ALL FRAMES...')
        while count < frames + 1:
            if count % 10 == 0:
                extracted, image = frame_data.read()
                if not extracted:
                    break
                cv2.imwrite("frame%d.jpg" % count, image) #write to folder
            # depth_map = self.frame_to_depth(image)
            # self.point_cloud_gen(depth_map) #Generate a point cloud for every map
            count += 1
        os.chdir('..') #get out of directory
        frame_data.release()
        print("READ ALL FRAMES SUCCESSFULLY")

    
    def frame_to_depth(self):
        # List all frame files in the /frames directory
        all_frames = os.listdir('frames')
        
        # Check if a GPU is available, if so use it else use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the MiDaS model and set it to evaluation mode
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS").to(device)
        model.eval()

        # Load MiDaS transformation for preprocessing
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
        # for frame in all_frames:
        frame = all_frames[0]
        # Load each frame as an image
        frame_path = os.path.join('frames', frame)
        input_img = PIL.Image.open(frame_path)

        # Apply transformation and prepare input tensor
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),        # Converts PIL image to a tensor and scales pixel values to [0, 1]
            transforms.Lambda(lambda img: img / 255.0),  # Scale again if needed, though ToTensor() already scales
        ])

        input_img = transform(input_img).unsqueeze(0).to(device)

        # Make depth prediction
        with torch.no_grad():
            prediction = model(input_img)

        # Convert prediction to depth map and normalize
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize

        self.depthMaps.append(depth_map)
        print(len(self.depthMaps))
        #Delete the frames directory as we won't need them anymore
        # if os.path.exists('frames'):
        #     shutil.rmtree('frames')
    
        #Generate the point cloud
    def point_cloud_gen(self):
        # self.frame_extract()
        # self.frame_to_depth()
        # # Load the depth map
        # depth_map = cv2.imread(self.depthMaps[0], cv2.IMREAD_UNCHANGED)

        # # Define image size
        # height, width = depth_map.shape
        # fov = 60  # Assumed field of view in degrees
        # fx = fy = width / (2 * np.tan(np.deg2rad(fov) / 2))  # Focal length
        # cx, cy = width / 2, height / 2  # Camera center

        # # Initialize point cloud
        # points = []

        # # Generate 3D points
        # for v in range(height):
        #     for u in range(width):
        #         z = depth_map[v, u] / 1000  # Convert depth value to meters
        #         if z == 0:  # Skip invalid depth
        #             continue
        #         x = (u - cx) * z / fx
        #         y = (v - cy) * z / fy
        #         points.append([x, y, z])

        # # Create Open3D point cloud object
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

        # return point_cloud
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

    def getDepthMap(self):
        return self.depthMaps


