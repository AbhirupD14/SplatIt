from gSplat import GSPLAT as gs
import matplotlib.pyplot as plt
import numpy as np
G = gs('C:/Users/abhio/Documents/SplatIt/videos/pole.MOV')
G.frame_extract()
# G.frame_to_depth()

inp = input("0 for no visualization, 1 for plot: ")
if inp == 1:
    plt.figure(figsize=(10, 6))
    plt.imshow(G.getDepthMap()[0], cmap='plasma')
    plt.colorbar(label='Depth Value')
    plt.title('Depth Map Visualization')
    plt.axis('off')
    plt.show()