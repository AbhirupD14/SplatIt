from gSplat import GSPLAT as gs
import matplotlib.pyplot as plt
import numpy as np
G = gs('C:/Users/abhio/Documents/SplatIt/videos/pole.MOV')
# G.frame_extract()
G.frame_to_depth()
plt.figure(figsize=(10, 6))
allMaps = G.getDepthMap()
plt.imshow(allMaps[0], cmap='plasma')
plt.colorbar(label='Depth Value')
plt.title('Depth Map Visualization')
plt.axis('off')
plt.show()