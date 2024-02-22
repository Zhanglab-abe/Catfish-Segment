import matplotlib.pyplot as plt
from skimage.io import imread

# Load the labeled image
image = imread('/home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/test/data_raw/annotations/cl8i0kllo6e8m070e4c4sbr6w.png')

# Visualize the labeled image with a color map
plt.imshow(image, cmap='nipy_spectral')
plt.colorbar()
plt.show()




