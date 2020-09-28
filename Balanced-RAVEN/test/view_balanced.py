import numpy as np
from matplotlib import pyplot as plt

data = np.load("left_center_single_right_center_single/RAVEN_0_train.npz")
image = data["image"].reshape(16, 160, 160)
print(data["target"])
for i in range(16):
    print(image[i, :, :])
    plt.imshow(image[i, :, :], cmap='gray', vmin=0, vmax=255)
    plt.show()