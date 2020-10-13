import numpy as np
from matplotlib import pyplot as plt

data = np.load("in_distribute_four_out_center_single/RAVEN_0_train.npz")
image = data["image"].reshape(16, 160, 160)

rows = 4
cols = 4
axes=[]
fig=plt.figure()

""" for a in range(rows*cols):
    axes.append( fig.add_subplot(rows, cols, a+1) )
    plt.imshow(image[a, :, :], cmap='gray', vmin=0, vmax=255) """

for matrix_ind in range(1, 9):
    axes.append(fig.add_subplot(3, 3, (matrix_ind + 1)))
    plt.imshow(image[matrix_ind, :, :], cmap='gray', vmin=0, vmax=255)

for b in axes:
    b.get_xaxis().set_visible(False)
    b.get_yaxis().set_visible(False)

fig.tight_layout() 
plt.show()

""" print(data["target"])
for i in range(16):
    print(image[i, :, :])
    plt.imshow(image[i, :, :], cmap='gray', vmin=0, vmax=255)
    plt.show() """