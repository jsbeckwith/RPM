import numpy as np
from matplotlib import pyplot as plt

data = np.load("distribute_nine/RAVEN_1_train.npz")
image = data["image"].reshape(16, 160, 160)
answer = data["target"]
# meta_target = data["meta_target"]
# meta_matrix = data["meta_matrix"]
print("ANSWER: ", answer)
# print("META-TARGET: ", meta_target)
# print("META-MATRIX: ", meta_matrix)

matrix_axes=[]
matrix=plt.figure()

## show matrix
for matrix_ind in range(0, 8):
    matrix_axes.append(matrix.add_subplot(3, 3, (matrix_ind + 1)))
    plt.imshow(image[matrix_ind, :, :], cmap='gray', vmin=0, vmax=255)

for ax in matrix_axes:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

matrix.tight_layout() 
plt.show()

answers=plt.figure()
answer_axes=[]

## show answers
for answer_ind in range(8, 16):
    answer_axes.append(answers.add_subplot(2, 4, (answer_ind - 7)))
    plt.imshow(image[answer_ind, :, :], cmap='gray', vmin=0, vmax=255)

for ax in answer_axes:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

answers.tight_layout() 
plt.show()

""" print(data["target"])
for i in range(16):
    print(image[i, :, :])
    plt.imshow(image[i, :, :], cmap='gray', vmin=0, vmax=255)
    plt.show() """