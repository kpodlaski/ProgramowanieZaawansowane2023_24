import random
import struct

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_mnist_labels(file_path):
    labels = None
    #open file
    file = open(file_path,"rb")
    #magic number
    file.read(4)
    size = struct.unpack(">i",file.read(4))[0]
    labels = []
    for i in range(size):
        label = struct.unpack(">B",file.read(1))
        labels.append(label)
    file.close()
    return labels


def read_mnist_images(file_path):
    images = None
    file = open(file_path, "rb")
    # magic number
    file.read(4)
    size, rows, cols = struct.unpack(">iii", file.read(12))
    print (size, rows, cols)
    images = []
    for i in range(size):
        image = []
        for x in range(rows*cols):
            pixel = struct.unpack(">B", file.read(1))
            image.append(pixel)
        images.append(np.array(image, dtype='float'))
    file.close()
    return np.array(images), rows, cols


test_labels = read_mnist_labels("../data/emnist-balanced-test-images-idx3-ubyte")
train_labels = read_mnist_labels("../data/emnist-balanced-test-labels-idx1-ubyte")
print(len(test_labels))
print(len(train_labels))
images, rows, cols = read_mnist_images("../data/emnist-balanced-test-images-idx3-ubyte")
print(images.shape)
r =3
c = 5
fig, axis = plt.subplots(r,c)
for x in range(r):
    for y in range(c):
        id = random.randrange(len(test_labels))
        axis[x,y].imshow(images[id].reshape((rows,cols)), cmap='gray')
        axis[x,y].set_title(str(test_labels[id]))
plt.show()