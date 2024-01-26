import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2 as cv

# Giving The Original image Directory
# Specified
src_img = ImageOps.grayscale(Image.open("imgs/page.png"))
rot_img = src_img.rotate(8)
src_img = np.array(src_img)
rot_img = np.array(rot_img)
h, w = src_img.shape

plt.title("orig image")
plt.imshow(src_img, cmap='gray')
plt.show()
plt.title("rot image")
plt.imshow(rot_img, cmap='gray')
plt.show()


print(src_img.shape)
dft_orig = np.fft.fftshift(np.fft.fft2(src_img))
dft_rot = np.fft.fftshift(np.fft.fft2(rot_img))
dft_rot[int(h/2)-10:int(h/2)+10, int(w/2)-10: int(w/2)+10]=100

plt.title("orig log intensity")
plt.imshow(np.log(abs(dft_orig)), cmap='gray')
plt.show()


plt.title("rot log intensity")
plt.imshow(np.log(abs(dft_rot)), cmap='gray')
plt.show()
plt.close()

mask = np.zeros(dft_orig.shape)

mask[:,int(w/2)-2:int(w/2)+2] = 1
plt.title("mask")
plt.imshow(mask, cmap='gray')
plt.show()
plt.close()
print(mask.shape)
print((mask*np.log(abs(dft_orig))).sum())
mask[:,int(w/2)] = 1
print((mask*np.log(abs(dft_orig))).sum())
mask[:,int(w/2)] = 1
print(sum(np.log(abs(dft_orig))).shape)



v_max = 0
a_max=-24
for angle in range (-24,26,1):
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    scale = 1
    rot_mat = cv.getRotationMatrix2D(center, angle/2, scale)
    rot_mask = cv.warpAffine(mask, rot_mat, (mask.shape[1], mask.shape[0]))
    v = (rot_mask * np.log(abs(dft_rot))).sum()
    if v<v_max:
        v_max = v
        a_max = angle/2
    print(angle/2, v)


print((mask*np.log(abs(dft_orig))).sum())
print(a_max, v_max)