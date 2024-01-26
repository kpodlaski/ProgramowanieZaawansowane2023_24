import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist

# from PIL import Image, ImageDraw, ImageOps
#
# w, h = 220, 220
# shape = [(40, 40), (w - 40, h - 40)]
#
# # creating new Image object
# img = Image.new("RGB", (w, h))
# #create rectangle image
# img1 = ImageDraw.Draw(img)
# img1.rectangle(shape, fill="#000000", outline="white")
# im2 = ImageOps.grayscale(img)
# im2.save("imgs/rect.png")
src_img = imread('imgs/rect.png')
src_img = rgb2gray(imread('imgs/ldi.png'))
print(src_img.shape)
dft = np.fft.fft2(src_img)
print(dft.shape)

plt.title("orig log intensity")
plt.imshow(np.log(abs(dft)), cmap='gray')
plt.show()

phase_spectrum = np.angle(dft)
plt.title("orig Phase spectrum")
plt.imshow(phase_spectrum, cmap='gray')
plt.show()

dft_shift = np.fft.fftshift(dft)
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.title("shifted intensity")
plt.imshow((abs(dft_shift)), cmap='gray')
plt.show()

plt.title("shifted Log intensity")
plt.imshow(np.log(abs(dft_shift)), cmap='gray')
plt.show()
dft_shift_copy = dft_shift.copy()
#dft_shift[:,:50] =1
#dft_shift[:,100:] =1
# dft_shift[:100,:] =1
# dft_shift[120:,:] =1

h, w = dft_shift.shape
hw = 2
ww = 100
dft_shift[int(h/2-hw):int(h/2+hw),:int(w/2-ww)] =1
dft_shift[int(h/2-hw):int(h/2+hw),int(w/2+ww):] =1



plt.title("shifted Log intensity after mask")
plt.imshow(np.log(abs(dft_shift)), cmap='gray')
plt.show()

phase_spectrum = np.angle(dft_shift)
plt.title("shifted  phase spectrum after mask")
plt.imshow(phase_spectrum, cmap='gray')
plt.show()

out_img = np.fft.ifft2(dft_shift)
plt.title("Recovered image")
plt.imshow(abs(out_img), cmap='gray')
plt.show()
plt.show()
imsave("imgs/ldi_orig.png", abs(src_img)*255)
imsave("imgs/ldi_out.png", abs(out_img)*255)

dft_mse = (np.square(abs(dft_shift - dft_shift_copy))).mean()
dft_norm = abs(dft_shift_copy).sum()
print(dft_mse/dft_norm)
img_mse = (np.square(src_img - abs(out_img))).mean()
img_norm = src_img.sum()
print(img_mse/img_norm)