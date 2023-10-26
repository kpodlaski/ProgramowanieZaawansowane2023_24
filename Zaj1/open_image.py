import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('../imgs/Weasel.jpg')
# plt.imshow(img)
# plt.show()

#print(img)
print(img.shape)
blue = img[:,:,0]
red = img[:,:,2]
green = img[:,:,1]

#print(np.histogram(blue,bins=255, range=(0,255)))
plt.plot(np.histogram(blue,bins=255, range=(0,255))[0], color="blue")
plt.plot(np.histogram(green,bins=255, range=(0,255))[0], color="green")
plt.plot(np.histogram(red,bins=255, range=(0,255))[0], color="red")
plt.show()
plt.title("Blue")

N, bins, patches = plt.hist(blue, label="blue")
plt.show()
plt.title("Green")
N, bins, patches =plt.hist(green, label="green")
plt.show()
plt.title("Green")
N, bins, patches =plt.hist(red,  label="red")
plt.show()

print(blue.shape)
r_th = 120
b_th = 120
g_th = 120

img_2 = np.copy(img)
img_2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# for r in (0,img_2.shape[])
cv.imshow("weasel", img)
# cv.imshow("weasel_2", img_2)
cv.imshow("value",img_2[:,:,2])
cv.imshow("saturation",img_2[:,:,1])
cv.imshow("hue",img_2[:,:,0])
plt.title("HSV hist")
plt.plot(np.histogram(img_2[:,:,0],bins=255, range=(0,255))[0], label="hue")
plt.plot(np.histogram(img_2[:,:,1],bins=255, range=(0,255))[0], label="saturation")
plt.plot(np.histogram(img_2[:,:,2],bins=255, range=(0,255))[0], label="value")
plt.legend(loc="upper left")
plt.show()

# plt.show()
cv.waitKey(0)
# closing all open windows
cv.destroyAllWindows()

