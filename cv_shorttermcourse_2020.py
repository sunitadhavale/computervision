# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:15:58 2020

@author: ASUS
"""

#--------------------------CV Lecture #01 introduction-------------------------------------

import cv2 
print(cv2.__version__)

from PIL import Image
Img=Image.open('E:/computervision/Lena.png') #read Image
import matplotlib.pyplot as plt
plt.imshow(Img) #display image

import numpy as np
print(np.shape(Img)) #see the size (512, 512, 3)
#convert into numpy array
img1=np.asarray(Img)
#read only one channel
plt.imshow(img1[:,:,1],cmap='gray')
#see pixel values
img1[1:10,1:10,1]
#see each channel of RGB color image
r,g,b=Img.split()
img1=np.asarray(g)
plt.imshow(Img)
plt.imshow(img1)

from skimage.color import rgb2hsv
hsvimg=rgb2hsv(Img)
print(img1[1:10,1:10])
print(hsvimg[1:10,1:10,0])
print(hsvimg[1:10,1:10,1])
print(hsvimg[1:10,1:10,2])
#if RGB=255 then H=S=0 and V=1

def histogram(im):
    h = np.zeros(255)
    for row in im.shape[0]:
        for col in im.shape[1]:
            val = im[row, col]
            h[val] += 1

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("E:/computervision/Lena.png")
#plot a histogram
histogram_image = cv2.calcHist([img],[0],None,[256],[0,256])
hist,bins = np.histogram(img.ravel(),256,[0,256])
np.shape(hist)  ## (256,)
hist[1:10]  ## array([  0,   0,  11,  65, 111, 164, 261, 315, 431], dtype=int64)
#flaten the histogram 
plt.hist(img.ravel(),256,[0,256]) 
plt.show()
#view color channels
color = ['b','g','r']
#seperate the colors and plot the histogram
for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color = col)
    plt.xlim([0,256])
plt.show ()

np.mean(img1)
#do homework

#--------------------------CV Lecture #02 image transformations-------------------------------------

from PIL import Image
Img=Image.open('E:/computervision/Lena.png') #read Image
import matplotlib.pyplot as plt
plt.imshow(Img) #display image

import numpy as np
print(np.shape(Img)) #see the size (512, 512, 3)
#convert into numpy array
img1=np.asarray(Img)

#RGB to Gray
import cv2
grayscale = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1=np.asarray(grayscale)
print(img1.shape)
plt.imshow(img1,cmap='gray')

#histogram equalization

equ = cv2.equalizeHist(img1)
img2 = np.hstack((img1,equ)) #stacking images side-by-side
plt.imshow(img2,cmap='gray')
img1[1:5,1:5]
equ[1:5,1:5]

hist,bins = np.histogram(img1.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')

hist1,bins1 = np.histogram(equ.flatten(),256,[0,256])
cdf1 = hist1.cumsum()
cdf_normalized1 = cdf1 * hist1.max()/ cdf1.max()
plt.plot(cdf_normalized1, color = 'b')

plt.hist(img1.flatten(),256,[0,256], color = 'r')
plt.hist(equ.flatten(),256,[0,256], color = 'b')

#gaussian filter

kernel = np.ones((5,5),np.float32)/25
kernel
#filtered = cv2.filter2D(img1,-1,kernel)
filtered = cv2.GaussianBlur(img1,(5,5),0)
img2 = np.hstack((img1,filtered))
plt.imshow(img2,cmap='gray')

# FFT of Image

fftofimg=np.fft.fft2(img1)
img1[0:4,0:4]
fftofimg[0:4,0:4]
fftofimg[0:4,0:4]=0
ifftofimg=np.fft.ifft2(fftofimg)
img2 = np.hstack((img1,ifftofimg))
plt.imshow(np.real(img2),cmap='gray')

fshift = np.fft.fftshift(fftofimg)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.imshow(magnitude_spectrum, cmap = 'gray')

#--------------------------CV Lecture #03 image thresholding-------------------------------------

from PIL import Image
Img=Image.open('E:/computervision/jiraffe.png') #read Image
import matplotlib.pyplot as plt
plt.imshow(Img) #display image

import numpy as np
print(np.shape(Img)) #see the size (512, 512, 3)
#convert into numpy array
img1=np.asarray(Img)

#RGB to Gray
import cv2
grayscale = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1=np.asarray(grayscale)
print(img1.shape)
plt.imshow(img1,cmap='gray')

ret,thresh1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
img2 = np.hstack((img1,thresh1))
plt.imshow(img2,cmap='gray')

hist,bins = np.histogram(img1.flatten(),256,[0,256])
plt.plot(hist, color = 'b')

th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
img2 = np.hstack((img1,th2, th3))
plt.imshow(img2,cmap='gray')

# Otsu's thresholding
ret2,th4 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2
# Otstru's thresholding after Gaussian filtering
blur= cv2.GaussianBlur(img1,(5,5),0)
ret3,th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img2 = np.hstack((img1,th4, th5))
plt.imshow(img2,cmap='gray')


#--------------------------CV Lecture #04 edge detection-------------------------------------

from PIL import Image
Img=Image.open('E:/computervision/jiraffe.png') #read Image

from skimage.data import camera
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h, farid_v, farid_h

# image = camera()
# edge_roberts = roberts(image)
# edge_sobel = sobel(image)

# fig, (ax0, ax1) = plt.subplots(ncols=2)

# ax0.imshow(edge_roberts, cmap=plt.cm.gray)
# ax0.set_title('Roberts Edge Detection')
# ax0.axis('off')

# ax1.imshow(edge_sobel, cmap=plt.cm.gray)
# ax1.set_title('Sobel Edge Detection')
# ax1.axis('off')

# plt.tight_layout()

import matplotlib.pyplot as plt
plt.imshow(Img) #display image

import numpy as np
print(np.shape(Img)) #see the size (512, 512, 3)
#convert into numpy array
img1=np.asarray(Img)

#RGB to Gray
import cv2
grayscale = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1=np.asarray(grayscale)
print(img1.shape)
plt.imshow(img1,cmap='gray')

#image gradients/edge operators

edge_roberts = roberts(img1)
edge_sobel = sobel(img1)
#edge_prewitt = prewitt(img1)
plt.imshow(edge_roberts, cmap='gray')
plt.imshow(edge_sobel, cmap='gray')
#plt.imshow(edge_prewitt, cmap='gray')
a=img1[100:103,100:103]
print(roberts(a))

#laplacian filter
from skimage.filters import laplace
im_laplace = laplace(img1,ksize=3, mask=None)
plt.imshow(im_laplace,cmap='gray')
im_laplace = laplace(img1,ksize=5, mask=None)
plt.imshow(im_laplace,cmap='gray')#laplacian filter
from skimage.filters import laplace
im_laplace = laplace(img1,ksize=3, mask=None)
plt.imshow(im_laplace,cmap='gray')
im_laplace = laplace(img1,ksize=5, mask=None)
plt.imshow(im_laplace,cmap='gray')

#canny edge detection
from skimage import feature
edges = feature.canny(img1, sigma=3)
plt.imshow(edges, cmap='gray')


#--------------------------CV Lecture #05 Harris corner detection-------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
filename = 'E:/computervision/jiraffe.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
#dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()

#--------------------------CV Lecture #06 SIFT and HOG -------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
filename = 'E:/computervision/jiraffe.png'
img = cv2.imread(filename)
data=img
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),None)
plt.imshow(cv2.drawKeypoints(img, kp, None))

#--------------------------CV Lecture #07 Binary shape analysis, connectedness, object labeling and counting-------------------------------------------------
#4/8 connected neighbours
import numpy as np
x = np.eye(3).astype(int)
print(x)
from skimage.measure import label
print(label(x, connectivity=1))
print(label(x, connectivity=2))
print(label(x, background=-1))
x = np.array([[1, 0, 0],
...                     [1, 1, 5],
...                     [0, 0, 0]])
print(x)
print(label(x))

# X=[[1 0 0]
#  [0 1 0]
#  [0 0 1]]

# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]

# [[1 2 2]
#  [2 1 2]
#  [2 2 1]]

# [[1 0 0]
#  [1 1 2]
#  [0 0 0]]


#--------------------------CV Lecture #08 Boundary tracking procedures, active contours-------------------------------------------------


import math 
import numpy as np 
import pandas as pd 
from skimage.draw import ellipse 
from skimage.measure  import label, regionprops, regionprops_table 
from skimage.transform import rotate 
import matplotlib.pyplot as plt

image = np.zeros((600, 600)) 
plt.imshow(image,cmap="gray") 
rr, cc = ellipse(300, 350, 100, 220) 
image[rr, cc] = 1 
plt.imshow(image,cmap="gray") 
image = rotate(image, angle=15, order=0) 
rr, cc = ellipse(100, 100, 60, 50) 
image[rr, cc] = 1 
plt.imshow(image,cmap="gray") 


label_img = label(image) 
regions = regionprops(label_img)
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)
    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)
ax.axis((0, 600, 600, 0))
plt.show()

#https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
props = regionprops_table(label_img, properties=('centroid',  'orientation', 'major_axis_length', 'minor_axis_length'))
pd.DataFrame(props)


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def circle_points(resolution, center, radius):
#Generate points which define a circle on an image
# Centre refers to the centre of the circle
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)
#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    return np.array([c, r]).T


img1 = np.zeros((600, 600)) 
plt.imshow(image,cmap="gray") 

# Exclude last point because a closed path should not have duplicate points
points = circle_points(200, [80, 250], 80)[:-1]
#fig, ax = image_show(img1)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
import skimage.segmentation as seg
snake = seg.active_contour(img1, points)
fig, ax = image_show(img1)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

#--------------------------CV Lecture #09 Boundary descriptors, chain codes, Fourier descriptors, region descriptors, moments-------------------------------------------------


#--------------------------CV Lecture #10 Hough Transform-------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
filename = 'E:/computervision/sudoku.png'
img = cv2.imread(filename)
cv2.imshow("original image", img)  
dst = cv2.Canny(img, 50, 200, None, 3)
cv2.imshow("canny edge detected", dst)
lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)        
        