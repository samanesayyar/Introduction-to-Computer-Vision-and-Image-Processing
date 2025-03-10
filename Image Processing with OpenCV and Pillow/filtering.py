import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
   
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

image=cv2.imread(r"c:\Users\Samane\Desktop\CV coursera\lenna.png")
print(type(image))
#print(image.shape)
#plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#plt.show()

rows, cols,_=image.shape
print(rows,cols)
##add manual noise
noise=np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
noisy_image=image+noise
#plot_image(image,noisy_image,title_1="orginal",title_2="Image_plus_noise")

##Filtering noise(filter2d)
kernel=np.ones((2,2))/4
image_filtered_2d=cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
#plot_image(image_filtered_2d,noisy_image,title_1="image_filtered",title_2="noisy")

##Filtering noise(Gaussian Blur)// ksize and sigma should be odd
image_filter_guassian=cv2.GaussianBlur(src=noisy_image,ksize=(37,37), sigmaX=23, sigmaY=23)
#plot_image(image_filter_guassian,noisy_image,title_1="image_filtered",title_2="noisy")


img_gray=cv2.imread(r"c:\Users\Samane\Desktop\CV coursera\zelda.png",cv2.IMREAD_GRAYSCALE)
print(img_gray.shape)
#plt.imshow(img_gray,cmap='grey')
#plt.show()

# Filters the images using GaussianBlur on the image with noise using a 3 by 3 kernel 
img_gray = cv2.GaussianBlur(img_gray,(3,3),sigmaX=0.5,sigmaY=0.5)
# Renders the filtered image
#plt.imshow(img_gray ,cmap='gray')
#plt.show()


##edge
ddepth=cv2.CV_16S
grade_x=cv2.Sobel(src=img_gray,ddepth=ddepth, dx=1,dy=0,ksize=3)
grade_y=cv2.Sobel(src=img_gray,ddepth=ddepth, dx=0,dy=1,ksize=3)
#plt.imshow(grade_x,cmap="gray")
#plt.show()
abs_grade_x=cv2.convertScaleAbs(grade_x)
abs_grade_y=cv2.convertScaleAbs(grade_y)
#print(abs_grade_x)
grad=cv2.addWeighted(abs_grade_x,0.5,abs_grade_y,0.5,0)
#plt.imshow(grad,cmap="gray")
#plt.show()

##Median
img_gray2=cv2.imread(r"c:\Users\Samane\Desktop\CV coursera\cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
image_filtered_med=cv2.medianBlur(img_gray2,5)

##Threshold
ret,outs=cv2.threshold(src=img_gray2,thresh=0,maxval=255,type=cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
print(ret,outs)
plt.imshow(outs,cmap="gray")
plt.show()

