import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
img = cv2.imread("rice.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# This method gets a threshold value as a parameter and then return output image as a binary image. Pixel intensities
# are specified via threshold value. But output image is very noisy.
_, threshold = cv2.threshold(gray, 112, 255, cv2.THRESH_BINARY)

# In this method I used a morphological operation (erosion) for two purpose: 1) To get rid of the noise. 2) To drive
# external contours apart.
kernel = np.ones((3, 3))
erosion = cv2.morphologyEx(threshold, cv2.MORPH_ERODE, kernel=kernel)

# Finding external contours. (This method is similar to bwlabel method from matlab)
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours into to original image.
img2 = img.copy()
cnt = cv2.drawContours(img2, contours, -1, (255, 0, 0), 1)

# Display all of the conclusions in a one window.
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title("Original"), axs[0, 0].axis('off')

axs[0, 1].imshow(threshold, cmap='gray')
axs[0, 1].set_title("Threshold"), axs[0, 1].axis('off')

axs[1, 0].imshow(erosion, cmap='gray')
axs[1, 0].set_title("Erosion"), axs[1, 0].axis('off')

axs[1, 1].imshow(cnt, cmap='gray')
axs[1, 1].set_title("Find/Draw Contours"), axs[1, 1].axis('off')

plt.text(-425, 325, 'And the number of rice seeds is: {}'.format(len(contours)), fontsize=18)
plt.show()
