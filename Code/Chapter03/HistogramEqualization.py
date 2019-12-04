import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('./Picture/Fig0208(a).tif', 0)

# Initialize intensity values with 256 zeros
intensity_count = [0] * 256

height, width = img.shape[:2]
N = height * width

# Array for new_image
high_contrast = np.zeros(img.shape)

for i in range(0, height):
    for j in range(0, width):
        # Find pixels count for each intensity
        intensity_count[img[i][j]] += 1

L = 256

intensity_count, total_values_used = np.histogram(img.flatten(), L, [0, L-1])
# Calculate the PDF
pdf_list = np.ceil(intensity_count * (L - 1) / img.size)
# Calculate the CDF
cdf_list = pdf_list.cumsum()

for y in range(0, height):
    for x in range(0, width):
        # Apply the new intensities in our new image
        high_contrast[y, x] = cdf_list[img[y, x]]

# Plot the Histograms
cv2.imwrite('high_contrast.png', high_contrast)

plt.hist(img.ravel(), 256, [0, 256])
plt.xlabel('Intensity Values')
plt.ylabel('Pixel Count')
plt.show()

plt.hist(high_contrast.ravel(), 256, [0, 256])
plt.xlabel('Intensity Values')
plt.ylabel('Pixel Count')
plt.show()
