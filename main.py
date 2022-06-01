import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray
import cv2


def create_Gaussian_Filter(cutoff_frequency):
    cutoff_frequency = int(cutoff_frequency)

    # k=l =Filter Shape(Kernel Size)
    #To make fiter size odd.
    k = cutoff_frequency * 4 + 1
    std_deviation = cutoff_frequency
    # This create array of 1
    x1 = np.ones(k)
    #print(x1)
    # used Gauesssien Equation to make Gaussien Filter
    x1 = (1 / np.sqrt(2 * math.pi)) * (1 / std_deviation) * np.exp((-1 / (2 * std_deviation * std_deviation)) * (x1) * (x1))
    # return outer product of x1,x1
    kernel = np.outer(x1, x1)
    # sum of array elements will be divide
    alpha = np.sum(kernel)
   # print(kernel)
   # print(alpha)
    filter = kernel / alpha
    #print(filter.shape)
    return filter


def my_imfilter(image, filter):
    # to check filter array shape should be odd
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    filter_size_w = filter.shape[0]
    filter_size_h = filter.shape[1]

    # padding Image so that filtered image have same size
    pad_w = int((filter.shape[0] - 1) / 2)
    pad_h = int((filter.shape[1] - 1) / 2)
    pad_width = ((pad_w, pad_w), (pad_h, pad_h), (0, 0))
    # print(pad_width)
    padded_image = np.pad(image, pad_width=pad_width, mode='reflect')
    #Create empty array to store the filtered metrix
    filtered_image = np.empty(image.shape)
    # applying filter 3 times because of RGB channel
    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i, j, c] = np.sum(np.multiply(padded_image[i:i + filter_size_w, j:j + filter_size_h, c], filter))
    return filtered_image


def create_hybrid_image(image1, image2, filter):
    assert image1.shape[0] == image2.shape[0], "Both Image Shape should be same"
    assert image1.shape[1] == image2.shape[1], "Both Image Shape should be same"
    assert image1.shape[2] == image2.shape[2], "Both Image Shape should be same"
    assert filter.shape[0] <= image1.shape[0], "Filter shape should be less or equal to image"
    assert filter.shape[1] <= image1.shape[1], "Filter shape should be less or equal to image"
    assert filter.shape[0] % 2 == 1, "Filter shape should be Odd number"
    assert filter.shape[1] % 2 == 1, "Filter shape should be Odd number"

    low_frequencies = my_imfilter(image1, filter)
    high_frequencies = image2 - my_imfilter(image2, filter)
    hybrid_image = low_frequencies + high_frequencies

    return low_frequencies, high_frequencies, hybrid_image


# Main Function
if __name__ == "__main__":
    image1 = Image.open("dog.bmp")
    # print(image1.size)
    image1Array = asarray(image1)
    #print(image1Array.shape)
    #print(image1Array.shape[2])
    plt.figure(figsize=(3, 3))
    plt.title('Original Image1')
    plt.imshow((image1Array).astype(np.uint8));
    plt.show()

    image2 = Image.open('cat.bmp')
    image2Array = asarray(image2)
    #print(image2Array.shape)
    plt.figure(figsize=(3, 3))
    plt.title('Original Image2')
    plt.imshow((image2Array).astype(np.uint8))
    plt.show()

    cut_off_frequency = int(input("Enter Cut-off Frequency:"))
    gaussian_filter = create_Gaussian_Filter(cut_off_frequency)
    low_frequencies, high_frequencies, hybrid_image = create_hybrid_image(image1Array, image2Array, gaussian_filter)

    plt.figure(figsize=(3, 3))
    plt.title("LowPass Image1")
    plt.imshow((low_frequencies).astype(np.uint8))
    plt.show()
    plt.figure(figsize=(3, 3))
    plt.title("HighPass Image2")
    plt.imshow((high_frequencies).astype(np.uint8))
    plt.show()
    plt.figure(figsize=(3, 3))
    plt.title("Hybrid Image")
    plt.imshow((hybrid_image).astype(np.uint8))
    plt.show()


    ##  Used cv2 for creating Gaussian Filter just for comperision
    cutoff_frequency = int(input("Enter Cut-off Frequency for CV2 Gaussian Filter:"))
    filter1 = cv2.getGaussianKernel(ksize=cutoff_frequency * 4 + 1, sigma=cutoff_frequency)
    filter1 = np.dot(filter1, filter1.T)
    low_frequencies, high_frequencies, hybrid_image = create_hybrid_image(image1Array, image2Array, filter1)
    plt.figure(figsize=(3, 3))
    plt.title("LowPass Image1 using CV2")
    plt.imshow((low_frequencies).astype(np.uint8))
    plt.show()
    plt.figure(figsize=(3, 3))
    plt.title("HighPass Image2 using cv2")
    plt.imshow((high_frequencies).astype(np.uint8))
    plt.show()
    plt.figure(figsize=(3, 3))
    plt.title("Hybrid Image using cv2")
    plt.imshow((hybrid_image).astype(np.uint8))
    plt.show()
