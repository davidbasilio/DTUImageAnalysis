import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import cv2
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

def show_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 6))
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.grid()
    plt.plot(histogram)
    plt.show()
    sleep(2)
    plt.close('all')

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)

def calculate_ranges(image):
    return np.min(image), np.max(image)

def calculate_ranges_as_float(image):
    return np.min(img_as_float(image)), np.max(img_as_float(image))

def calculate_ranges_as_ubyte(image):
    return np.min(img_as_ubyte(img_as_float(image))), np.max(img_as_ubyte(img_as_float(image)))

def histogram_stretch(img_in):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0
	
    # compute the output image applying the formula 
    img_out = (max_desired - min_desired) / (max_val - min_val) * (img_float - min_val) + min_desired
 
    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)

def gamma_map(img, gamma):
    return img_as_ubyte(np.power(img_as_float(img), gamma))

if __name__ == '__main__':
    vertebra = cv2.imread("exercises/ex3-PixelwiseOperations/data/vertebra.png", cv2.IMREAD_GRAYSCALE)
    show_histogram(vertebra)
    stop = False
    while not stop:
        show_in_moved_window("Vertebra", vertebra, 0, 10)
        vertebra_strech = histogram_stretch(vertebra)
        show_in_moved_window("Vertebra Strech", vertebra_strech, 600, 10)
        vertebra_gamma = gamma_map(vertebra, .2)
        show_in_moved_window("Vertebra Gamma", vertebra_gamma, 1200, 10)
        if cv2.waitKey(1) == ord('q'):
            stop = True


