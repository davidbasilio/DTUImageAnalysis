import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import cv2
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu

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

def threshold_image(img_in, thres):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    thres = thres / 255
    img_in = img_as_float(img_in)
    img_thres = (np.where(img_in < thres, 0, 255))
    return img_as_ubyte(img_thres)

def segment_rgb(im_org, color = 'blue'):
    r_comp = im_org[:, :, 0]
    g_comp = im_org[:, :, 1]
    b_comp = im_org[:, :, 2]
    if color == 'blue':
        segm = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & (b_comp > 180) & (b_comp < 200)
    elif color == 'red':
        segm = (r_comp < 180) & (r_comp > 160) & (g_comp > 50) & (g_comp < 60) & (b_comp > 50) & (b_comp < 60)
    img_out = (np.where(segm == True, 255, 0))
    print((segm == True).sum())
    return img_as_ubyte(img_out)

def segment_hsv(im_org, color = 'blue'):
    hsv_img = cv2.cvtColor(im_org, cv2.COLOR_BGR2HSV)
    hue_comp = hsv_img[:, :, 0]
    sat_comp = hsv_img[:, :, 1]
    val_comp = hsv_img[:, :, 2]
    if color == 'blue':
        segm = (hue_comp < 130) & (hue_comp > 90) & (sat_comp < 255) & (sat_comp > 50) & (val_comp < 255) & (val_comp > 50)
    elif color == 'red':
        segm = ((hue_comp < 10) or (hue_comp > 160) & (hue_comp < 180)) & (sat_comp < 255) & (sat_comp > 50) & (val_comp < 255) & (val_comp > 50)
    img_out = (np.where(segm == True, 255, 0))
    print((segm == True).sum())
    return img_as_ubyte(img_out)

if __name__ == '__main__':
    vertebra = cv2.imread("exercises/ex3-PixelwiseOperations/data/vertebra.png", cv2.IMREAD_GRAYSCALE)
    # show_histogram(vertebra)
    stop = True
    while not stop:
        show_in_moved_window("Vertebra", vertebra, 0, 10)
        vertebra_strech = histogram_stretch(vertebra)
        # show_in_moved_window("Vertebra Strech", vertebra_strech, 600, 10)
        vertebra_gamma = gamma_map(vertebra, .2)
        # show_in_moved_window("Vertebra Gamma", vertebra_gamma, 1200, 10)
        vertebra_thres = threshold_image(vertebra, 190)
        show_in_moved_window("Vertebra Threshold", vertebra_thres, 800, 10)
        if cv2.waitKey(1) == ord('q'):
            stop = True    
            
    dark_bg = cv2.imread("exercises/ex3-PixelwiseOperations/data/dark_background.png")
    dark_bg_gray = cv2.cvtColor(dark_bg, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(dark_bg_gray)
    dark_bg_thres = threshold_image(dark_bg_gray, threshold)
    stop = True
    while not stop:    
        show_in_moved_window("Original", dark_bg, 0, 10)
        show_in_moved_window("Dark Background", dark_bg_thres, 600, 10)
        if cv2.waitKey(1) == ord('q'):
            stop = True 
    
    dtu_sign = cv2.imread("exercises/ex3-PixelwiseOperations/data/DTUSigns2.jpg", cv2.IMREAD_COLOR)    
    dtu_sign = cv2.cvtColor(dtu_sign, cv2.COLOR_BGR2RGB)
    dtu_sign = cv2.resize(dtu_sign, (int(dtu_sign.shape[1]/8), int(dtu_sign.shape[0]/8)))
    dtu_sign_rgb = segment_rgb(dtu_sign, color='blue')
    dtu_sign_hsv = segment_hsv(dtu_sign, color='blue')
    stop = False
    while not stop:    
        show_in_moved_window("Original", cv2.cvtColor(dtu_sign, cv2.COLOR_RGB2BGR), 0, 10)
        show_in_moved_window("RGB segmentation", dtu_sign_rgb, 350, 10)
        show_in_moved_window("HSV segmentation", dtu_sign_rgb, 800, 10)
        if cv2.waitKey(1) == ord('q'):
            stop = True 
