
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, n_channels).
    """
    out = None
    # YOUR CODE HERE
    
    #Read in image
    out = io.imread(img_path)
    
    #Convert image values to between 0 and 1
    out = out / 255
    
    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    
    # YOUR CODE HERE
    
    #Get first height and width dimension
    dimensions = np.shape(image)
    print("Height: ", dimensions[0])
    print("Width: ", dimensions[1])
    
    #Check if image is 2D or 3D
    if len(np.shape(image)) != 3:
        print("Channel: 1")
    else:
        print("Channel: ", dimensions[2])

    return None

def crop(image, x1, y1, x2, y2):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        (x1, y1): the coordinator for the top-left point
        (x2, y2): the coordinator for the bottom-right point
        

    Returns:
        out: numpy array of shape(x2 - x1, y2 - y1, 3).
    """

    out = None
    ### YOUR CODE HERE
    
    out = image[y1:y2, x1:x2]
    
    return out
    
def resize(input_image, fx, fy):
    """Resize an image using the nearest neighbor method.
    Not allowed to call the matural function.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.
        fx (float): the resize scale on the original width.
        fy (float): the resize scale on the original height.

    Returns:
        np.ndarray: Resized image, with shape `(image_height * fy, image_width * fx, 3)`.
    """
    
    #Get input image shape
    R = int(np.shape(input_image)[0])
    C = int(np.shape(input_image)[1])
    
    #Calculate amount of rows and cols for the scaled image
    newR = int(R*fy)
    newC = int(C*fx)
    out = np.zeros((newR, newC, 3))
    
    # Fill in every pixel in the scaled image
    for row in range(newR):
        for col in range(newC):
            
            #Get nearest input pixel after scaling
            row_n = int(row/fy)
            col_n = int(col/fx)
            
            #Set scaled image pixel to nearest scaled pixel
            out[row][col] = input_image[row_n][col_n]

    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, divided by 255.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    ### YOUR CODE HERE

    #Get shape of the image
    R = np.shape(image)[0]
    C = np.shape(image)[1]
    out = np.zeros((R, C, 3))
    
    #Loop over each pixel and edit the contrast
    for row in range(R):
        for col in range(C):
            x_p = image[row][col]
            
            for i in range(len(x_p)):
                out[row][col][i] = (factor * (x_p[i] - 0.5)) + 0.5
                
    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(image_height, image_width)`.
    """
    out = None
    ### YOUR CODE HERE
    
    #Get shape of input image
    R = np.shape(input_image)[0]
    C = np.shape(input_image)[1]
    out = np.zeros((R, C))
    
    #Loop over image and sum the R, G, B values.
    for row in range(R):
        for col in range(C):
            out[row][col] = sum(input_image[row][col])/3
                 
    return out
    
def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.
  
                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th
    
    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None
    ### YOUR CODE HERE
    
    #Get shape of input image
    R = np.shape(grey_img)[0]
    C = np.shape(grey_img)[1]
    out = np.zeros((R, C))
    
    #Loop over image and set pixel value to 1 if over threshold
    for row in range(R):
        for col in range(C):
            if grey_img[row][col] > th:
                out[row][col] = 1

    return out

def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    
    out = None
    
    ### YOUR CODE HERE
    
    #Get size of original image
    R = np.shape(image)[0]
    C = np.shape(image)[1]
    
    #kernel = kernel / np.sum(kernel)
    
    #Pad out image
    padding = int(np.floor(np.shape(kernel)[0]/2))
    padded_image = np.zeros((R + (2*padding), C + (2*padding)))
    padded_image[padding:R+padding, padding:C+padding] = image
    
    #Create output image
    out = np.zeros((R, C))
    
    #Flip the kernel
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)

    #Perform convolution
    for row in range(padding, R+padding):
        for col in range(padding, C+padding):
            out[row-padding][col-padding] = np.sum(kernel * padded_image[row-padding:row+padding+1, col-padding:col+padding+1])
    
    return out

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])
    
    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3
    
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."
    
    print("Your solution is correct")


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE
    
    #Get size of original image
    R = np.shape(image)[0]
    C = np.shape(image)[1]
    
    if len(np.shape(image)) == 2:
        return conv2D(image, kernel)
    
    out = np.zeros((R, C, 3))
    
    for i in range(3):
        cur = conv2D(image[:,:,i], kernel)
        
        for row in range(R):
            for col in range(C):
                out[row][col][i] = cur[row][col]
        
    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE
    
    #Get size of original image
    R = np.shape(image)[0]
    C = np.shape(image)[1]
    
    kernel = np.flip(kernel, axis = 0)
    kernel = np.flip(kernel, axis = 1)


    if len(np.shape(image)) == 2:
        return conv2D(image, kernel)
    
    out = np.zeros((R, C, 3))
    
    for i in range(3):
        cur = conv2D(image[:,:,i], kernel)
        
        for row in range(R):
            for col in range(C):
                out[row][col][i] = cur[row][col]
        
    return out

