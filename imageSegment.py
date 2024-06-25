# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_dir = 'dataset/test'
output_dir = 'dataset/output'
gt_dir = 'dataset/groundtruth'

# you are allowed to import other Python packages above
##########################
def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    # convert image to HSV and RGB
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # denoise colored image and its HSV version
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img_hsv_denoised = cv2.fastNlMeansDenoisingColored(img_hsv, None, 10, 10, 7, 21)

    # thresholding to extract fruit mask
    _, fruit_mask = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)

    # create a mask of zeros with the same height and width as the input image
    mask = np.zeros(img_hsv.shape[:2], np.uint8)

    # define the bounding box around the object (fruit)
    rect = (25, 25, img_hsv.shape[1] - 65, img_hsv.shape[0] - 65)
    
    # allocate memory for two arrays used by the algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # apply the GrabCut algorithm
    cv2.grabCut(img_hsv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # modify the mask such that sure and probable backgrounds set to 0, and sure and probable foreground set to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # multiply the mask with the input image to extract the segmented object
    background_mask = img_hsv * mask2[:, :, np.newaxis]
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_RGB2GRAY)

    # define lower and upper bounds for rotten brown parts
    rotten_brown_lower = np.array([5, 10, 10])
    rotten_brown_upper = np.array([150, 255, 180])

    # define lower and upper bounds for rotten green parts
    rotten_green_lower = np.array([26, 79, 79])
    rotten_green_upper = np.array([150, 199, 192])

    # create mask for the rotten part
    rotten_brown_mask = cv2.inRange(img_hsv_denoised, rotten_brown_lower, rotten_brown_upper)
    rotten_green_mask = cv2.inRange(img_hsv_denoised, rotten_green_lower, rotten_green_upper)

    # morphological operations to refine the mask
    rotten_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rotten_brown_mask = cv2.erode(rotten_brown_mask, rotten_kernel, iterations=1)
    rotten_brown_mask = cv2.dilate(rotten_brown_mask, rotten_kernel, iterations=5)
    
    rotten_green_mask = cv2.erode(rotten_green_mask, rotten_kernel, iterations=1)
    rotten_green_mask = cv2.dilate(rotten_green_mask, rotten_kernel, iterations=5)

    # create a copy of the hsv image
    outImg = img_hsv.copy()

    # combine the masks for all rotten regions using a bitwise OR operation
    mask_combined = cv2.bitwise_or(rotten_brown_mask,rotten_green_mask)
    # apply the combined mask to the output image using a bitwise AND operation
    outImg = cv2.bitwise_and(outImg, outImg, mask=mask_combined)
    # convert the resulting image to grayscale
    outImg = cv2.cvtColor(outImg, cv2.COLOR_RGB2GRAY)

    # intensity mapping
    outImg[np.where(background_mask)] = [2]
    outImg[np.where(mask_combined)] = [1]
    outImg[np.where(fruit_mask)] = [0]
    
    # END OF YOUR CODE
    #########################################################################
    return outImg
