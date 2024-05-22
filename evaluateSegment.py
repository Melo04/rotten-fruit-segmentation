# -*- coding: utf-8 -*-
"""
DO NOT MODIFY ANY CODES IN THIS FILE EXCEPT THE DEFAULT PARAMETERS
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: LOH

For questions or bug reporting, please send an email to yploh@mmu.edu.my
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import importlib
import sys, getopt
from prettytable import PrettyTable
import imageSegment as seg
from os import listdir
from os.path import isfile, join, splitext

# Default parameters (the only code you can change)
verbose = False#, 1, or 2
setting = 'data' #'add' #
if setting == 'data':
    input_dir = 'dataset/test'
    output_dir = 'dataset/output'
    groundtruth_dir = 'dataset/groundtruth'
    numImages = 80
else:
    input_dir = 'add_dataset/test' 
    output_dir = 'add_dataset/output'
    groundtruth_dir = 'add_dataset/groundtruth'
    numImages = 25
eps = 0.00000001

onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
files = onlyfiles[0:numImages]

## Read command linehargs
myopts, args = getopt.getopt(sys.argv[1:],"i:vph")

# Reload module
importlib.reload(seg)

################################################
# o == option    a == argument passed to the o #
################################################

# parsing command line args
for o, a in myopts:
    #print(o)
    #print(a)
    if o == '-v':
        verbose = 1
    elif o == '-p':
        verbose = 2
    elif o == '-h':
        print("\nUsage: %s -v to show evaluation for every image" % sys.argv[0])
        print("                           -p to show average evaluation of every region")
        sys.exit()
    else:
        print(' ')

error = np.zeros((numImages,3),)
precision = np.zeros((numImages,3),)
recall = np.zeros((numImages,3),)
iou = np.zeros((numImages,3),)

# Evaluate each image and compare with ground-truth
for i,name in enumerate(files):
    inputImg = cv2.imread(input_dir + '/' + name)
    outputImg = seg.segmentImage(inputImg).astype('float32')
    imgName = splitext(name)
    plt.imsave(output_dir + '/' + imgName[0] + '.png',255*outputImg/3,cmap=cm.gray)
    gt = cv2.imread(groundtruth_dir + '/' + imgName[0] + '.png',0)
    gt = np.round(2*gt.astype('float32')/gt.max())
    
    for p in range(3):
        outputPart = (outputImg == p)*1
        gtPart = (gt == p)*1       
        precision[i,p] = np.round(sum(sum(gtPart*outputPart))/(sum(sum(outputPart))+eps),4)
        recall[i,p] = sum(sum(gtPart*outputPart))/sum(sum(gtPart))
        error[i,p] = 1 - ((2*precision[i,p]*recall[i,p])/(precision[i,p]+recall[i,p]+eps))
        iou[i,p] = sum(sum(gtPart*outputPart))/sum(sum(np.clip(gtPart+outputPart,0,1)))

# Print performance scores        
if verbose==1:
    print('####  IMAGE RESULTS  ####')
    t = PrettyTable(['Image', 'Error','Precision','Recall','IoU'])
    avg_error = np.mean(error,axis=1)
    avg_precision = np.mean(precision,axis=1)
    avg_recall = np.mean(recall,axis=1)
    avg_iou = np.mean(iou,axis=1)
    
    for i in range(numImages):
        t.add_row([i+1, str(round(avg_error[i],4)),str(round(avg_precision[i],4)),\
                   str(round(avg_recall[i],4)),str(round(avg_iou[i],4))]) 
                   
    t.add_row([' ',' ',' ',' ',' '])
    t.add_row(['All', str(round(np.sum(avg_error)/numImages,4)),str(round(np.sum(avg_precision)/numImages,4)),\
               str(round(np.sum(avg_recall)/numImages,4)),str(round(np.sum(avg_iou)/numImages,4))])
    print(t)
    
elif verbose==2:
    print('####  PARTS RESULTS  ####')
    t = PrettyTable(['Segment', 'Error','Precision','Recall','IoU'])
    pt = ['Background','Rot','Fruit']
    avg_error = np.mean(error,axis=0)
    avg_precision = np.mean(precision,axis=0)
    avg_recall = np.mean(recall,axis=0)
    avg_iou = np.mean(iou,axis=0)
    
    for i in range(3):
        t.add_row([pt[i], str(round(avg_error[i],4)),str(round(avg_precision[i],4)),\
                   str(round(avg_recall[i],4)),str(round(avg_iou[i],4))]) 
                   
    t.add_row([' ',' ',' ',' ',' '])
    t.add_row(['All', str(round(np.sum(avg_error)/3,4)),str(round(np.sum(avg_precision)/3,4)),\
               str(round(np.sum(avg_recall)/3,4)),str(round(np.sum(avg_iou)/3,4))])
    print(t)
else:
    
    avg_error = np.mean(error,axis=1)
    avg_precision = np.mean(precision,axis=1)
    avg_recall = np.mean(recall,axis=1)
    avg_iou = np.mean(iou,axis=1)
    
    print('Adapted Rand Error: %d%%' % (np.sum(avg_error)/numImages*100))
    print('Precision: %d%%' % (np.sum(avg_precision)/numImages*100))
    print('Recall: %d%%' % (np.sum(avg_recall)/numImages*100))
    print('IoU: %d%%' % (np.sum(avg_iou)/numImages*100))
        
        
# END OF EVALUATION CODE####################################################
