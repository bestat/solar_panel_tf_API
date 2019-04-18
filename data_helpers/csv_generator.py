

#!/usr/bin/env python

"""
Copyright 2019 @ Bestat
"""

import argparse
import os
import glob
import numpy as np
import cv2
import pandas as pd
from imutils import paths
import argparse

import pdb
from sklearn.model_selection import train_test_split

#-------------------------------------------------
#Path Configuration
#-------------------------------------------------    
parser = argparse.ArgumentParser(description='CSV File Generator')
parser.add_argument('--api_dir',type=str, help='path to API main directory',default='./')
parser.add_argument('--data_dir',type=str, help='path to data directory',default='./')
parser.add_argument('--rgb_dir',type=str,help='rgb images folder')
parser.add_argument('--mask_dir',type=str,help='mask images folder')
parser.add_argument('--csv_dir',type=str,help='csv files output folder')
parser.add_argument('--train_val_split_rate',type=float,help='Splitting data into train and validation sets',default=0.8)


args = parser.parse_args()

API_DIR = args.api_dir
DATA_DIR = args.data_dir
RGB_DIR = args.rgb_dir
MASK_DIR = args.mask_dir
CSV_DIR = args.csv_dir
train_val_split_rate = args.train_val_split_rate

#-------------------------------------------------
# Get bbox Coordinates
#------------------------------------------------- 
mask_dir_files = sorted(list(paths.list_images(os.path.join(DATA_DIR,MASK_DIR))))
rgb_dir_files = sorted(list(paths.list_images(os.path.join(DATA_DIR,RGB_DIR))))

bboxes = {}
bboxes_rois = {}
for inx,mask_path in enumerate(mask_dir_files): 

    # read img and convert to gray-scale
    img = cv2.imread(mask_path)
    img2gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # derive contour points
    contours, hierarchy = cv2.findContours(img2gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # read rgb image
    rgb_img = cv2.imread(rgb_dir_files[inx])
 
    #filename for saving 
    def filename_splitter(paths):
        return os.path.split(paths)[-1]

    filename = filename_splitter(rgb_dir_files[inx])
    
    print("**********************************")
    print('Prosessed image:',filename)
    print("**********************************")
    
    #mask image shape
    height = np.shape(img)[0]
    width = np.shape(img)[1]

    
    if len(contours) == 0:
        
        print(filename,'>>> no contour found')
        
        bbox_dict = {}
        # parameters to dictionary
        bbox_dict['filename'] = filename
        bbox_dict['width'] = width
        bbox_dict['height'] = height
        bbox_dict['class'] = 'no panel exists'
        bbox_dict['xmin'] = None
        bbox_dict['ymin'] = None
        bbox_dict['xmax'] = None
        bbox_dict['ymax'] = None
        bboxes[inx] = bbox_dict
        
    else:    
        
        rois=[]
        bbox_contours = {}
        for i,contour in enumerate(contours):
            
            bbox_dict = {}
            # evaluate bbox parameters
            x,y,w,h = cv2.boundingRect(contour)
            
            # parameters to dictionary
            bbox_dict['filename'] = filename
            bbox_dict['width'] = width
            bbox_dict['height'] = height
            bbox_dict['class'] = 'solar panel'
            bbox_dict['xmin'] = x 
            bbox_dict['ymin'] = y 
            bbox_dict['xmax'] = x + w 
            bbox_dict['ymax'] = y + h 
            
            bboxes['{} {}'.format(inx,i)] = bbox_dict # adding 5 pixels
            
            print('{} >>> contour {} parameters'.format(filename,i),':',bboxes['{} {}'.format(inx,i)])
            
#-------------------------------------------------
# Create 'train.csv' & 'val.csv' files
#------------------------------------------------- 
def create_dataFrame():
    
    # Get values from bboxes
    values = []
    for key,value in bboxes.items():
        values.append(value)

    # Create Dataframe
    df = pd.DataFrame(values,columns = ['filename','width','height','class','xmin','ymin','xmax','ymax'])
    df_with_mask = df.dropna()
    
    print('Total files with annotations:',len(df_with_mask))
    print('Total files with no annotation:',len(df) - len(df_with_mask))
        
    # Split into Train and Validation sets
    df_train, df_val = train_test_split(df_with_mask, test_size=train_val_split_rate)
    
    if not os.path.exists(os.path.join(DATA_DIR,CSV_DIR)):
        os.mkdir(os.path.join(DATA_DIR,CSV_DIR))
        
    # Save Dataframes as csv
    df_train.to_csv(os.path.join(API_DIR,CSV_DIR,'train.csv'),index=False)
    df_val.to_csv(os.path.join(DATA_DIR,CSV_DIR,'val.csv'),index=False)

    
if __name__ == "__main__":
    
    create_dataFrame()