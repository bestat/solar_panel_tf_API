
# coding: utf-8
# @Bestat 2019

#-------------------------------------------
# Use Frozen Model and Calculate Predictions
#-------------------------------------------

# Import packages
import sys
sys.path.append("./")

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import tensorflow as tf

import label_map_util
import visualization_utils as vis_util 


class TF_Detector():
    
    def __init__(self):
        
        self.BASE_PATH = '/opt/solar_panel_detection'
        self.PATH_TO_MODEL_FOLDER = 'frozen_model/faster_rcnn_experiment_1'
        self.PATH_TO_FROZEN_GRAPH = os.path.join(self.BASE_PATH , self.PATH_TO_MODEL_FOLDER,'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(self.BASE_PATH,'data/labelmap.pbtxt')
        self.TEST_IMAGE_PATH = '/share/data_solar_panel/RGB_patches'
        
        self.NUM_CLASSES = 1
        self.visualize = False

    def get_test_images(self,test_dir):

        # Get test image paths
        TEST_IMAGES = [os.path.split(i)[-1] for i in glob.glob(os.path.join(test_dir,'*.png'))]

        return sorted(TEST_IMAGES)


    def main(self,image_filepath):

        TEST_IMAGES = self.get_test_images(image_filepath)

        box_coord = []
        for IMAGE_NAME in tqdm(TEST_IMAGES):

            print('********************************')
            print('Processing image:',IMAGE_NAME)
            print('********************************')

            # Path to image
            PATH_TO_IMAGE = os.path.join(self.TEST_IMAGE_PATH,IMAGE_NAME)

            img = cv2.imread(PATH_TO_IMAGE)
            width = img.shape[1]
            height = img.shape[0]

            # Load the label map.
            label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            # Load Tensorflow model into memory.
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as f:
                    serialized_graph = f.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                sess = tf.Session(graph=detection_graph)

            # Input tensor is the image
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Output tensors are the detection boxes, scores, and classes
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represents level of confidence for each of the objects.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

            # Number of objects detected
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Load image using OpenCV and
            # expand image dimensions to have shape: [1, None, None, 3]
            image = cv2.imread(PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run([detection_boxes, 
                                                      detection_scores, 
                                                      detection_classes, 
                                                      num_detections],
                                                     feed_dict={image_tensor: image_expanded})

            if self.visualize:
                # Draw the results of the detection
                vis_util.visualize_boxes_and_labels_on_image_array(image,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=6,
                                                                   min_score_thresh=0.2)
                
                
                # All the results have been drawn on image
                cv2.imwrite(os.path.join(self.BASE_PATH,'results/{}'.format(IMAGE_NAME)),image)
                #plt.imshow(image,cmap='gnuplot')
                #plt.show()
                

            # Get scores and bboxes
            for i,score in enumerate(np.squeeze(scores)):

                # inx of bbox
                bbox = 'bbox_{}'.format(i)

                if score >= 0.2:

                    #get bbox coordinates
                    ymin,xmin,ymax,xmax = np.squeeze(boxes)[i]

                    ymin = int(ymin * height)
                    xmin = int(xmin * width)
                    ymax = int(ymax * height)
                    xmax = int(xmax * width)

                    coordinates = (ymin,xmin,ymax,xmax)
                    box_coord.append(coordinates)

                else:
                    continue

            print('*****************************************************')
            print('Image Name:',IMAGE_NAME)
            print('Box Coordinates:',box_coord)
            print('Number of Detected Boxes',np.array(box_coord).shape[0])
            print('*****************************************************')

        print('Total Detected Objects:',np.array(box_coord).shape[0])
        print('Shape of Box Array:',np.array(box_coord).shape)

        return np.array(box_coord)

if __name__ == '__main__':

    inference = TF_Detector()
    inference.main()