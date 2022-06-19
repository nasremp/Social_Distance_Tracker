import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import sys
import math

class DetectPeople:
    
    def __init__(self, frozenpath):
        self.frozenpath = frozenpath
        # Path to frozen detection graph of the actual model that is used for the object detection.

        self.detect = tf.Graph() #A TensorFlow computation, represented as a dataflow graph.
        with self.detect.as_default(): #to use the graph directly without tf function
            df = tf.GraphDef()
            with tf.gfile.GFile(self.frozenpath, 'rb') as file: #convert to an API that is close to Python's file I/O objects
                get_graph = file.read()
                df.ParseFromString(get_graph) #extract the graphh from string
                tf.import_graph_def(df, name='')

        self.default_graph = self.detect.as_default()
        self.operate = tf.Session(graph=self.detect) #to be able to run

        #Definite input and output Tensors for detection_graph
        self.image_tensor = self.detect.get_tensor_by_name('image_tensor:0')
        #Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detect.get_tensor_by_name('detection_boxes:0')
        #Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detect.get_tensor_by_name('detection_scores:0')
        self.classes = self.detect.get_tensor_by_name('detection_classes:0')
        self.numbers = self.detect.get_tensor_by_name('num_detections:0')

    def frames(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.     
        (new_boxes, new_scores, new_classes, new_numbers) = self.operate.run(
            [self.boxes, self.scores, self.classes, self.numbers],
            feed_dict={self.image_tensor: image_np_expanded})
        height, width,_ = image.shape
        boxes_list = [None for i in range(new_boxes.shape[1])]
        for i in range(new_boxes.shape[1]):
            boxes_list[i] = (int(new_boxes[0,i,0] * height),
                        int(new_boxes[0,i,1]*width),
                        int(new_boxes[0,i,2] * height),
                        int(new_boxes[0,i,3]*width))
        return boxes_list, new_scores[0].tolist(), [int(x) for x in new_classes[0].tolist()], int(new_numbers[0])
