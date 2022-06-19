import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import sys
import math
from Social_Distance_Tracker import DetectPeople

print("done importing")

class CalcDistance:
    def __init__(self,object,img):
        self.threshold=0.7
        self.height_avg = 165   #in cm
        self.centers = []
        boxes, scores, classes, num = obj1.frames(img)
        self.pick = []
        self.img = img
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                self.pick.append(box)
                center = self.center(box[1], box[0], box[3], box[2])
                self.centers.append(center)
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(0,255,0),2)

    #calculate center of rect
    def center(self,xA, yA, xB, yB):
        Xmid = (xA + xB)/2
        Ymid = (yA + yB)/2
        return [Xmid,Ymid]

    #calculate distance between two rects in pixels
    def dist(self,xA1, xA2, xB1, xB2, i, j):
        inf = sys.maxsize
        a = abs(xA1-xB2)
        b = abs(xA2-xB1)
        c = abs(self.centers[i][0] - self.centers[j][0])

        xDist = min(a if a>0 else inf, b if b>0 else inf, c)
        xDist = xDist**2
        yDist = abs(self.centers[i][1] - self.centers[j][1])**2
        sqDist = xDist + yDist
        return math.sqrt(sqDist)

    def distchecker(self):
        img = self.img
        for i in range(len(self.pick)-1):
            boxI = self.pick[i]
            (xA1, yA1, xB1, yB1) = (boxI[1], boxI[0], boxI[3], boxI[2])
            for j in range(i+1,len(self.pick)):
                boxJ = self.pick[j]
                (xA2, yA2, xB2, yB2) = (boxJ[1], boxJ[0], boxJ[3], boxJ[2])

                #calculate distance in pixels
                dist = self.dist(xA1, xA2, xB1, xB2, i, j)

                #calculate actual distance in cm
                heightI = abs(yA1 - yB1)
                heightJ = abs(yA2 - yB2)

                if heightI==0 or heightJ==0:
                    continue

                ratioI = self.height_avg/heightI     # in cm/pixels
                ratioJ = self.height_avg/heightJ

                meanRatio = (ratioI + ratioJ)/2

                dist = dist * meanRatio       # in cm
                

                if dist<100: # violation ==> red
                    cv2.rectangle(img,(xA1,yA1),(xB1,yB1),(0,0,255),2)
                    cv2.rectangle(img,(xA2,yA2),(xB2,yB2),(0,0,255),2)


if __name__ == "__main__":
    #Edit your training model path
    model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'   
    obj1 = DetectPeople(frozenpath=model_path)
    
    #Edit your input video path
    cap = cv2.VideoCapture('input1.mp4')                                

    while True:
        r, img = cap.read()

        #resize frame
        height, width, layers = img.shape
        new_h=height/2
        new_w=width/2
        img = cv2.resize(img, (int(new_w), int(new_h)))

        # Verify if physical distancing rules are followed or not
        obj2 = CalcDistance(obj1, img)
        obj2.distchecker()

        #Display frame
        cv2.imshow("frame", img)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows() 
