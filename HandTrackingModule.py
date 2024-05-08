"""
Own Hand Detection Module to use it easily in different projects
Module is created out of HandTrackingMin.py
"""

# Installed packagaes before starting: opencv-python and mediapipe


##### OpenVC #####

"""
OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library.
OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of
machine perception in the commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses
to utilize and modify the code.

The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic
and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to detect and recognize faces,
identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects,
roduce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene,
find similar images from an image database, remove red eyes from images taken using flash, follow eye movements,
recognize scenery and establish markers to overlay it with augmented reality, etc.

Source: https://opencv.org/about/
"""

##### MediaPipe #####

"""
MediaPipe is a cross-platform framework for building multimodal applied machine learning pipelines

MediaPipe is a framework for building multimodal (eg. video, audio, any time series data),
cross platform (i.e Android, iOS, web, edge devices) applied ML pipelines.
With MediaPipe, a perception pipeline can be built as a graph of modular components, including, for instance, inference models
(e.g., TensorFlow, TFLite) and media processing functions.

Cutting edge ML models

    Face Detection
    Multi-hand Tracking
    Hair Segmentation
    Object Detection and Tracking
    Objectron: 3D Object Detection and Tracking
    AutoFlip: Automatic video cropping pipeline

Source: https://opensource.google/projects/mediapipe    
"""


import cv2
import mediapipe as mp
import time  # to check the framerate
import math


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):  # Konstruktor
        # basic required parameters for the hands (look at hands.py from mediapipe)
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands  # this is a formality you have to do before using this module
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)  # create object hands
        self.mpDraw = mp.solutions.drawing_utils  # method provided by mediapipe, helps to draw the 21 points of the hand
        self.tipIds = [4, 8, 12, 16, 20]  # ids of the tips of the five fingers

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB, send RGB Image to object hands
        self.results = self.hands.process(imgRGB)  # process the frame and give the results
        # results self because if not you can not use it for other methods

        """
        print(results.multi_hand_landmarks)  check if something is detected or not, 
        returns none if no hand in front of camera, if a hand is in front of the camera it returns 
        the landmarks, for example: 
        landmark {
          x: 0.4831007421016693
          y: 0.2747529149055481
          z: -0.6213211417198181
        }
        """

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:   # it will only draw if we ask it to draw
                    # draws (the points and the connections of) a hand (handLms is a single hand)
                    # handLms for the red points (Picture 1), mpHands.HAND_CONNECTIONS for the green connections (Picture 2)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):  # need the img for width and height, handNumber
        xList = []  # contains all the values of x
        yList = []  # contains all the values of y
        bbox = []
        self.lmList = []  # list for all the landmark positions, self to use it in other methods and define it as an instance
        if self.results.multi_hand_landmarks:  # check if any hands were detected or not
            myHand = self.results.multi_hand_landmarks[handNo]  # it will get the first hand
            # and get all the landmarks of it and put them in a list
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape  # give weight and height
                cx, cy = int(lm.x * w), int(lm.y * h)  # give cx and cy position in pixels
                # print(id, cx, cy) # have to return id so you know which position is for which landmark
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    # draws white circles at all points
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    # we can put the cx and cy information in a list to use it in other projects
            # find min and max of x/y of any of the points and save it in bbox
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                # draws a rectangle around hand with 20 pixels extra space (picture 4)
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
                              (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):  # tells which of the fingers are up
        fingers = []  # create a list called fingers

        # Thumb
        # check if the tip of the thumb is on the right or on the left
        # tells us if its open or closed
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        # check if the tip of the finger is above the other landmark which is two steps below it or not
        # if below then it's closed, if above then it's open
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # 1 for open/up
            else:
                fingers.append(0)  # 0 for closed/down
        return fingers  # returns a list with five values, tells if the finger is up or not, 0 -> down, 1 -> up
                        # for example when you do the peace sign: [0, 1, 1, 0, 0]

    # p1 and p2 are ids, find length of the distance between two fingers
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:  # if we want to draw we do all of this
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)  # distance between Thumb and indexfinger
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0  # previous Time
    cTime = 0  # current Time
    cap = cv2.VideoCapture(0)  # create video object, (1) if you have multiple cams
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:  # check that list is not empty
            print(lmList[4])  # return list of position where index is 4 in this format: [id, cx, cy]
            print(bbox)  # return max and mins with extra space in format: (xmin, ymin, xmax, ymax)
        cTime = time.time()  # give the current time
        fps = 1 / (cTime - pTime)  # frames per second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)  # return fps framerate (number on top left corner)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":  # if we run this script then do main()
    main()  # dummy code that will be used to showcase what can this module do
