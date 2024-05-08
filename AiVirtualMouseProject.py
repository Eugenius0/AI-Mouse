# Installed packages before starting: opencv-python, mediapipe and autopy


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

##### AutoPy #####

"""
AutoPy is a simple, cross-platform GUI automation library for Python. 
It includes functions for controlling the keyboard and mouse, 
finding colors and bitmaps on-screen, and displaying alerts.

Source: https://pypi.org/project/autopy/
"""

##### NumPy #####

"""
NumPy is an open source project aiming to enable numerical computing with Python. 
It was created in 2005, building on the early work of the Numeric and Numarray libraries. 
NumPy will always be 100% open source software, free for all to use and released 
under the liberal terms of the modified BSD license.

Source: https://numpy.org/about/
"""



import cv2
import numpy as np  # Program library for Python, which allows easy handling of vectors,
# matrices or large multidimensional arrays in general.
import HandTrackingModule as htm
import time  # to check the framerate
import autopy  # allow you to move around with your mouse


##########################
wCam, hCam = 640, 480  # width and height of the cam
frameR = 100  # Frame Reduction
smoothening = 5  # find the right value to get smooth motion of the mouse, when higher then slower
#########################

pTime = 0
plocX, plocY = 0, 0  # previous location
clocX, clocY = 0, 0  # current location


cap = cv2.VideoCapture(0)  # (1) if you have multiple cams
# fix width and height of the cam
cap.set(3, wCam)  # prop id for width 3
cap.set(4, hCam)  # prop id for height 4

detector = htm.handDetector(maxHands=1)  # create object, one hand to control the mouse
wScr, hScr = autopy.screen.size()  # give the size of the screen
# print(wScr, hScr) returns 1280.0 720.0

while True:
    # 1. Find hand landmarks, result of this -> picture 4
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)  # detect hand and draw

    """
    idea: if we have just the index finger then the mouse will move
    if we have the middle finger up as well then clicking mode
    if clicking mode and distance between the two fingers is less than a certain value
    then we will detect it as a click
    so bring fingers together to click, move the mouse in index mode
    """
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # x1 and y1 are points of index finger, gives element number 1 and 2
        x2, y2 = lmList[12][1:]  # give us the coordinates of tip of index finger (8) and middle finger(12)
        # print(x1, y1, x2, y2) returns points, for example: 348 149 400 124

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers) shows which fingers are up and down, for example peace sign: [0, 1, 1, 0, 0]
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)  # draws a purple rectangle, when finger at top of it then mouse also at top
                                         # picture 6
        # 4. Only Index Finger : Moving Mode -> check if index finger is up, so if it's 1
        if fingers[1] == 1 and fingers[2] == 0:  # index finger up, middle finger down

            # 5. Convert Coordinates
            # (cam gives values of 640x480, but normal screen has 920x1080, so need that for correct positioning)
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # 6. Smoothen values  (prevent flickering)
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening


            # 7. Move Mouse
            autopy.mouse.move(wScr-clocX, clocY)  # when you move to the left the mouse also moves to the left
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # picture 5
            plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:  # if both fingers are up

            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)  # get distance between index and middle finger
            print(length)  # returns distance for example: 68.24954212300622, picture 7

            # 10. Click mouse if distance short
            if length < 20:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)  # green circle when click detected -> picture 8
                autopy.mouse.click()  # click when index and middle finger are up and distance lower 20



    # 11. Frame Rate
    cTime = time.time()  # give the current time
    fps = 1 / (cTime - pTime)  # frames per second
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)  # return fps framerate (number on top left corner)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
