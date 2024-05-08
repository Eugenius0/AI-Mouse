"""
Basic File for HandTracking (Pictures 1 - 3), HandTrackingModule.py is created out of this file
"""

# Installed packages before starting: opencv-python and mediapipe


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


import cv2  # cv2 to detect something
import mediapipe as mp
import time  # to check the framerate

cap = cv2.VideoCapture(0)  # create video object, (1) if you have multiple cams

mpHands = mp.solutions.hands  # this is a formality you have to do before using this module
hands = mpHands.Hands()  # create object hands
mpDraw = mp.solutions.drawing_utils  # method provided by mediapipe, helps to draw the 21 points of the hand

pTime = 0  # previous Time
cTime = 0  # current Time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB, send RGB Image to object hands
    results = hands.process(imgRGB)  # process the frame and give the results

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

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                """
                print(id, lm) returns the postion of the landmarks, for example:
                11 x: 0.6253311634063721
                y: 0.30744481086730957
                z: -0.27804046869277954

                12 x: 0.6112002730369568
                y: 0.3365429639816284
                z: -0.2670424282550812
                ...
                """
                h, w, c = img.shape  # give weight and height
                cx, cy = int(lm.x * w), int(lm.y * h)  # give cx and cy position in pixels
                # print(id, cx, cy) # have to return id so you know which position is for which landmark
                """
                print(id, cx, cy) returns the position of the 21 points of the hand in pixels, for example:
                0 446 251
                1 397 238
                2 356 239
                3 327 250
                4 305 264
                5 345 214
                6 298 247
                7 290 269
                8 293 280
                9 363 230
                10 316 271
                11 331 280
                12 344 273
                13 380 250
                14 336 285
                15 353 288
                16 367 278
                17 395 271
                18 361 297
                19 373 296
                20 384 287
                ...
                """
                if id == 4:
                    # draws a white circle at the point with the id 4 (Picture 3)
                    cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
                    # we can put the cx and cy information in a list to use it in other projects

            # draws (the points and the connections of) a hand (handLms is a single hand)
            # handLms for the red points (Picture 1), mpHands.HAND_CONNECTIONS for the green connections (Picture 2)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()  # give the current time
    fps = 1/(cTime-pTime)  # frames per second
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 255,), 3)  # return fps framerate (number on top left corner)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
