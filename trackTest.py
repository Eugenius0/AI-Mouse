import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0  # previous Time
cTime = 0  # current Time
cap = cv2.VideoCapture(0)  # create video object, (1) if you have multiple cams
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if len(lmList) != 0:  # check that list is not empty
        print(lmList[4])  # return list of position where index is 4 in this format: [id, cx, cy]

    cTime = time.time()  # give the current time
    fps = 1 / (cTime - pTime)  # frames per second
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255,), 3)  # return fps framerate (number on top left corner)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
