import cv2
import mediapipe as mp
import time

if __name__=="__main__":
    capture = cv2.VideoCapture(1)

    while True:
        success, img = capture.read()

        cv2.imshow("Image", img)
        cv2.waitKey(1)
