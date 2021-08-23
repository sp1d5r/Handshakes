import cv2
import mediapipe as mp
import time

if __name__=="__main__":
    capture = cv2.VideoCapture(0)

    # We can use the Hand Detection Model from media pipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = capture.read()

        # Data Preprocessing - Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Applying the model to the image
        results = hands.process(img_rgb)

        # Extract out the information for the hands
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
