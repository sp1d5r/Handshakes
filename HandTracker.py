import cv2
import mediapipe as mp
import time

if __name__=="__main__":
    capture = cv2.VideoCapture(0)

    # We can use the Hand Detection Model from media pipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mpDraw = mp.solutions.drawing_utils

    # Adding a frame rate
    prev_time = 0
    curr_time = 0

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

        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, "the fps:" + str(int(fps)),
                    (10,70),
                    cv2.FONT_ITALIC,
                    3,
                    (255, 255, 0),
                    2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
