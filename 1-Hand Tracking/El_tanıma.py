import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands

hands = mpHand.Hands()

mpDraw = mp.solutions.drawing_utils

ctime = 0
ptime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h , w , c = img.shape

                cx, cy = int(lm.x*w), int(lm.y*h)

                # bilek
                if id == 4:
                    cv2.circle(img, (cx,cy), 9, (255,0,0), -1)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = time.time()

    cv2.putText(img, "FPS: " + str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cap.release()
cv2.destroyAllWindows()










