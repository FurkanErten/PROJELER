import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]
ctime = 0
ptime = 0

while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2GRAY)

    results = hands.process(imgrgb)
    #print(results.multi_hand_landmarks)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                # # işaret uç = 8
                # if id == 8:
                #     cv2.circle(img, (cx,cy), 9, (255,0,0), -1)
                # # işaret uç = 6
                # if id == 6:
                #     cv2.circle(img, (cx,cy), 6, (0,0,255), -1)

    if len(lmList) != 0:
        fingers = []

        # bas parmak
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 parmak
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)

        totalFingers = sum(fingers)
        cv2.putText(img,str(totalFingers), (30,125), cv2.FONT_HERSHEY_PLAIN,10,(255,0,0), 8)
        # print(totalFingers)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = time.time()

    cv2.putText(img, "FPS: " + str(int(fps)), (350, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):break

cap.release()
cap.release()
cv2.destroyAllWindows()