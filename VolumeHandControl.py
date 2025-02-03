import time
import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###################################
wCam, hCam = 640, 480
###################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(detectioncon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # Get coordinates of thumb (landmark 4) and index finger (landmark 8)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        # Compute distance between thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Update volume based on distance using the full hand range (50 - 300)
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        vol = minVol + (maxVol - minVol) * (np.log10(volPer + 1) / np.log10(101))
        # Debug print: print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        # Visual feedback when fingers are very close together
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 2)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (30, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
