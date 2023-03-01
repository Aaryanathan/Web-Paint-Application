import numpy as np
import cv2
from collections import deque

blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

kernel = np.ones((5, 5), np.uint8)

bPoints = [deque(maxlen=512)]
gPoints = [deque(maxlen=512)]
rPoints = [deque(maxlen=512)]
yPoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorindex = 0

paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)
cv2.putText(paintWindow, 'Clear All', (49, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, 'Blue', (185, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, 'Green', (298, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, 'Red', (420, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, 'Yellow', (520, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

camera = cv2.VideoCapture(0)

while True:
    (hasframe, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (120, 120, 120), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)
    cv2.putText(frame, 'Clear All', (49, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Blue', (185, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Green', (298, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Red', (420, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Yellow', (520, 33), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    if not hasframe:
        break
    blue_mask = cv2.inRange(hsv,blueLower, blueUpper)
    blue_mask = cv2.erode(blue_mask,kernel,iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
    (cnts, _) = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        if center[1] <= 65:
            if 40 <= center[0] <= 140:
                bPoints = [deque(maxlen=512)]
                gPoints = [deque(maxlen=512)]
                rPoints = [deque(maxlen=512)]
                yPoints = [deque(maxlen=512)]
                bindex = 0
                gindex = 0
                rindex = 0
                yindex = 0
                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                colorindex = 0
            elif 275 <= center[0] <= 370:
                colorindex = 1
            elif 390 <= center[0] <= 495:
                colorindex = 2
            elif 505 <= center[0] <= 600:
                colorindex = 3
        else:
            if colorindex == 0:
                bPoints[bindex].appendleft(center)
            elif colorindex == 1:
                gPoints[gindex].appendleft(center)
            elif colorindex == 2:
                rPoints[rindex].appendleft(center)
            elif colorindex == 3:
                yPoints[yindex].appendleft(center)
    else:
        bPoints.append(deque(maxlen=512))
        bindex += 1
        gPoints.append(deque(maxlen=512))
        gindex += 1
        rPoints.append(deque(maxlen=512))
        rindex += 1
        yPoints.append(deque(maxlen=512))
        yindex += 1
    points = [bPoints, gPoints, rPoints, yPoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
    cv2.imshow('Camera',frame)
    cv2.imshow('Paint', paintWindow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()