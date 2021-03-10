import numpy as np
import cv2
import sys
import math
import time
writer = None
prev_x = -100
prev_y = -100
count = 0
euclidean_distance = 1

cap = cv2.VideoCapture('C:/video/1.mp4')

sub = cv2.createBackgroundSubtractorMOG2(100, 30, True)
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    mask = sub.apply(frame, learningRate=0.06)

    (contours, hierarchy) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 2500:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        if y + h / 2 > 140:
            continue

        euclidean_distance = math.sqrt((x + w / 2 - prev_x) ** 2 + (y + h / 2 - prev_y) ** 2)
        print(euclidean_distance)

        prev_y = h / 2 + y
        prev_x = w / 2 + x

        if euclidean_distance > 50:
            count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Person {}".format(count)
        cv2.putText(frame, text, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    time.sleep(0.06)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('C:/video/GMM.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

    cv2.imshow('', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()