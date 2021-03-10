import imutils
import cv2
import math
import time
import numpy as np


prev_x = -100
prev_y = -100
count = 0
writer = None
euclidean_distance = 1

cap = cv2.VideoCapture('C:/video/1.mp4')


first_frame = cap.read()[1]

first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first = cv2.medianBlur(first, 5)
edges = cv2.adaptiveThreshold(first, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
color = cv2.bilateralFilter(first_frame, 9, 250, 250)
first_cartoon = cv2.bitwise_and(color, color, mask=edges)

while cap.isOpened():
    frame = cap.read()[1]
    if frame is None:
        break

    gr = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gr = cv2.medianBlur(gr, 5)
    edges_frame = cv2.adaptiveThreshold(gr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color_frame = cv2.bilateralFilter(frame, 9, 250, 250)
    cartoon_frame = cv2.bitwise_and(color_frame, color_frame, mask=edges_frame)
    delta = cv2.absdiff(first_cartoon, cartoon_frame)

    light = np.array([80, 50, 18])
    dark = np.array([255, 255, 255])
    frame_threshold = cv2.inRange(delta, light, dark)
    contours = cv2.findContours(frame_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        if cv2.contourArea(c) < 3500:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        if y + h / 2 > 220:
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
        writer = cv2.VideoWriter('C:/video/cartoon.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
