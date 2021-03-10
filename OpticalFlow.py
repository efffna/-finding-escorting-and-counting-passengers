import numpy as np
import cv2
import math
import time

writer = None
count = 0
t = 0
prev_x = -100
prev_y = -100
data_x = []
data_y = []

cap = cv2.VideoCapture('C:/video/1.mp4')
first_frame = cap.read()[1]
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(first_gray,
                             maxCorners=60,
                             qualityLevel=0.03,
                             minDistance=14,
                             blockSize=3)
mask = np.zeros_like(first_frame)
while cap.isOpened:
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    frame = cap.read()[1]
    if frame is None:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, status, err = cv2.calcOpticalFlowPyrLK(prevImg=first_gray,
                                               nextImg=frame_gray,
                                               prevPts=p0,
                                               nextPts=None,
                                               winSize=(55, 30),
                                               maxLevel=2)

    good_old = p0[status == 1]
    good_new = p1[status == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        euclidean_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        if euclidean_distance > 4 and y1 > 50 and y1<230:
            data_x.append(x1)
            data_y.append(y1)
            frame = cv2.circle(frame, (x1, y1), 2, (0, 0, 255), -1)

    if data_x != [] and data_y != []:
        min_x = min(data_x)
        max_x = max(data_x)
        min_y = min(data_y)
        max_y = max(data_y)
        x_h = min_x + (max_x - min_x)
        y_w = min_y + (max_y - min_y)
        area = ((max_x - min_x) * (max_y - min_y))

        # if 200 < x_h < 350 and 170 < y_w < 300 :
        if 6000 < area < 12000:
            cv2.rectangle(frame, (min_x, min_y), (x_h, y_w), (0, 255, 0), 2)
            euclidean_distance1 = math.sqrt((min_x + x_h / 2 - prev_x) ** 2 + (min_y + y_w / 2 - prev_y) ** 2)
            prev_y = y_w / 2 + min_y
            prev_x = x_h / 2 + min_x
            print(euclidean_distance1)
            if euclidean_distance1 > 70:
                count += 1

            text = "Person {}".format(count)
            cv2.putText(frame, text, (int(min_x) - 5, int(min_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    time.sleep(0.1)
    cv2.imshow('OpticalFlow', frame)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('C:/video/OpticalFlow.mp4', fourcc, 15, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    data_x.clear()
    data_y.clear()
    first_gray = frame_gray.copy()

    p0 = cv2.goodFeaturesToTrack(first_gray,
                                 maxCorners=64,
                                 qualityLevel=0.03,
                                 minDistance=14,
                                 blockSize=3)

writer.release()
cv2.destroyAllWindows()
cap.release()
