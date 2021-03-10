import imutils
import cv2
import math
import time

prev_x = -100
prev_y = -100
count = 0
writer = None
euclidean_distance = 1

cap = cv2.VideoCapture('C:/video/1.mp4')
first_frame = cap.read()[1]
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

while cap.isOpened():
    frame = cap.read()[1]
    if frame is None:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    delta = cv2.absdiff(first_gray, gray_frame)
    delta = cv2.threshold(delta, 72, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        if cv2.contourArea(c) < 3400:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        if y + h/2 > 220:
            continue

        euclidean_distance = math.sqrt((x + w/2 - prev_x)**2 + (y + h/2 - prev_y)**2)
        print(x + w/2, y + h/2, euclidean_distance)

        prev_y = h/2 + y
        prev_x = w/2 + x

        if euclidean_distance > 50:
            count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Person {}".format(count)
        cv2.putText(frame, text, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    time.sleep(0.06)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter('C:/video/output_1.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

    cv2.imshow("input", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
