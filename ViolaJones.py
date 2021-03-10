import cv2
import time

writer = None

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    person = person_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in person:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


person_cascade = cv2.CascadeClassifier('C:/haarcascade/result/cascade.xml')
cap = cv2.VideoCapture('C:/video/1.mp4')
while True:

    frame = cap.read()[1]
    if frame is None:
        break

    detectAndDisplay(frame)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('C:/video/ViolaJones2.mp4', fourcc, 20, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

    cv2.imshow('detection', frame)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
