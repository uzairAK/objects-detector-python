import cv2, time
import numpy as np
video = cv2.VideoCapture(1)

first_frame = None;
status_list =[]
while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0 ) # Blur image to reduce noise and increase accuracy

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray) # Comparing blurray scale images
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY) [1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (_,cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    for contour in cnts:
        if cv2.contourArea(contour) < 220:
            status=0
            continue
        status=1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3) # Drawing rectangle over objects
        # cv2.circle(frame, (x,y), (h), (0,255,0),3) # Drawing circle over objects

    status_list.append(status)
    cv2.imshow("Gray frame", gray)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Threshold frame", thresh_frame)
    cv2.imshow("Color frame", frame)
    key= cv2.waitKey(1)

    if key == ord("q"):
        break

print(status_list)

video.release()
cv2.destroyAllWindows()
