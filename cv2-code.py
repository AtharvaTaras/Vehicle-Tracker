import cv2

# Getting video file, no path needed if its located in same folder
cap = cv2.VideoCapture("road.mp4")

while True:

    # Capturing video frame wise
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Stored height and width info inside a Tuple
    h, w, _ = frame1.shape
    # print(h, w)

    # Region of interest
    roi = frame1[0: 800, 10:600]
    #cv2.imshow("ROI", roi)

    # Greyscale Lines
    mask = cv2.absdiff(frame1, frame2, 2)
    grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Frame", frame1)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Greyscale", thresh)
    cont, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing contours
    for cnt in cont:
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 1)
            cv2.imshow("Final", frame1)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Failsafe
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
