import cv2
import numpy as np

# load image
img = cv2.imread("court.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 1)

# edge detection
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# kernel = np.ones((3,3), np.uint8)
# edges = cv2.dilate(edges, kernel, iterations=1)

# line detection
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=200, maxLineGap=30)

# draw lines on copy
out = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

# show
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.imshow("lines", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
