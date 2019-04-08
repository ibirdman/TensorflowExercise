import numpy
import cv2

# Create a black image
img = numpy.zeros((512, 512, 3), numpy.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# draw a rectangle
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# draw a circle
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

# add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Meng', (10, 500), font, 4, (255, 255, 255), 2)
# cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)   # cv2.LINE_AA not found

# write file to disk
# cv2.namedWindow("Image")

cv2.imshow("Image", img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()