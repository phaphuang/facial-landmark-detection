# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
from PIL import Image

from cv2 import WINDOW_NORMAL

width = 800
height = 600

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to input video")
#ap.add_argument("-n", "--num-frames", type=int, default=2000,
#    help="# of frames to loop over for FPS test")
args = vars(ap.parse_args())

#initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# load the input image, resize it, and convert it to grayscale
#cap = cv2.VideoCapture(args["video"])
print("[INFO] starting video file thread...")
cap = cv2.VideoCapture(args["video"])
#time.sleep(1)

# start the FPS timer
fps = FPS().start()

window_name = "cctv"
cv2.namedWindow(window_name, WINDOW_NORMAL)
cv2.resizeWindow(window_name, width, height)

if cap.isOpened():
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Assign new height and width for ROI
    (xr, yr, wr, hr) = cv2.boundingRect(gray)
    startYROI = 2*hr/5
    endYROI = hr-2*hr/5
else:
    print "Camera is not open"

while ret:
    cv2.rectangle(image, (0, startYROI), (wr, endYROI), (0, 255, 0), 2)
    roi_image = image[startYROI:endYROI, 0:wr]
    #image = cv2.imread(args["image"])
    #image = cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)
    #image_scaled = cv2.resize(image, (600, 500), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    #pil_gray = Image.fromarray(gray)
    #pil_gray = pil_gray.resize((width, height), Image.BICUBIC)
    #cv_gray = np.array(pil_gray)

    #roi = image[]

    # detect faces in the grayscale image
    rects = detector(gray, 0)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, (y + startYROI)), (x + w, (y + startYROI) + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y + startYROI - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, (y + startYROI)), 1, (0, 0, 255), -1)
    # show the output image with the face detections + facial landmarks
    cv2.imshow(window_name, image)
    ret, image = cap.read()

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
