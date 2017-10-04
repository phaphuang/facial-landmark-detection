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
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
else:
    print "no video found!."

while ret:
    #ret, frame = cap.read()
    #image = cv2.imread(args["image"])
    #image = cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)
    #image_scaled = cv2.resize(image, (400, 300), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #pil_gray = Image.fromarray(gray)
    #pil_gray = pil_gray.resize((width, height), Image.BICUBIC)
    #cv_gray = np.array(pil_gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    image_copy = image.copy()

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        print dlib_rect

        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        # copying the image so we can see side-by-side

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            # annotate the positions
            cv2.putText(image_copy, str(idx), pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(0, 0, 255))

            # draw points on the landmark positions
            cv2.circle(image_copy, pos, 3, color=(0, 255, 255))

    # show the output image with the face detections + facial landmarks
    cv2.imshow(window_name, image_copy)
    #cv2.imshow("Landmarks found", image_copy)

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
