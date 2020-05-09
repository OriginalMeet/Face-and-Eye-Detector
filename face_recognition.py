# importing the opencv2 library
import cv2

# loading the cascades one for face and one for eyes
# the xml file contains cascades for face and eyes.
# first object for face harr cascade and second object for eye harr cascade

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# defining the functions that will do the detections


def detect(gray, frame):
    # grey is BW image and frame is the original image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # we are detecting eyes in the region of intrest of face means inside the face frame.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame


# doing the face recognition using the webcame
video_capture = cv2.VideoCapture(0)  # use 1 for external webcame

while True:
    _, frame = video_capture.read()  # gets us the last frame of the webcame
    # converting image to the B/W
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # storing the output of the detect function
    canvas = detect(gray, frame)
    # display all the output in one window.
    cv2.imshow('Video', canvas)
    # stop the code when we press q or interupt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()  # to turn off the camera
cv2.destroyAllWindows()  # to close the window.
