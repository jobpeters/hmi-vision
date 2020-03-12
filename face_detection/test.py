import cv2 as cv2
import numpy as np

def fetch_files():
    classifiers = [
        "face_haar_cascades/haarcascade_frontalface_alt.xml",
        "face_haar_cascades/haarcascade_frontalface_alt2.xml",
        "face_haar_cascades/haarcascade_frontalface_alt_tree.xml",
        "face_haar_cascades/haarcascade_frontalface_default.xml"
    ]
    face_cascades = {classifier: cv2.CascadeClassifier(classifier) for classifier in classifiers}
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread('group2013.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascades, img
# def process_im(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
#     return gray


def face_detection(frame, classifier_name, classifier):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = classifier.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        faceROI = frame_gray[y:y+h,x:x+w]
    # cv2.imshow('{} - Face detection'.format(classifier_name.replace("face_haar_cascades/", "").replace(".xml", "")), frame)
    cv2.imwrite("detected_faces/{}.png".format(classifier_name.replace("face_haar_cascades/", "").replace(".xml", "")), frame)

if __name__ == "__main__":
    face_cascades, img = fetch_files()
    # gray = process_im(img)
    # cv2.imshow('color', img)
    # cv2.imshow('gray', gray)
    # for face_cascade in face_cascades:
    #     face_detection(img, face_cascade)
    # face_detection(img, face_cascades[0])
    # cv2.waitKey(0)
    for key, value in face_cascades.items():
        face_detection(img, key, value)
        cv2.waitKey(0)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x, y, w, h) in faces:
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = img[y:y + h, x:x + w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    # cv2.imshow('image', img)

    cv2.destroyAllWindows()

