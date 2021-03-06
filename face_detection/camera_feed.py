import cv2


def detect(frame_gray, frame, frontal_faces, profile_faces, smiles, eyes, glasses):

    frontal_faces = frontal_faces.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50), maxSize=(250, 250))
    profile_faces = profile_faces.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50), maxSize=(250, 250))
    for (x, y, w, h) in frontal_faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.putText(frame, "face front", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), thickness=2)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame_gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smiles.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            frame = cv2.putText(frame, "smile", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), thickness=1)
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        eyes = eyes.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=15, minSize=(20, 20), maxSize=(80, 80))
        for (ex, ey, ew, eh) in eyes:
            center = (ex + ew // 2, ey + eh // 2)
            frame = cv2.putText(frame, "eye".format(ew, eh), (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                thickness=1)
            frame = cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    for (x, y, w, h) in profile_faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.putText(frame, "face profile", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), thickness=2)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # faceROI = frame_gray[y:y + h, x:x + w]
    return frame


def detect_eyes(frame_gray, frame, eyes, glasses):
    eyes = eyes.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=15, minSize=(20,20), maxSize=(80,80))
    for (x, y, w, h) in eyes:
        center = (x + w // 2, y + h // 2)
        frame = cv2.putText(frame, "eye: ({},{})".format(w, h), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # faceROI = frame_gray[y:y + h, x:x + w]
    # glasses = glasses.detectMultiScale(frame_gray, minNeighbors=2, minSize=(20,20), maxSize=(100,100))
    # for (x, y, w, h) in glasses:
    #     center = (x + w // 2, y + h // 2)
    #     frame = cv2.putText(frame, "glass eye: ({},{})".format(w, h), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
    #     frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     # faceROI = frame_gray[y:y + h, x:x + w]
    return frame


def detect_smiles(frame_gray, frame, smiles):
    smiles = smiles.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=15, minSize=(20,20), maxSize=(80,80))
    for (x, y, w, h) in smiles:
        center = (x + w // 2, y + h // 2)
        frame = cv2.putText(frame, "smile", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), thickness=1)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # faceROI = frame_gray[y:y + h, x:x + w]
    return frame

def camera_feed(frontal_faces, profile_faces, smiles, eyes, glasses):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        frame = detect(frame_gray, frame, frontal_faces, profile_faces, smiles, eyes, glasses)
        # frame = detect_smiles(frame_gray, frame, smiles)
        # frame = detect_eyes(frame_gray, frame, eyes, glasses)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    vc.release()
    cv2.destroyWindow("preview")


if __name__ == "__main__":
    frontal_faces = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_alt.xml")
    profile_faces = cv2.CascadeClassifier("haar_cascades/haarcascade_profileface.xml")
    smiles = cv2.CascadeClassifier("haar_cascades/haarcascade_smile.xml")
    eyes = cv2.CascadeClassifier("haar_cascades/haarcascade_eye.xml")
    glasses = cv2.CascadeClassifier("haar_cascades/haarcascade_eye_tree_eyeglasses.xml")
    camera_feed(frontal_faces, profile_faces, smiles, eyes, glasses)
