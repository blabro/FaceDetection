import cv2

# face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
# grab webcam feed
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y, w, h), (100,200,50), 4)

        face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    smiles = smile_detector.detectMultiScale(frame_grayscale, 1.7, 20)
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y, w, h), (255, 200, 50), 4)

    #print(face)
    cv2.imshow('Why so serious?', frame)
    cv2.waitKey(1)


webcam.release()
cv2.destroyAllWindows()

print('code finished')