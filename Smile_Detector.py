import cv2

# face classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
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
        cv2.putText(frame, 'Twarz', (x,y-20), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

        face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, 1.3, 5)
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(frame, (x+x_, y+y_, w_, h_), (255, 200, 50), 4)
            cv2.putText(frame, 'usmiech', (x+x_, y+y_- 20), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

        eyes = eye_detector.detectMultiScale(face_grayscale)
        for (xE, yE, wE, hE) in eyes:
            cv2.rectangle(frame, (x+xE, y+yE, wE, hE), (55, 100, 55), 4)
            cv2.putText(frame, 'oko', (x+xE, y+yE- 10), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

    #print(face)
    cv2.imshow('Why so serious?', frame)
    cv2.waitKey(1)


webcam.release()
cv2.destroyAllWindows()

print('code finished')